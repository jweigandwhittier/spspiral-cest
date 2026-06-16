import numpy as np
import pypulseq as pp

from .PSeq_Base import PSeq_Base

class PSeq_Excite_PE(PSeq_Base):
    """
    Sequence component that plays slice excitation and optional phase encoding

    Thiss is made for GIRF measurements, so the FOV is sassumed to be square.
    """

    def __init__(
        self,
        pparams,
        duration = 6e-3, thickness = 1e-3, 
                 tbw = 4, flip = 50, app = 0.4, 
                 fov = 200e-3, N_pe = 0,
                 max_slew = 100,
                 do_refocus = True,
                 do_prewind = True,
                 *args, **kwargs
    ):
        """
        Construct.

        Parameters
        ----------

        """
        super().__init__(pparams)
        
        self.duration = duration
        self.do_refocus = do_refocus
        self.do_prewind = do_prewind
        self.fov = fov
        self.N_pe = N_pe
        self.thickness = thickness
        self.tbw = tbw
        if flip > np.pi:  # Assume anything higher than pi is in degrees
            flip = flip * np.pi / 180
        self.flip = flip
        self.app = app
        
        if max_slew is not None:
            self.slew = self.pparams.system.gamma*max_slew
        else:
            self.slew = self.pparams.system.max_slew

        # Get RF, slice select and refocus/prewinding gradient (assumes RF is coming out symmetric for prewinder)
        self.rfp, self.gss, self.gss_re = pp.make_sinc_pulse(flip_angle=self.flip, apodization=self.app, duration=self.duration, 
                                                system=self.pparams.system, time_bw_product=self.tbw, delay=self.pparams.system.rf_dead_time,
                                                slice_thickness=self.thickness, return_gz=True, max_slew = self.slew)
        
        # Figure out the longest time needed between refocusing and phase encoding
        self.refocus_time = 0
        if N_pe > 0:
            self.pe_areas = (np.arange(N_pe) - N_pe//2)/fov
            self.max_area = np.max(np.abs(self.pe_areas))  # This gets used to rescale amplitude
            pe_temp = pp.make_trapezoid(channel = self.pparams.channels[0], area=self.max_area, system=self.pparams.system)
            self.refocus_time = pp.calc_duration(pe_temp)
        if do_refocus:
            if pp.calc_duration(self.gss_re) >= self.refocus_time:
                self.refocus_time = pp.calc_duration(self.gss_re)
            else:
                # Remake gss_re to match longer phase encode time
                self.gss_re = pp.make_trapezoid(channel = self.pparams.channels[2], duration = self.refocus_time, 
                                                area=self.gss_re.area, system=self.pparams.system)
           
        # Set up final phase encode gradients if needed     
        if N_pe > 0:
            self.pe_grad0 = pp.make_trapezoid(channel = self.pparams.channels[0], duration = self.refocus_time, 
                                                    area=self.max_area, system=self.pparams.system)        
            
            self.pe_grad1 = pp.make_trapezoid(channel = self.pparams.channels[1], duration = self.refocus_time, 
                                                    area=self.max_area, system=self.pparams.system)  
            
            self.amp0 = self.pe_grad0.amplitude
            self.amp1 = self.pe_grad1.amplitude

    def build_blocks(
        self,
        idx0=0,
        idx1=0,
        offset=0
    ):
        """Build all blocks to add to sequence.

        Parameters
        ----------
        idx0, idx1 : int
            Index for the 0th channel area.  Will be ignored if the areas given for this
            channel were not inputted with multiple values.
            
        offset: float
            Slice offset [m]

        Returns
        -------
        list of lists
            All blocks to add, outer list is for sequential blocks, inner blocks are all
            components to play within the block.
        """
        self.gss.channel = self.pparams.channels[2]
        self.rfp.freq_offset = self.gss.amplitude * offset
        
        all_blocks = []
        
        # --- Slice select prewinder
        if self.do_prewind:
            self.gss_re.channel = self.pparams.channels[2]
            all_blocks.append([self.gss_re])
            
        # --- RF and slice select gradient
        all_blocks.append([self.rfp, self.gss])
        
        # --- Spatial encoding and slice select refocusing
        _blocks = []
        
        if self.N_pe > 0:
            area0 = self.pe_areas[idx0]
            area1 = self.pe_areas[idx1]
            
            self.pe_grad0.channel = self.pparams.channels[0]
            self.pe_grad1.channel = self.pparams.channels[1]
            
            self.pe_grad0.amplitude = self.amp0 * area0/self.max_area
            self.pe_grad1.amplitude = self.amp1 * area1/self.max_area
            
            _blocks.append(self.pe_grad0)
            _blocks.append(self.pe_grad1)
            
            
        if self.do_refocus:
            self.gss_re.channel = self.pparams.channels[2]
            _blocks.append(self.gss_re)
            
        all_blocks.append(_blocks)
        return all_blocks
