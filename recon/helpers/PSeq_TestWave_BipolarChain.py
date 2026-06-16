from copy import deepcopy

import numpy as np
import pypulseq as pp

from .PSeq_TestWave import PSeq_TestWave


class PSeq_TestWave_BipolarChain(PSeq_TestWave):
    """
    Bipolar chain test waveforms.
    
    This is just a quick test of this test waveform, so everything is currently hard-coded.
    """

    def __init__(
        self,
        pparams,
        wave_delay=2e-3,
        *args,
        **kwargs,
    ):
        """Construct.

        Parameters
        ----------
        pseq : Pseq_Base derived, optional
            Another Base sequence or sequence component to copy parameters from, by
            default None
        wave_delay : float, optional
            Delay time [seconds] before playing out the waveform, to allow some sampling
            of the ADC before any gradients are played out, by default 2e-3
        """
        self.wave_delay = wave_delay
        self.N_waves = 12

        super().__init__(pparams, *args, **kwargs)

    def prep_waves(self):
        """Build up all of the bipolar waveforms."""
        self.all_test_waves = []
        self.all_test_waves_neg = []
        self.all_areas = []

        for area in [300, 500]:
            _grad = pp.make_trapezoid(
                    channel=self.pparams.channels[2],
                    area=area,
                    delay=0,
                    system=self.pparams.system,
                    max_slew=self.slew
                )
            
            for gap_time in np.linspace(4.5e-3, 7e-3, 6):
                all_grad = []
                start_time = self.wave_delay
                
                for _ in range(4):
                    # Positive Lobe
                    _grad.delay = start_time
                    all_grad.append(deepcopy(_grad))
                    start_time = pp.calc_duration(_grad)

                    # Negative Lobe
                    _grad.amplitude = -_grad.amplitude
                    _grad.delay = start_time
                    all_grad.append(deepcopy(_grad))
                    start_time = pp.calc_duration(_grad)
                    
                    # Gap
                    start_time += gap_time
                    
                    # Negative Lobe after gap
                    _grad.delay = start_time
                    all_grad.append(deepcopy(_grad))
                    start_time = pp.calc_duration(_grad)
                    
                    # Positive Lobe after gap
                    _grad.amplitude = -_grad.amplitude
                    _grad.delay = start_time
                    all_grad.append(deepcopy(_grad))
                    start_time = pp.calc_duration(_grad)
                    
                    # Gap
                    start_time += gap_time
               
                
                all_grad = pp.add_gradients(all_grad, system=self.pparams.system)
                
                self.all_test_waves.append(all_grad)
                
                neg_grad = deepcopy(all_grad)
                neg_grad.waveform *= -1
                self.all_test_waves_neg.append(neg_grad)
                
                self.all_areas.append(0)
                
