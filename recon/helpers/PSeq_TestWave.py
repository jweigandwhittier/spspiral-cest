import pypulseq as pp
from pypulseq.make_adc import calc_adc_segments

from .PSeq_Base import PSeq_Base


class PSeq_TestWave(PSeq_Base):
    """
    Base class for GIRF measurement test waves.

    Supports thin slice and field camera measurements.  Derived classes should implement
    prep_waves() to create the lists: all_test_waves, all_test_waves_neg, and
    all_areas.
    """

    def __init__(
        self,
        pparams,
        do_adc=True,
        dt_adc=4e-6,
        N_adc=20000,
        total_duration=100e-3,
        adc_delay=1e-3,
        slew=None,
        *args,
        **kwargs,
    ):
        """
        Construct.

        Parameters
        ----------
        pseq : Pseq_Base derived, optional
            Another Base sequence or sequence component to copy parameters from, by
            default None
        do_adc : bool, optional
            Should the ADC be played out, False for Skope measurements, True for thin
            slice, by default True
        dt_adc : float, optional
            dwell time of ADC in [seconds], by default 4e-6
        N_adc : int, optional
            number of points in ADC, by default 20000
        total_duration : _type_, optional
            How long should this component be in [seconds], intended to get more
            consistent timings between differnet length test waves, by default 100e-3
        adc_delay : float, optional
            Delay time for the ADC, in [seconds], by default 1e-3
        slew : float, optional
            Slew rate override, in [T/m/s], if None keep using the current system setting,
            by default None
        """
        super().__init__(pparams)

        self.adc_delay = adc_delay

        self.dt_adc = dt_adc
        self.N_adc = N_adc

        self.do_adc = do_adc
        self.total_duration = total_duration

        # Override slew rate if defined in argument
        if slew is not None:
            self.slew = self.pparams.system.gamma * slew
        else:
            self.slew = self.pparams.system.max_slew

        if self.do_adc:
            if self.N_adc > self.pparams.system.adc_samples_limit:
                self.adc_segments, self.adc_samples_seg = calc_adc_segments(
                    self.N_adc,
                    self.dt_adc,
                    self.pparams.system,
                )
                self.N_adc = self.adc_segments * self.adc_samples_seg
            else:
                self.adc_segments, self.adc_samples_seg = (0, 0)

            self.adc = pp.make_adc(
                num_samples=self.N_adc,
                dwell=self.dt_adc,
                delay=self.adc_delay,
                system=self.pparams.system,
            )

            if pp.calc_duration(self.adc) > self.total_duration:
                print(
                    'WARNING: ADC duration longer than total_duration, this is probably unintended',
                )

        self.duration_delay = pp.make_delay(self.total_duration)

        self.prep_waves()

    def prep_waves(self):
        """Abstract base class for deriving the test waveforms."""
        raise NotImplementedError

    def build_blocks(self, idx=0, polarity=1):
        """Buld sequence blocks.

        Parameters
        ----------
        idx : int, optional
            Index of the test waves to play.
            By default 0
        polarity : int, optional
            Polarity of the test wave to play, either -1, 0, or 1.  0 does not play anything.
            By default 1

        Returns
        -------
        list of lists
            All blocks to add, outer list is for sequential blocks, inner blocks are all
            components to play within the block.
        """            
        if self.adc_delay > self.wave_delay:
            print('WARNING: wave_delay is shorter than adc_delay, gradient may be starting before the ADC does.')  # noqa: E501
            
        blocks_to_play = []

        if polarity == 1:
            self.all_test_waves[idx].channel = self.pparams.channels[2]
            blocks_to_play.append(self.all_test_waves[idx])
        elif polarity == -1:
            self.all_test_waves_neg[idx].channel = self.pparams.channels[2]
            blocks_to_play.append(self.all_test_waves_neg[idx])

        if self.do_adc:
            blocks_to_play.append(self.adc)

        blocks_to_play.append(self.duration_delay)

        return [blocks_to_play]
