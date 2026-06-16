import pypulseq as pp

from .PSeq_TestWave import PSeq_TestWave


class PSeq_TestWave_Blip(PSeq_TestWave):
    """Blip test waveforms."""

    def __init__(
        self,
        pparams,
        min_ramp=60e-6,
        ramp_inc=10e-6,
        wave_delay=2e-3,
        N_waves=12,
        *args,
        **kwargs,
    ):
        """Construct.

        Parameters
        ----------
        pseq : Pseq_Base derived, optional
            Another Base sequence or sequence component to copy parameters from, by
            default None
        min_ramp : float, optional
            Ramp time [seconds] for the smallest blip, by default 60e-6
        ramp_inc : float, optional
            Time [seconds] to increase each subsequent ramp time in N_waves, by default 10e-6
        wave_delay : float, optional
            Delay time [seconds] before playing out the waveform, to allow some sampling
            of the ADC before any gradients are played out, by default 2e-3
        N_waves : int, optional
            Nuber of waveforms to generate, by default 12
        """
        self.wave_delay = wave_delay
        self.N_waves = N_waves
        self.min_ramp = min_ramp
        self.ramp_inc = ramp_inc

        super().__init__(pparams, *args, **kwargs)

    def prep_waves(self):
        """Build up all of the blip waveforms."""
        self.all_test_waves = []
        self.all_test_waves_neg = []
        self.all_areas = []

        for i in range(self.N_waves):
            ramp_time = self.min_ramp + i * self.ramp_inc

            # .99999 due to float rounding errors sometimes exceeding slew rate limit
            amp = 0.99999 * self.slew * ramp_time

            wave = pp.make_trapezoid(
                channel=self.pparams.channels[2],
                flat_time=0,
                amplitude=amp,
                rise_time=ramp_time,
                fall_time=ramp_time,
                delay=self.wave_delay,
                system=self.pparams.system,
            )
            self.all_test_waves.append(wave)

            self.all_areas.append(wave.area)

            wave = pp.make_trapezoid(
                channel=self.pparams.channels[2],
                flat_time=0,
                amplitude=-amp,
                rise_time=ramp_time,
                fall_time=ramp_time,
                delay=self.wave_delay,
                system=self.pparams.system,
            )
            self.all_test_waves_neg.append(wave)
