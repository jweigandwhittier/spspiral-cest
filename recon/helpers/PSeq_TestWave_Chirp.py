import numpy as np
import pypulseq as pp
from scipy.signal import windows

from .PSeq_TestWave import PSeq_TestWave


def get_chirp(dt, t_chirp, f1, f2, max_krange=None, max_kmax=None, gmax=30e-3, smax=100):
    """
    Design a chirp waveform subject to moment constraints.

    Parameters
    ----------
    dt : float
        Raster time [seconds]
    t_chirp : float
        Duration of the chirp [seconds]
    f1 : float
        Starting frequency [Hz]
    f2 : float
        Ending frequency [Hz]
    max_krange : float, optional
        Maximum range of M0 [1/m]
    max_kmax : int, optional
        Maximum absolute M0 [1/m]
    gmax : float, optional
        Maximum gradient strength [T/m], by default 30e-3
    smax : int, optional
        Maximum slew rate [T/m/s], by default 100

    Returns
    -------
    ndarray
        The chirp waveform [T/m] with raster time dt
    """
    # doi: 10.1002/mrm.23217
    tt = np.arange(0, t_chirp, dt)
    ft = f1 + (f2 - f1) * tt / t_chirp

    Gct = gmax * np.sin(2 * np.pi * (f1 * tt + (f2 - f1) * tt**2 / 2 / t_chirp))

    senv = smax / (2 * np.pi * gmax * ft + 1e-16)
    senv[senv > 1] = 1

    blip = senv * Gct

    m0_min, m0_max = get_m0_range(blip, dt)
    if max_kmax is not None:
        if m0_max <= max_kmax:
            np.hstack([blip, 0])
        else:
            return get_chirp(
                dt,
                t_chirp,
                f1,
                f2,
                max_krange,
                max_kmax,
                0.99 * max_kmax / m0_max * gmax,
                smax,
            )
    if max_krange is not None:
        if (m0_max - m0_min) <= max_krange:
            np.hstack([blip, 0])
        else:
            return get_chirp(
                dt,
                t_chirp,
                f1,
                f2,
                max_krange,
                max_kmax,
                0.99 * max_krange / (m0_max - m0_min) * gmax,
                smax,
            )

    return np.hstack([blip, 0])


def trap_balance(wave, dt, gmax=30e-3, smax=100, delay=0.5e-3, gamma=42576000):
    """Add a trapezoid to the start of the waveform to balance M0 min and max."""
    m0_min, m0_max = get_m0_range(wave, dt)
    m0_balance = (m0_max - m0_min) / 2 + m0_min

    g_temp = pp.make_trapezoid(
        channel='z',
        area=-m0_balance,
        max_grad=gamma * gmax,
        max_slew=gamma * smax,
    )

    ramp_N = int(np.ceil(g_temp.rise_time / dt) + 1)
    flat_N = int(np.ceil(g_temp.flat_time / dt))
    amp = g_temp.amplitude / gamma

    refocus = [np.linspace(0, amp, ramp_N), amp * np.ones(flat_N), np.linspace(amp, 0, ramp_N)]
    refocus = np.hstack(refocus)

    return np.hstack([refocus, np.zeros(int(delay / dt)), wave])


def kais_balance(wave, dt, duration=2e-3, delay=0.5e-3, gamma=42576000):
    """Add a Kaiser window to the start of the waveform to balance M0 min and max."""
    m0_min, m0_max = get_m0_range(wave, dt)
    m0_balance = (m0_max - m0_min) / 2 + m0_min

    ww = windows.kaiser(int(duration / dt), 14)
    ww[0] = 0
    ww[-1] = 0

    scale = -m0_balance / (gamma * dt * np.sum(ww))
    ww *= scale

    return np.hstack([ww, np.zeros(int(delay / dt)), wave])


def get_m0_range(wave, dt, gamma=42576000):
    """Get the minimum and maximum M0 of a waveform [1/m]."""
    m0 = gamma * dt * np.cumsum(wave)
    return m0.min(), m0.max()


def prep_all_chirps(
    thickness=1e-3,
    dt=10e-6,
    gmax=30e-3,
    smax=150,
    all_f2=(5e3, 15e3),
    all_t_chirp=(20e-3, 40e-3),
):
    """
    Compute all chirp waveforms for testing.
    
    Currently a lot is hardcoded here, this all needs to eventually be re-worked (TODO).
    """
    gamma = 42576000
    f1 = 0

    all_chirps = []
    for f2 in all_f2:
        for t_chirp in all_t_chirp:
            chirp = get_chirp(dt, t_chirp, f1, f2, gmax=gmax, smax=smax, max_kmax=0.3 / thickness)

            chirp_r = get_chirp(
                dt,
                t_chirp,
                f1,
                f2,
                gmax=gmax,
                smax=smax,
                max_krange=0.6 / thickness,
            )
            chirp2 = trap_balance(chirp_r, dt, delay=0.5e-3, gmax=0.9 * gmax, smax=0.9 * smax)
            chirp3 = trap_balance(chirp_r, dt, delay=2e-3, gmax=0.9 * gmax, smax=0.9 * smax)

            areas = [gamma * dt * chirp.sum(), gamma * dt * chirp2.sum(), gamma * dt * chirp3.sum()]

            all_chirps.append(
                {
                    'f2': f2,
                    't_chirp': t_chirp,
                    'mode': 0,
                    'chirp': chirp,
                    'area': areas[0],
                },
            )

            all_chirps.append(
                {
                    'f2': f2,
                    't_chirp': t_chirp,
                    'mode': 1,
                    'chirp': chirp2,
                    'area': areas[1],
                },
            )

            all_chirps.append(
                {
                    'f2': f2,
                    't_chirp': t_chirp,
                    'mode': 2,
                    'chirp': chirp3,
                    'area': areas[2],
                },
            )

    return all_chirps


class PSeq_TestWave_Chirp(PSeq_TestWave):
    """Chirp test waveforms."""

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

        Notes
        -----
        A lot of this is hard coded for initial testing, this needs to be much more
        customizeable and a final design strategy selected after testing how the different
        chirps perform.
        """
        self.wave_delay = wave_delay

        super().__init__(pparams, *args, **kwargs)

    def prep_waves(self):
        """Build up all of the chirp waveforms."""
        self.all_test_waves = []
        self.all_test_waves_neg = []
        self.all_areas = []

        self.all_chirps = prep_all_chirps(
            thickness=1e-3,
            dt=self.pparams.system.grad_raster_time,
            gmax=0.98 * self.pparams.system.max_grad / self.pparams.system.gamma,
            smax=0.98 * self.slew / self.pparams.system.gamma,
        )

        self.N_waves = len(self.all_chirps)

        self.all_test_waves = []
        self.all_test_waves_neg = []
        for i in range(self.N_waves):
            wave = pp.make_arbitrary_grad(
                channel=self.pparams.channels[2],
                waveform=self.pparams.system.gamma * self.all_chirps[i]['chirp'],
                delay=self.wave_delay,
                first=0,
                last=0,
                system=self.pparams.system,
            )
            self.all_test_waves.append(wave)

            self.all_areas.append(self.all_chirps[i]['area'])

            wave = pp.make_arbitrary_grad(
                channel=self.pparams.channels[2],
                waveform=-self.pparams.system.gamma * self.all_chirps[i]['chirp'],
                delay=self.wave_delay,
                first=0,
                last=0,
                system=self.pparams.system,
            )
            self.all_test_waves_neg.append(wave)
