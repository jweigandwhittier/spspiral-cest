"""
Adiabatic T1prep / inversion block & non-adiabatic composite T2prep block.
Taken from cMRF repository (https://github.com/PTB-MR/cMRF/blob/main/utils/preparation_blocks.py)
"""

import numpy as np
import pypulseq as pp
from typing import Literal # JWW added this import


def _add_composite_refocusing_block(
    system: pp.Opts,
    duration_180: float,
    rf_gap_time: float,
    seq: pp.Sequence | None = None,
    negative_amp: bool = False,
) -> tuple[pp.Sequence, float]:
    """Add a 90°x, +/-180°y, 90°x refocusing block to a sequence.

    Parameters
    ----------
    system
        system limits
    duration_180
        duration of 180° refocussing block puls
    rf_gap_time
        time between 2 consecutive RF pulses
    seq
        PyPulseq Sequence object
    negative_amp
        toggles negative amplitude for 180°y pulse

    Returns
    -------
    seq
        PyPulseq Sequence object
    duration
        duration of the composite refocusing block in seconds
    """
    if not seq:
        seq = pp.Sequence()

    flip_angles = [90, 180, 90]
    durations = [duration_180 / 2, duration_180, duration_180 / 2]
    if not negative_amp:
        phases = [0, 90, 0]
    else:
        phases = [180, 270, 180]

    for n, (fa, phase, dur) in enumerate(zip(flip_angles, phases, durations, strict=True)):
        rf = pp.make_block_pulse(
            flip_angle=fa * np.pi / 180, phase_offset=phase * np.pi / 180, duration=dur, system=system
        )
        seq.add_block(rf)
        if n < len(flip_angles) - 1:
            seq.add_block(pp.make_delay(rf_gap_time))

    total_dur = 2 * duration_180 / 2 + duration_180 + 2 * rf_gap_time

    return (seq, total_dur)


def add_t1prep(
    seq: pp.Sequence | None = None,
    inversion_time: float = 21e-3,
    rf_duration: float = 10.24e-3,
    spoil_duration: float = 9.6e-3,
    spoil_ramp_time: float = 7e-4,
    system: pp.Opts | None = None,
) -> tuple[pp.Sequence, float]:
    """Add an adiabatic T1 preparation block to a sequence.

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    inversion_time
        inversion time in seconds
    rf_duration
        duration of the inversion pulse
    spoil_duration
        duration of the spoiler gradient
    spoil_ramp_time
        duration of the gradient spoiling ramp
    system
        system limits


    Returns
    -------
    seq
        PyPulseq Sequence object
    total_duration
        total duration of the T2 preparation block in seconds
    """
    if not seq:
        seq = pp.Sequence()

    if not system:
        system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=100, slew_unit="T/m/s")

    # create adiabatic hyperbolic secant inversion pulse
    rf = pp.make_adiabatic_pulse(
        pulse_type="hypsec",
        adiabaticity=6,
        beta=800,
        mu=4.9,
        duration=rf_duration,
        system=system,
        use="inversion",
    )

    # create spoiler gradient
    gz_spoil = pp.make_trapezoid(
        channel="z",
        amplitude=0.5 * system.max_grad,
        duration=spoil_duration,
        rise_time=spoil_ramp_time,
    )

    # calculate inversion time delay
    time_delay = inversion_time - pp.calc_duration(rf) / 2 - pp.calc_duration(gz_spoil)

    # round delay to gradient raster time
    time_delay = np.ceil(time_delay / system.grad_raster_time) * system.grad_raster_time

    # check if delay is valid
    if not time_delay > 0:
        raise ValueError("Inversion time too short for given RF and spoiler durations.")

    # create delay event
    delay = pp.make_delay(time_delay)

    # add add events to sequence
    seq.add_block(rf)
    seq.add_block(gz_spoil)
    seq.add_block(delay)

    # calculate total duration of T1prep block
    total_duration = pp.calc_duration(rf) + pp.calc_duration(gz_spoil) + pp.calc_duration(delay)

    return (seq, total_duration)


def add_t2prep(
    seq: pp.Sequence | None = None,
    echo_time: float = 0.1,
    duration_180: float = 1e-3,
    rf_gap_time: float = 150e-6,
    spoil_ramp_time: float = 6e-4,
    spoil_flat_time: float = 6e-3,
    system: pp.Opts | None = None,
) -> tuple[pp.Sequence, float]:
    """Add a T2 preparation block to a sequence.

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    echo_time
        echo time in seconds
    duration_180
        duration of 180° block pulse
    rf_gap_time
        time between 2 consecutive RF pulses
    spoil_ramp_time
        duration of gradient spoiling ramp
    spoil_flat_time
        duration of gradient spoiling flat top
    system
        system limits

    Returns
    -------
    seq
        PyPulseq Sequence object
    total_duration
        total duration of the T2 preparation block in seconds
    """
    if not seq:
        seq = pp.Sequence()

    if not system:
        system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=100, slew_unit="T/m/s")

    # add 90°x excitation pulse at the beginning
    rf_90 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=duration_180 / 2, system=system)
    seq.add_block(rf_90)
    total_duration = duration_180 / 2

    # add delay before 1st MLEV-4 refocusing pulse
    delay = (
        echo_time / 8 - duration_180 / 4 - duration_180 / 2 - rf_gap_time - duration_180 / 2
    )  # TE/8 - 90°x/4 - 180°x/2 - rf_gap - 180°x/2
    if delay < 0:
        raise ValueError("Echo time too short for T2 preparation block.")
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add first MLEV-4 refocusing pulse
    seq, refoc_dur = _add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        rf_gap_time=rf_gap_time,
        seq=seq,
        negative_amp=False,
    )
    total_duration += refoc_dur

    # add delay before 2nd MLEV-4 refocusing pulse
    delay = echo_time / 4 - refoc_dur
    if delay < 0:
        raise ValueError("Echo time too short for T2 preparation block.")
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add second MLEV-4 refocusing pulse
    seq, refoc_dur = _add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        rf_gap_time=rf_gap_time,
        seq=seq,
        negative_amp=False,
    )
    total_duration += refoc_dur

    # add delay before 3rd MLEV-4 refocusing pulse
    delay = echo_time / 4 - refoc_dur
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add third MLEV-4 refocusing pulse
    seq, refoc_dur = _add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        rf_gap_time=rf_gap_time,
        seq=seq,
        negative_amp=True,
    )
    total_duration += refoc_dur

    # add delay before 4th MLEV-4 refocusing pulse
    delay = echo_time / 4 - refoc_dur
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add fourth MLEV-4 refocusing pulse
    seq, refoc_dur = _add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        rf_gap_time=rf_gap_time,
        seq=seq,
        negative_amp=True,
    )
    total_duration += refoc_dur

    # add delay before first tip-up pulse
    delay = echo_time / 8 - refoc_dur / 2 - duration_180 / 2 * 3 / 2  # TE/8 - refoc_dur/2 - 270°x/2
    if delay < 0:
        raise ValueError("Echo time too short for T2 preparation block.")
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add composite tip-up pulse (270°x + [-360]°x)
    rf_tip_up_270 = pp.make_block_pulse(flip_angle=3 * np.pi / 2, duration=duration_180 / 2 * 3, system=system)
    rf_tip_up_360 = pp.make_block_pulse(flip_angle=-2 * np.pi, duration=duration_180 * 2, system=system)
    seq.add_block(rf_tip_up_270)
    seq.add_block(pp.make_delay(rf_gap_time))
    seq.add_block(rf_tip_up_360)
    total_duration += duration_180 / 2 * 3 + rf_gap_time + duration_180 * 2

    # add spoiler gradient
    gz_spoil = pp.make_trapezoid(
        channel="z",
        amplitude=0.5 * system.max_grad,
        flat_time=spoil_flat_time,
        rise_time=spoil_ramp_time,
        fall_time=spoil_ramp_time,
        system=system,
    )
    seq.add_block(gz_spoil)
    total_duration += pp.calc_duration(gz_spoil)

    return (seq, total_duration)

def add_cestprep( # JWW added this function 5/11/26
    seq: pp.Sequence | None = None,
    pulse_shape: Literal['block', 'gauss'] = 'block',
    n_pulses: int = 3,
    b1p: float = 2.0,
    offset_ppm: float = 3.5,
    tp: float = 200e-3,
    td: float = 50e-3,
    spoil_amp: float | None = None,
    spoil_dur: float = 6.5e-3,
    spoil_rise_time: float = 1e-3,
    system: pp.Opts | None = None,
) -> tuple[pp.Sequence, float]:
    """Add a CEST preparation block to a sequence.

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    pulse_shape
        shape of saturation pulses
    n_pulses
        number of CEST prep pulses
    b1p
        peak B1 for CEST prep pulses in microtesla 
    offset 
        frequency offset for prep pulses in ppm
    tp
        duration of CEST pulses in seconds
    td
        duration of interpulse delay in seconds
    spoil_amp
        spoiler amplitude
    spoil_dur
        duration of each spoiler gradient
    spoil_rise_time
        rise time for spoiler gradients
    system
        system limits

    Returns
    -------
    seq
        PyPulseq Sequence object
    total_duration
        total duration of the CEST preparation block in seconds
    """
    if not seq:
        seq = pp.Sequence()

    if not system:
        system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=100, slew_unit="T/m/s")
        
    if not spoil_amp:
        spoil_amp = 0.8 * system.max_grad
    elif spoil_amp > system.max_grad:
        raise ValueError(
            f"Requested spoil_amp ({spoil_amp}) exceeds system.max_grad ({system.max_grad})."
            )
        
    valid_shapes = ['block', 'gauss']
    if pulse_shape not in valid_shapes:
        raise ValueError(
            f"Invalid pulse_shape '{pulse_shape}'. Must be one of {valid_shapes}"
        )
    
    # Align timing to gradient raster
    tp = np.ceil(tp / system.grad_raster_time) * system.grad_raster_time
    td = np.ceil(td / system.grad_raster_time) * system.grad_raster_time
    spoil_dur = np.ceil(spoil_dur / system.grad_raster_time) * system.grad_raster_time
    
    if spoil_dur > td:
        raise ValueError(
            f"Requested spoiler duration ({spoil_dur}) exceeds interpulse delay ({td})."
        )
    else:
        td -= spoil_dur
        
    # Create saturation pulse    
    gamma_hz = system.gamma * 1e-6
    freq = system.B0 * system.gamma * 1e-6
    offset_hz = offset_ppm * freq
    
    if pulse_shape == 'block':
        flip_angle = 2 * np.pi * gamma_hz * b1p * tp
        sat_pulse = pp.make_block_pulse(
            flip_angle=flip_angle, 
            duration=tp, 
            delay=system.rf_dead_time, 
            freq_offset=offset_hz,
            system=system
            )
        
    elif pulse_shape == 'gauss':
        # Placeholder sat pulse to calculate flip angle and scale
        sat_pulse = pp.make_gauss_pulse(flip_angle=np.pi, 
                                        duration=tp, 
                                        time_bw_product=0.2,
                                        apodization=0.5, 
                                        delay=system.rf_dead_time, 
                                        freq_offset=offset_hz,
                                        system=system)
        # Scale sat pulse to requested peak B1
        target_peak_hz = b1p * gamma_hz
        current_peak_hz = np.max(np.abs(sat_pulse.signal))
        sat_pulse.signal *= (target_peak_hz / current_peak_hz)
        
    # Create gradients
    gx_spoil_cest, gy_spoil_cest, gz_spoil_cest = [
        pp.make_trapezoid(channel=c, amplitude=spoil_amp, duration=spoil_dur, rise_time=spoil_rise_time, system=system)
        for c in ["x", "y", "z"]
    ]
    
    # Make delay once
    delay = pp.make_delay(td)
    
    # Write pulse train with spoilers
    pulse_dur = pp.calc_duration(sat_pulse)
    total_duration = 0
    accum_phase = 0 # Reset accumulated phase
    for n in range(n_pulses):
        sat_pulse.phase_offset = accum_phase % (2 * np.pi)
        seq.add_block(sat_pulse)
        accum_phase = (accum_phase + offset_hz * 2 * np.pi * tp) % (2 * np.pi)
        total_duration += pulse_dur # Update duraiton
        seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)
        total_duration += spoil_dur
        if n < n_pulses - 1:
            seq.add_block(delay)
            total_duration += td
    
    return (seq, total_duration)
        
        
        
        
        
        
    
        
    

    
