#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:36:26 2026

@author: jonah
"""
import time
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
import seqeyes
from utils.vds import variable_density_spiral_trajectory

# Prisma hardware limits
sys = pp.opts.Opts(
    # max_grad = 80,
    max_grad = 30, # Try to fix crashing (could probably boost this back up for slice select)
    grad_unit = 'mT/m',
    # max_slew = 150,
    max_slew = 100, # Back off a bit here, too (needs to be around 100 mT/(m*ms) to avoid PNS violation)
    slew_unit = 'mT/m/ms',
    rf_ringdown_time = 20e-6,
    rf_dead_time = 100e-6,
    adc_dead_time = 10e-6,
    B0 = 3.0,
    )

PLOT_VDS = False
CEST_PREP = True
TRIGGER = True

FLAG_PLOT = False
FLAG_PLOT_ECG = False
FLAG_TEST = False
FLAG_SEQEYES = False

def input_hr():
    while True:
        hr = input("Enter subject's current heart rate: ")
        try:
            return int(hr)
        except:
            print("Error: Please enter a valid integer heart rate.")  

def write_sequence(sys, hr, PLOT_VDS, CEST_PREP, TRIGGER, FLAG_PLOT, FLAG_TEST, FLAG_SEQEYES):
    # Initialize new sequence
    seq = pp.Sequence(system=sys)
    gamma_hz = sys.gamma * 1e-6
    freq = sys.B0 * gamma_hz
    
    # Parameters 
    fov = 300e-3 
    nx = 128
    resolution = fov / nx
    n_interleaves = 12
    slice_thickness = 8e-3
    tissue_t1 = 2.3 # Assumed tissue T1 [s]; used for Ernst angle calculation
    
    # CEST parameters
    # offsets_ppm = offset # [ppm]
    offsets_ppm = [-75, -4, -3.73, -3.46, -3.2, -2.93, -2.67, -2.4, -2.13, -1.86, -1.6, -1.33, -1.06, -0.8, -0.53, -0.26, 0, 0.26, 0.53, 0.8, 1.06, 1.33, 1.67, 1.86, 2.13, 2.4, 2.66, 2.93, 3.2, 3.46, 3.73, 4]
    trec = [15, 0.5, 1, 2, 3, 4, 5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 5, 4, 3, 2, 1, 0.5]
    b1 = 3.7 # Peak B1 [uT]
    n_pulses = 1
    tp = 5.12e-3
    spoil_rise_time = 1e-3
    spoil_dur = 4.5e-3
    spoil_amp = 0.8 * sys.max_grad
    
    # RF spoiling 
    rf_phase = 0
    rf_spoiling_inc = 117 # RF spoiling increment [°]
    
    # Heart rate calculations
    raster = sys.grad_raster_time
    if TRIGGER:
        rr = 60 / hr
        dias_delay = np.round(0.559 * np.sqrt(60/hr) - 0.137, 3) # From simulations
        if CEST_PREP:
            # Pre-calculate aligned durations 
            tp_aligned = np.round(tp / raster) * raster
            sat_pulse_dur = tp_aligned + sys.rf_dead_time
            sat_time = (n_pulses * sat_pulse_dur)
            spoil_time = spoil_dur
            prep_time = sat_time + spoil_time
            n_beats = max(0, int(np.ceil((prep_time - dias_delay) / rr)))
            trig_delay = (n_beats * rr) + dias_delay - prep_time
        else:
            trig_delay = dias_delay
        # Make trigger with delay
        trig_delay = np.round(trig_delay / raster) * raster
        trig = pp.make_trigger(channel='physio1', duration=trig_delay)    
        print(f"Using R-R interval of {rr:.2f} s, calculated trigger delay is {trig_delay:.2f} s\n")
    
    # Spiral parameters
    max_kspace_radius = 0.5 / (resolution)
    sampling_period = sys.grad_raster_time
    # fov_coefficients = [fov, -fov/3]
    fov_coefficients = [fov, -1/4*fov]
    
    # Write spiral
    (k, g, s, timing, r, theta) = variable_density_spiral_trajectory(
            system=sys,
            sampling_period=sampling_period,
            n_interleaves=n_interleaves,
            fov_coefficients=fov_coefficients,
            max_kspace_radius=max_kspace_radius
        )
    
    # Number of samples in the gradient waveform
    num_grad_samples = np.shape(g)[0]
    
    # Calculate ADC
    adc_dwell = sys.grad_raster_time
    adc_total_samples = num_grad_samples - 1
    print('ADC total samples: {adc_total_samples}')
    assert adc_total_samples <= 8192, 'ADC samples exceed maximum value of 8192.'
    adc = pp.make_adc(num_samples=adc_total_samples, dwell=adc_dwell, delay=sys.adc_dead_time, system=sys)
    print(f'ADC Samples: {adc_total_samples}\n')
    
    # Make gradients
    n_points_g = np.shape(g)[0]
    n_points_k = np.shape(k)[0]
    spiral_readout_grad = np.zeros((n_interleaves, 2, n_points_g))
    spiral_trajectory = np.zeros((n_interleaves, 2, n_points_k))
    angle_increment = 2 * np.pi / n_interleaves
    angle_array = np.arange(0, 2 * np.pi, angle_increment)
    
    gx_readout_list = []
    gy_readout_list = []
    gx_rewinder_list = []
    gy_rewinder_list = []
    
    max_rewinder_duration = 0
    
    for n, angle in enumerate(angle_array):
        exp_angle = np.exp(1j * angle)
        exp_angle_pi = np.exp(1j * (angle + np.pi))
        
        spiral_readout_grad[n, 0, :] = np.real(g * exp_angle)
        spiral_readout_grad[n, 1, :] = np.imag(g * exp_angle)
        spiral_trajectory[n, 0, :] = np.real(k * exp_angle_pi)
        spiral_trajectory[n, 1, :] = np.imag(k * exp_angle_pi)
        
        gx_readout = pp.make_arbitrary_grad(
            channel='x',
            waveform=spiral_readout_grad[n, 0],
            first=0,
            delay=adc.delay,
            system=sys,
        )
    
        gy_readout = pp.make_arbitrary_grad(
            channel='y',
            waveform=spiral_readout_grad[n, 1],
            first=0,
            delay=adc.delay,
            system=sys,
        )
        
        gx_rewinder, gxr_times, gxr_amps = pp.make_extended_trapezoid_area(
           area=-gx_readout.area,
           channel='x',
           grad_start=gx_readout.last,
           grad_end=0,
           system=sys,
        )

        gy_rewinder, gyr_times, gyr_amps = pp.make_extended_trapezoid_area(
           area=-gy_readout.area,
           channel='y',
           grad_start=gy_readout.last,
           grad_end=0,
           system=sys,
        )
        
        # Safely calculate diffs using the actual time nodes (avoiding divide-by-zero on triangles)
        dt_x = np.diff(gxr_times)
        dt_y = np.diff(gyr_times)
        
        gxr_slew = np.abs(np.divide(np.diff(gxr_amps), dt_x, out=np.zeros_like(dt_x), where=dt_x!=0))
        gyr_slew = np.abs(np.divide(np.diff(gyr_amps), dt_y, out=np.zeros_like(dt_y), where=dt_y!=0))
        
        # Get highest slew and convert to T/m/s
        max_slew = max(np.max(gxr_slew), np.max(gyr_slew)) / sys.gamma
        
        # print(f"Max slew rate of rewinder gradients: {round(max_slew, 2)} T/m/s")
        
        gx_readout_list.append(gx_readout)
        gy_readout_list.append(gy_readout)
        gx_rewinder_list.append(gx_rewinder)
        gy_rewinder_list.append(gy_rewinder)
        
        # Calculate max rewinder duration
        max_rewinder_duration = max(pp.calc_duration(gx_rewinder), pp.calc_duration(gy_rewinder))
    
    # Check waveform start/end values
    for i in range(n_interleaves):
        gx_read = gx_readout_list[i]
        gy_read = gy_readout_list[i]
        gx_rew = gx_rewinder_list[i]
        gy_rew = gy_rewinder_list[i]
        tol = 0
        # Assert starting point of readouts is 0
        assert abs(gx_read.first) == tol, f"Gx readout {i} starts at {gx_read.first:.6e}, expected 0"
        assert abs(gy_read.first) == tol, f"Gy readout {i} starts at {gy_read.first:.6e}, expected 0"
        # Assert end of readout matches start of rewinder
        assert abs(gx_read.last - gx_rew.first) == tol, f"Gx readout {i} end ({gx_read.last:.6e}) mismatches rewinder start ({gx_rew.first:.6e})"
        assert abs(gy_read.last - gy_rew.first) == tol, f"Gy readout {i} end ({gy_read.last:.6e}) mismatches rewinder start ({gy_rew.first:.6e})"
        # Assert end point of rewinders is 0
        assert abs(gx_rew.last) == tol, f"Gx rewinder {i} ends at {gx_rew.last:.6e}, expected 0"
        assert abs(gy_rew.last) == tol, f"Gy rewinder {i} ends at {gy_rew.last:.6e}, expected 0"
    
    print("All boundary assertions passed: Readouts and rewinders are cleanly matched and zero-bounded.")
    
    # Write excitation
    rf_dummy, gz_dummy, gzr_dummy = pp.make_sinc_pulse(
        flip_angle = 90 / 180 * np.pi, # Placeholder flip angle
        duration = 1e-3, # [s]
        slice_thickness = slice_thickness,
        apodization = 0.5,
        time_bw_product = 4.0,
        system = sys,
        return_gz = True,
        delay = sys.rf_dead_time)
    
    # Write spoiler
    n_cycles = 4
    spoil_area = n_cycles / slice_thickness - gz_dummy.area / 2
    
    gz_spoil = pp.make_trapezoid(
        channel='z',
        area=spoil_area,
        system=sys
    )
    
    # Update max rewinder duration one more time
    max_rewinder_duration = max(max_rewinder_duration, pp.calc_duration(gz_spoil))
    
    # Get duration of excitation
    exc_duration = pp.calc_duration(gz_dummy) + pp.calc_duration(gzr_dummy)
    print(f"\nExcitation Duration: {exc_duration * 1000:.2f} ms")
    
    # Get duration of single readout arm + rewinder
    readout_duration = pp.calc_duration(gx_readout_list[0])
    arm_duration = readout_duration + max_rewinder_duration
    print(f"Readout Duration: {readout_duration * 1000:.2f} ms")
    print(f"Max Rewinder Duration: {max_rewinder_duration * 1000:.2f} ms")
    print(f"Arm Duration: {arm_duration * 1000:.2f} ms")
    
    # Find Ernst angle
    tr = exc_duration + arm_duration
    ernst_angle = np.arccos(np.exp(-tr/tissue_t1))
    
    rf_exc, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle = ernst_angle, 
        duration = 1e-3, 
        slice_thickness = slice_thickness,
        apodization = 0.5,
        time_bw_product = 8.0,
        system = sys,
        return_gz = True,
        delay = sys.rf_dead_time)
    
    print(f"\nFinal TR: {tr*1000:.2f} ms | Ernst Angle: {ernst_angle * 180 / np.pi:.2f}°")
    print(f"Final readout duration: {tr*n_interleaves*1000:.2f} ms")
    
    if PLOT_VDS:
        fig, ax = plt.subplots()
        for i in range(n_interleaves):
            ax.plot(spiral_trajectory[i, 0, :], spiral_trajectory[i, 1, :])
            ax.set_title(f"$K$-Space Trajectory\n({n_interleaves} Interleaves)")
            ax.set_xlabel("$k_x$ (1/m)")
            ax.set_ylabel("$k_y$ (1/m)")
            ax.axis('equal')
    
    for m, offset in enumerate(offsets_ppm):
    
        if TRIGGER:
            seq.add_block(trig)
        
        if CEST_PREP:
            # Define saturation block
            offset_hz = offset * freq
            fa_sat = b1 * gamma_hz * tp_aligned * 2 * np.pi
            sat_pulse = pp.make_block_pulse(fa_sat, duration=tp_aligned, system=sys, freq_offset = offset_hz)
            # Make spoiler
            gx_spoil_cest, gy_spoil_cest, gz_spoil_cest = [
                pp.make_trapezoid(channel=c, system=sys, amplitude=spoil_amp, duration=spoil_dur, rise_time=spoil_rise_time)
                for c in ["x", "y", "z"]]
            # Make sat loop
            for n in range(n_pulses):
                seq.add_block(sat_pulse) 
            seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)
        
        # Put it together
        rf_phase = 0
        rf_inc = 0
        
        # Calculate the absolute longest duration needed
        # raw_target_dur = max(max_rewinder_duration, pp.calc_duration(gz_spoil))
        # Enforce raster alignment
        # target_rewinder_block_dur = np.round(raw_target_dur / raster) * raster
        # Quick diagnostic to prove it worked
        # print(f"Raw duration:     {raw_target_dur:.9f} s")
        # print(f"Aligned duration: {target_rewinder_block_dur:.9f} s")
        # assert (target_rewinder_block_dur % raster) < 1e-9 or (raster - (target_rewinder_block_dur % raster)) < 1e-9, \
        #     f"Target duration {target_rewinder_block_dur} is not aligned to the {raster}s raster!"
            
        for n in range(n_interleaves):
            # Apply phase offsets for spoiling
            rf_exc.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            # Excitation
            seq.add_block(rf_exc, gz)
            # Rephase
            seq.add_block(gz_reph)
            # Spiral readout
            # Shift ADC for troubleshooting
            # if n == 0:
            #     seq.add_block(gx_readout_list[n], gy_readout_list[n])
            # else:
            seq.add_block(gx_readout_list[n], gy_readout_list[n], adc)
            # Rewind and spoil
            gx_rew = gx_rewinder_list[n]
            gy_rew = gy_rewinder_list[n]
            seq.add_block(gx_rew, gy_rew, gz_spoil)
            # Calculate how much 'dead time' is left in this TR
            current_grad_dur = max(pp.calc_duration(gx_rew), pp.calc_duration(gy_rew), pp.calc_duration(gz_spoil))
            extra_delay = max_rewinder_duration - current_grad_dur
            # Add a separate delay block if there's time left
            if extra_delay > 0:
                # Align to raster (10us) to be safe
                extra_delay = np.ceil(extra_delay / sys.grad_raster_time) * sys.grad_raster_time
                seq.add_block(pp.make_delay(extra_delay))
            # Increment RF and ADC phase
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
            
        # Add delay
        trec_delay = pp.make_delay(trec[m])
        seq.add_block(trec_delay)
    
    # Define seq filename
    # seq_id = f"spiral_wasabi_{hr}_bpm_{offset}_ppm.seq"
    seq_id = f"spiral_wasabi_{hr}_bpm.seq"
    
    # Write definitions for recon   
    seq.set_definition('Name', f'Spiral_WASABI_{hr}_bpm')
    seq.set_definition('B0', sys.B0)
    seq.set_definition('FOV', [fov, fov, slice_thickness])
    seq.set_definition('Slice_Thickness', slice_thickness)
    seq.set_definition('Nx', nx)
    seq.set_definition('N_Interleaves', n_interleaves)
    seq.set_definition('Offsets_ppm', offsets_ppm)
    seq.set_definition('tp', tp_aligned)
    seq.set_definition('B1peak', b1)
    seq.set_definition('N_Pulses', n_pulses)
    seq.set_definition('HR', hr)
    seq.set_definition('MaxAdcSegmentLength', adc_total_samples)
    
    # Check timing and write sequence
    ok, error_report = seq.check_timing()
    if ok:
        print('\nTiming check passed successfully!')
        if FLAG_TEST:
            print(seq.test_report())
    else:
        print('\nTiming check failed! Error listing follows\n')
        [print(e) for e in error_report]
    seq.write(f'sequences/wasabi/{seq_id}')
    seq.plot(grad_disp='mT/m')
    
    # Load in Seqeyes
    if FLAG_SEQEYES:
        seqeyes.seqeyes(seq_id)
        
def main(sys):
    hr = input_hr()
    write_sequence(sys, hr, PLOT_VDS, CEST_PREP, TRIGGER, FLAG_PLOT, FLAG_TEST, FLAG_SEQEYES)

if __name__ == "__main__":
    tic = time.time()
    main(sys)
    toc = time.time()
    print(f'\nAll sequences written in {np.round(toc-tic, 2)} seconds.')