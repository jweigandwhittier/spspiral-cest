#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:22:50 2026

@author: jonah
"""
import os
import time
from pathlib import Path
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
import seqeyes
from utils import vds, sim_cest_rf, prep_pge2
from types import SimpleNamespace

# Prisma hardware limits
sys = pp.opts.Opts(
    max_grad = 30, # Try to fix crashing (could probably boost this back up for slice select)
    grad_unit = 'mT/m',
    max_slew = 90, # 
    slew_unit = 'mT/m/ms',
    rf_ringdown_time = 60e-6, # JWW change from 20 to 60 for parity with GE
    rf_dead_time = 100e-6,
    adc_dead_time = 20e-6, 
    adc_raster_time = 10e-6,
    B0 = 3.00
    )

sys_ge = pp.opts.Opts(
    max_grad = 30, # Try to fix crashing (could probably boost this back up for slice select)
    grad_unit = 'mT/m',
    max_slew = 90, # Had to change for GE PNS check
    slew_unit = 'mT/m/ms',
    rf_ringdown_time = 60e-6, 
    rf_dead_time = 100e-6,
    adc_dead_time = 20e-6, 
    grad_raster_time = 4e-6, # This is important
    adc_raster_time = 2e-6, # This is also important
    rf_raster_time = 2e-6, # Finally, need this
    block_duration_raster = 4e-6, # This might fix everything?
    B0 = 3.00
    )

# Flags
flags = {
'PLOT_VDS': False, # Plot spiral trajectories
'TRIGGER': True, # Cardiac triggering
'SPSP': True, # Calculate and write sequences with spatial-spectral saturation
'ZSPEC': False, # Full Z-spectral acquisition with delays

'FLAG_PLOT': False, # Plot sequence 
'FLAG_PLOT_ECG': False, # Plot with ECG
'FLAG_TEST_REPORT': False, # Print full test report
'FLAG_SEQEYES': False, # Open sequences in Seqeyes
'FLAG_MYOCARDIUM': False, # For ROI drawing

'FLAG_GE': True # For writing the sequence on the GE scanner
}

# Sequence definitions (for all sequences)
defs_dict = {
'fov': 300e-3, # [m]
'nx': 128,
'n_interleaves': 12,
'slice_thickness': 8e-3, # [m]
'tissue_t1': 1.2, # Assumed tissue T1 [s]; used for Ernst angle calculation

# CEST parameters
'b1': 1.20, # Peak B1 [uT]
'dc': 0.63,
'n_pulses': 23, 
'tp': 36e-3, # [s]
'spoil_rise_time': 1e-3, # [s]
'spoil_dur': 6.5e-3, # [s]
'spoil_amp': 0.8 * sys.max_grad,

# RF spoiling 
'rf_spoiling_inc': 117, # RF spoiling increment [°]

# Constants
'gamma_hz': sys.gamma * 1e-6,
'freq': sys.B0 * sys.gamma * 1e-6
}

defs = SimpleNamespace(**defs_dict)

# If writing sequences with spsp pulses, provide B1 map and WASABI seq filename
b1_map = np.load('data/recon/cindy_example_b1.npy') # Or paste DICOM file path directly
wasabi_seq_filename = 'sequences/example/cindy_b1.seq' # Put 'dicom_siemens' or 'dicom_ge' here if the map is from a DICOM

# Helper functions
def input_hr():
    while True:
        hr = input(">Enter subject's current heart rate: ")
        try:
            return int(hr)
        except:
            print("Error: Please enter a valid integer heart rate.")
            
def input_cv(cv_str):
    while True:
        cv = input(f">Enter {cv_str}: ")
        try:
            return float(cv)
        except:
            print(f"Error: Please enter a valid value for {cv_str}.")
            
def convert_ge(seq_filename, cvs, n_interleaves):
    import matlab.engine
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab("-java")
    # Force light mode
    eng.eval("set(groot, 'DefaultFigureColor', [1, 1, 1]);", nargout=0)
    eng.eval("set(groot, 'DefaultAxesColor', [1, 1, 1]);", nargout=0)
    eng.eval("set(groot, 'DefaultAxesXColor', [0, 0, 0]);", nargout=0)
    eng.eval("set(groot, 'DefaultAxesYColor', [0, 0, 0]);", nargout=0)
    eng.eval("set(groot, 'DefaultTextColor', [0, 0, 0]);", nargout=0)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Adding {script_dir} to MATLAB path")
    eng.addpath(script_dir, nargout=0)
    sys_dict = {
        'maxGrad': float(sys.max_grad),
        'maxSlew': float(sys.max_slew),
        'gamma': float(sys.gamma)
    }
    # Pass the simple dictionary instead of the complex 'sys' object.
    pislquant = n_interleaves 
    eng.convert_pge2(str(seq_filename), sys_dict, cvs, pislquant, True, nargout=0)
    # Python handles the input safely in your terminal
    input("Press Enter in this Python console to close the MATLAB plot and continue...")
    # Force the engine to close the open figure
    eng.eval("close(gcf);", nargout=0)

def write_sequence(sys, hr, offsets, flags, defs, spsp_objects):
    # Initialize new sequence
    seq = pp.Sequence(system=sys)
    gamma_hz = defs.gamma_hz
    freq = defs.freq
    resolution = defs.fov / defs.nx
    
    # Define GE block delay
    # Per JFN, there is an (empirically determined) 116us delay added per segment on GE, account for this
    ge_delay = 116e-6 
    
    # Pre-calculate aligned durations 
    raster = sys.grad_raster_time
    tp_aligned = np.round(defs.tp / raster) * raster
    td_aligned = np.round((defs.tp / defs.dc - defs.tp) / raster) * raster
    if flags['FLAG_GE']:
        td_aligned -= ge_delay
    sat_pulse_dur = tp_aligned + sys.rf_dead_time
    sat_time = (defs.n_pulses * sat_pulse_dur) + ((defs.n_pulses - 1) * td_aligned)
    spoil_time = defs.spoil_dur
    prep_time = sat_time + spoil_time
    
    # Heart rate calculations
    if flags['TRIGGER']:
        rr = 60 / hr
        dias_delay = np.round(0.559 * np.sqrt(rr) - 0.137, 3) # From simulations
        print(f'Calculated diastolic delay is {dias_delay} s')
        n_beats = max(0, int(np.ceil((prep_time - dias_delay) / rr)))
        trig_delay = (n_beats * rr) + dias_delay - prep_time
        # Make trigger with delay
        trig_delay = np.round(trig_delay / raster) * raster
        if not flags['FLAG_GE']:
            trig_delay -= ge_delay
            trig = pp.make_trigger(channel='physio1', duration=trig_delay)
        else: 
            trig = pp.make_trigger(channel='physio1', duration=20e-6)
        print(f"Using R-R interval of {rr:.2f} s, calculated trigger delay is {trig_delay:.2f} s\n")
        
    # Spiral parameters
    max_kspace_radius = 0.5 / (resolution)
    sampling_period = sys.grad_raster_time
    fov_coefficients = [defs.fov, -1/4 * defs.fov]
    
    # Write spiral
    (k, g, s, timing, r, theta) = vds.variable_density_spiral_trajectory(
            system=sys,
            sampling_period=sampling_period,
            n_interleaves=defs.n_interleaves,
            fov_coefficients=fov_coefficients,
            max_kspace_radius=max_kspace_radius
        )
    
    # Number of samples in the gradient waveform
    num_grad_samples = np.shape(g)[0]
    
    # Calculate ADC
    adc_dwell = 10e-6 # Hardcode this to preserve readout bandwidth (BWPP --> SNR)
    active_grad_time = num_grad_samples * sys.grad_raster_time
    sampling_time = active_grad_time - 2 * sys.adc_dead_time
    max_samples = int(np.floor(sampling_time / adc_dwell))
    adc_total_samples = max_samples - (max_samples % 4)
    assert adc_total_samples <= 8192, 'ADC samples exceed maximum value of 8192.'
    adc = pp.make_adc(num_samples=adc_total_samples, dwell=adc_dwell, delay=sys.adc_dead_time, system=sys)
    print(f'ADC Samples: {adc_total_samples}\n')
    
    # Make gradients
    n_points_g = np.shape(g)[0]
    n_points_k = np.shape(k)[0]
    spiral_readout_grad = np.zeros((defs.n_interleaves, 2, n_points_g))
    spiral_trajectory = np.zeros((defs.n_interleaves, 2, n_points_k))
    angle_increment = 2 * np.pi / defs.n_interleaves
    angle_array = np.arange(0, 2 * np.pi, angle_increment)
    
    gx_readout_list = []
    gy_readout_list = []
    gx_rewinder_list = []
    gy_rewinder_list = []
    
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
        
        # Make rewinders with common raster time
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
        
        gx_readout_list.append(gx_readout)
        gy_readout_list.append(gy_readout)
        gx_rewinder_list.append(gx_rewinder)
        gy_rewinder_list.append(gy_rewinder)
        
    # Check waveform start/end values
    for i in range(defs.n_interleaves):
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
        slice_thickness = defs.slice_thickness,
        apodization = 0.5,
        time_bw_product = 8.0,
        system = sys,
        return_gz = True,
        delay = sys.rf_dead_time,
        use = 'excitation')
    
    # Write spoiler
    n_cycles = 4
    spoil_area = n_cycles / defs.slice_thickness - gz_dummy.area / 2
    
    gz_spoil = pp.make_trapezoid(
        channel='z',
        area=spoil_area,
        system=sys
    )
    
    # Update max rewinder duration 
    max_rewinder_duration = max(
    max(pp.calc_duration(gx_rewinder_list[i]) for i in range(defs.n_interleaves)),
    max(pp.calc_duration(gy_rewinder_list[i]) for i in range(defs.n_interleaves)),
    pp.calc_duration(gz_spoil))
    
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
    ernst_angle = np.arccos(np.exp(-tr/defs.tissue_t1))
    ernst_angle = 15 * (np.pi / 180) # Just try 15deg
    
    rf_exc, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle = ernst_angle, 
        duration = 1e-3, 
        slice_thickness = defs.slice_thickness,
        apodization = 0.5,
        time_bw_product = 8.0,
        system = sys,
        return_gz = True,
        delay = sys.rf_dead_time,
        use = 'excitation')

    print(f"\nFinal TR: {tr*1000:.2f} ms | Ernst Angle: {ernst_angle * 180 / np.pi:.2f}°")
    print(f"Final readout duration: {tr*defs.n_interleaves*1000:.2f} ms")
    
    # Plot the spirals
    if flags['PLOT_VDS']:
        fig, ax = plt.subplots()
        for i in range(defs.n_interleaves):
            ax.plot(spiral_trajectory[i, 0, :], spiral_trajectory[i, 1, :])
            ax.set_title(f"$K$-Space Trajectory\n({defs.n_interleaves} Interleaves)")
            ax.set_xlabel("$k_x$ (1/m)")
            ax.set_ylabel("$k_y$ (1/m)")
            ax.axis('equal')
            
    # Grab SPSP waveforms if needed
    if flags['SPSP']:
        spsp_grad_x = spsp_objects['full_gx']
        spsp_grad_y = spsp_objects['full_gy']
        spsp_rf_shape = spsp_objects['full_rf']
        
    # Define sat pulses outside the loop
    placeholder_offset = offsets[0] * freq
    # Placeholder sat pulse to calculate flip angle and scale
    sat_pulse = pp.make_gauss_pulse(flip_angle=np.pi, 
                                    duration=tp_aligned, 
                                    time_bw_product=0.2,
                                    apodization=0.5, 
                                    delay=100e-6, 
                                    freq_offset=placeholder_offset,
                                    system=sys,
                                    use='preparation')
    
    # Scale sat pulse and find total flip angle
    target_peak_hz = defs.b1 * gamma_hz
    current_peak_hz = np.max(np.abs(sat_pulse.signal))
    sat_pulse.signal *= (target_peak_hz / current_peak_hz)
    dt = sys.rf_raster_time
    total_flip_angle = np.abs(np.sum(sat_pulse.signal)) * dt * 2 * np.pi
    
    if flags['SPSP']:
        spsp_pulse = pp.make_arbitrary_rf(spsp_rf_shape, 
                                          flip_angle=total_flip_angle, 
                                          dwell=sys.rf_raster_time, 
                                          delay=sys.rf_dead_time,
                                          freq_offset=placeholder_offset,
                                          system=sys,
                                          use='preparation')

    # Define fixed labels once, outside the offset loop
    if flags['FLAG_GE']:
        trigger_label      = 1
        prep_label         = 2  # sat pulse + spoiler + delay (pulses 0 to N-2)
        last_prep_label    = 3  # sat pulse + spoiler only (pulse N-1)
        readout_base_label = 4  # one per interleave (4 through 4+N-1)
        delay_label        = 4 + defs.n_interleaves  # ZSPEC recovery delay
        
    # Calibration block for GE
    if flags['FLAG_GE']:
        print("Adding GE receiver gain calibration readouts...")
        rf_phase_cal = 0
        rf_inc_cal = 0
        
        for n in range(defs.n_interleaves):
            current_readout_label = readout_base_label + n
            
            # Apply phase offsets for spoiling
            rf_exc.phase_offset = rf_phase_cal / 180 * np.pi
            adc.phase_offset = rf_phase_cal / 180 * np.pi
            
            # Excitation
            seq.add_block(rf_exc, gz, pp.make_label('TRID', 'SET', current_readout_label))
            # Rephase
            seq.add_block(gz_reph)
            # Spiral readout
            seq.add_block(gx_readout_list[n], gy_readout_list[n], adc)
            # Rewind and spoil
            gx_rew = gx_rewinder_list[n]
            gy_rew = gy_rewinder_list[n]
            seq.add_block(gx_rew, gy_rew, gz_spoil)
            
            # TR matching dead time
            current_grad_dur = max(pp.calc_duration(gx_rew), pp.calc_duration(gy_rew), pp.calc_duration(gz_spoil))
            extra_delay = max_rewinder_duration - current_grad_dur - ge_delay
            
            if extra_delay > 0:
                extra_delay = np.ceil(extra_delay / raster) * raster
                seq.add_block(pp.make_delay(extra_delay))
            
            # Increment RF and ADC phase
            rf_inc_cal = divmod(rf_inc_cal + defs.rf_spoiling_inc, 360.0)[1]
            rf_phase_cal = divmod(rf_phase_cal + rf_inc_cal, 360.0)[1]
        
    # Loop over offsets
    for offset_idx, offset_ppm in enumerate(offsets):
        
        if flags['TRIGGER']:
            if not flags['FLAG_GE']:
                seq.add_block(trig)
            else:
                seq.add_block(trig, pp.make_label('TRID', 'SET', trigger_label))
                seq.add_block(pp.make_delay(trig_delay))
        
        # Define saturation offset frequency
        offsets_hz = offset_ppm * freq
        
        sat_pulse.freq_offset = offsets_hz
        print(f'Frequency offset set to {sat_pulse.freq_offset} Hz!') # For debugging
        
        if flags['SPSP']:
            spsp_pulse.freq_offset = offsets_hz
        
        for n in range(defs.n_pulses):
            is_last = (n == defs.n_pulses - 1)
            label = last_prep_label if is_last else prep_label

            if flags['SPSP']:
                if flags['FLAG_GE']:
                    seq.add_block(spsp_pulse, spsp_grad_x, spsp_grad_y,
                                  pp.make_label('TRID', 'SET', label))
                else:
                    seq.add_block(spsp_pulse, spsp_grad_x, spsp_grad_y)

            else:
                if flags['FLAG_GE']:
                    seq.add_block(sat_pulse, pp.make_label('TRID', 'SET', label))
                else:
                    seq.add_block(sat_pulse)

            # Spoiler — no label, TRID carries over from sat pulse
            gx_spoil_cest, gy_spoil_cest, gz_spoil_cest = [
                pp.make_trapezoid(channel=c, system=sys, amplitude=defs.spoil_amp,
                                  duration=defs.spoil_dur, rise_time=defs.spoil_rise_time)
                for c in ["x", "y", "z"]
            ]
            seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)

            # Inter-pulse delay — no label, TRID carries over, only for non-last pulses
            if not is_last:
                seq.add_block(pp.make_delay(td_aligned - defs.spoil_dur))
                
        # Put it together (Readout)
        rf_phase = 0
        rf_inc = 0
            
        for n in range(defs.n_interleaves):
            current_readout_label = readout_base_label + n
            
            # Apply phase offsets for spoiling
            rf_exc.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            
            # Excitation
            if flags['FLAG_GE']:
                seq.add_block(rf_exc, gz, pp.make_label('TRID', 'SET', current_readout_label))
            else:
                seq.add_block(rf_exc, gz)
            # Rephase
            seq.add_block(gz_reph)
            # Spiral readout
            seq.add_block(gx_readout_list[n], gy_readout_list[n], adc)
            # Rewind and spoil
            gx_rew = gx_rewinder_list[n]
            gy_rew = gy_rewinder_list[n]
            seq.add_block(gx_rew, gy_rew, gz_spoil)
            
            # Calculate how much 'dead time' is left in this TR
            current_grad_dur = max(pp.calc_duration(gx_rew), pp.calc_duration(gy_rew), pp.calc_duration(gz_spoil))
            extra_delay = max_rewinder_duration - current_grad_dur
            if flags['FLAG_GE']:
                extra_delay -= ge_delay
            
            # Add a separate delay block if there's time left
            if extra_delay > 0:
                # Align to raster
                extra_delay = np.ceil(extra_delay / raster) * raster
                seq.add_block(pp.make_delay(extra_delay))
            
            # Increment RF and ADC phase
            rf_inc = divmod(rf_inc + defs.rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
            
        # Add magnetization recovery delay if ZSPEC is enabled and not the last offset
        if flags['ZSPEC'] and offset_idx < len(offsets) - 1:
            recovery_delay_time = 5 * defs.tissue_t1
            aligned_recovery_delay = np.round(recovery_delay_time / sys.grad_raster_time) * sys.grad_raster_time
            if flags['FLAG_GE']:
                seq.add_block(pp.make_delay(aligned_recovery_delay), pp.make_label('TRID', 'SET', delay_label))
            else:
                seq.add_block(pp.make_delay(aligned_recovery_delay))

    # Define sequence naming prefix based on SPSP flag
    pulse_type = "SPSP" if flags['SPSP'] else "Gauss"
    if flags['ZSPEC']:
        seq_name = f'Spiral_CEST_{pulse_type}_ZSPEC_{hr}_bpm'
    else:
        # Format the offset: replace decimal point with 'p' (e.g., 75.0 -> 75p0)
        offset_str = str(offsets[0]).replace('.', 'p')
        seq_name = f'Spiral_CEST_{pulse_type}_{hr}_bpm_{offset_str}_ppm'
    if flags['FLAG_GE']:
        seq_name += '_ge'
        
    seq_id = f"{seq_name.lower()}.seq"

    # Write definitions for recon   
    seq.set_definition('Name', seq_name)
    seq.set_definition('FOV', [defs.fov, defs.fov, defs.slice_thickness])
    seq.set_definition('Slice_Thickness', defs.slice_thickness)
    seq.set_definition('Nx', defs.nx)
    seq.set_definition('N_Interleaves', defs.n_interleaves)
    seq.set_definition('Offsets_ppm', [float(o) for o in offsets])
    seq.set_definition('B1peak', defs.b1)
    seq.set_definition('N_Pulses', defs.n_pulses)
    seq.set_definition('DC', defs.dc)
    seq.set_definition('HR', hr)
    seq.set_definition('MaxAdcSegmentLength', adc_total_samples)
    
    # Check timing and write sequence
    ok, error_report = seq.check_timing()
    if ok:
        print('\nTiming check passed successfully!')
        if flags['FLAG_TEST_REPORT']:
            print(seq.test_report())
    else:
        print('\nTiming check failed! Error listing follows\n')
        [print(e) for e in error_report]
        
    seq_filename = f'sequences/cest/{seq_id}'
    seq.write(seq_filename)
    
    if flags['FLAG_PLOT']:
        seq.plot(grad_disp='mT/m')

    # Load in Seqeyes
    if flags['FLAG_SEQEYES']:
        seqeyes.seqeyes(seq_filename)
        
    return seq_filename
    
def main(sys):
    hr = input_hr()
    pge_entry = 60

    # Destination configurations
    scanner_sequence_dir = Path('/srv/nfs/psd/usr/psd/pulseq/v7/sequences')
    scanner_entry_dir = Path('/srv/nfs/psd/usr/psd/pulseq/v7/')
    
    # Track files to process at the end
    sequences_to_batch = []

    if flags['SPSP']:
        myocardium = flags['FLAG_MYOCARDIUM']
        spsp_objects = sim_cest_rf.calc_spsp(b1_map, wasabi_seq_filename, defs.tp, sys, myocardium)
    else:
        spsp_objects = None
        
    if flags['FLAG_GE']:
        cvs = {}
        print("\n╔════════════════════════════════════════════════════════════╗")
        print("║x/y/zloc are obtained from the User CVs menu on the console.║")
        print("║   Please check completed mid-ventricle short axis scan.    ║")
        print("╚════════════════════════════════════════════════════════════╝\n")
        for cv in ['xloc', 'yloc', 'zloc']: 
         cvs[cv] = input_cv(cv)
        
    if flags['ZSPEC']:
        offsets_part0 = np.array([75.00])            
        offsets_part1 = np.arange(-10, -5, 1)        
        offsets_part2 = np.arange(-5, 5.01, 0.2)     
        offsets_part3 = np.arange(6, 10.01, 1)       
        offsets = np.concatenate((offsets_part0, offsets_part1, offsets_part2, offsets_part3))
        offsets = np.round(offsets, 2)               
        
        print(f'\n--- Writing Z-Spectrum sequence with {len(offsets)} offsets ---\n')
        seq_filename = write_sequence(sys, hr, offsets, flags, defs, spsp_objects)
        
        if flags['FLAG_GE']:
            convert_ge(seq_filename, cvs, defs.n_interleaves)
            sequences_to_batch.append({"seq_path": Path(seq_filename), "pge_n": pge_entry})
            pge_entry += 1
            
    else:
        offsets = [75.00, 2.00, -2.00]
        for offset in offsets:
            print(f'\n--- Writing sequence: {offset} ppm ---\n')
            seq_filename = write_sequence(sys, hr, [offset], flags, defs, spsp_objects)
            
            if flags['FLAG_GE']:
                convert_ge(seq_filename, cvs, defs.n_interleaves)
                sequences_to_batch.append({"seq_path": Path(seq_filename), "pge_n": pge_entry})
                pge_entry += 1

    # --- SINGLE BATCH HANDOFF ---
    if flags['FLAG_GE'] and sequences_to_batch:
        master_script_path = Path('sequences/deploy_all_sequences_cest.sh')
        prep_pge2.prep_pge2_batch(sequences_to_batch, scanner_sequence_dir, scanner_entry_dir, master_script_path)
        
if __name__ == "__main__":
    tic = time.time()
    if flags['FLAG_GE']:
        sys = sys_ge
    main(sys)
    toc = time.time()
    print(f'\nAll sequences written, converted, and deployed in {np.round(toc-tic, 2)} seconds.')