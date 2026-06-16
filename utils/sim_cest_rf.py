#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:24:26 2026

@author: jonah
"""
import roipoly
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
from . import read_dicom

# Helper functions
def align_grad_raster(time, sys):
    return np.ceil(time / sys.grad_raster_time) * sys.grad_raster_time

def draw_roi(b1_map, myocardium):
    fig, ax = plt.subplots()
    ax.imshow(b1_map, cmap='gray')
    ax.axis('off')
    if myocardium:
        rois = roipoly.MultiRoi(roi_names=['Epicardium', 'Endocardium'])
        mask_epi = rois.rois['Epicardium'].get_mask(b1_map)
        mask_endo = rois.rois['Endocardium'].get_mask(b1_map)
        final_mask = np.logical_and(mask_epi, np.logical_not(mask_endo))
    else:
        ax.set_title('Draw ROI')
        roi = roipoly.RoiPoly()
        final_mask = roi.get_mask(b1_map)
    return final_mask

def domintrap_pypulseq(area_m_inv, sys, channel='z'):
    trap = pp.make_trapezoid(channel=channel, system=sys, area=area_m_inv)
    return trap

def dotrap_pypulseq(area_m_inv, sys, channel='z'):
    trap = pp.make_trapezoid(channel=channel, system=sys, area=area_m_inv)
    return trap

def find_optimal_spsp_pair(sys, target_duration_s=250e-6, min_flat_time_s=50e-6):
    safe_max_grad = sys.max_grad * 0.99
    safe_max_slew = sys.max_slew * 0.99
    safe_sys = pp.opts.Opts(
        max_grad=safe_max_grad,
        max_slew=safe_max_slew,
        grad_raster_time=sys.grad_raster_time
    )
    low_amp = 0.0
    high_amp = sys.max_grad
    best_amp = low_amp
    
    for _ in range(50):
        mid_amp = (low_amp + high_amp) / 2.0
        test_trap1 = pp.make_trapezoid(channel='z', system=safe_sys, 
                                       amplitude=mid_amp, flat_time=min_flat_time_s)
        test_trap2 = pp.make_trapezoid(channel='z', system=safe_sys, 
                                       area=-test_trap1.area)
        total_dur = pp.calc_duration(test_trap1) + pp.calc_duration(test_trap2)
        if total_dur <= target_duration_s:
            best_amp = mid_amp
            low_amp = mid_amp 
        else:
            high_amp = mid_amp
    final_trap1 = pp.make_trapezoid(channel='z', system=safe_sys, amplitude=best_amp, flat_time=min_flat_time_s)
    final_trap2 = pp.make_trapezoid(channel='z', system=safe_sys, area=-final_trap1.area)
    return final_trap1, final_trap2

def calc_spsp(b1_map, seq_filename, tp, sys, myocardium):
    if seq_filename != 'dicom':
        # Load sequence and defs
        seq = pp.Sequence()
        seq.read(seq_filename)
        # Get parameters
        defs = seq.definitions
        fov = defs['FOV'] # [m]
        nx = defs['Nx']

    else:
        b1_map, nx, fov = read_dicom.dicom_b1_siemens(b1_map)
    
    dx = fov[0] / nx # Pixel spacing [m]
    dy = fov[1] / nx # Pixel spacing (y) [m]
    gambar = sys.gamma/1e4
    
    # Make meshgrid
    x_lin = (np.arange(nx) - nx / 2) * dx
    y_lin = (np.arange(nx) - nx / 2) * dy
    x, y = np.meshgrid(x_lin, y_lin)
    
    # Draw mask on B1 map
    mask = draw_roi(b1_map, myocardium)
    x_masked = x[mask]
    y_masked = y[mask]
    b1_masked = b1_map[mask]
    
    # Fix requested durations to gradient raster
    target_duration_s = align_grad_raster(250e-6, sys)
    min_flat_time_s = align_grad_raster(50e-6, sys)
    
    base_trap, rewinder = find_optimal_spsp_pair(sys, target_duration_s, min_flat_time_s)

    dt = sys.rf_raster_time # This might work?
    # dt = 1e-5 # Fix equal to CA's code 
    nt = int(np.round(base_trap.flat_time / dt))
    flat_duration = base_trap.flat_time
    g_amp_Hz = base_trap.amplitude 
    print(f"Gradient amplitude = {np.round(g_amp_Hz,2)} Hz/m")
    
    angles = np.arange(0, 91)
    lambda_reg = 0.08 
    
    # Time array centered on the plateau
    t_arr = np.linspace(-flat_duration/2, flat_duration/2, nt)
    cost = np.zeros(len(angles))
    
    # Our goal is a perfectly uniform flip (1.0) inside the mask
    target_b1 = np.ones_like(b1_masked)
    
    for jj, angle in enumerate(angles):
        # Rotate the spatial coordinates
        spatial_proj = x_masked * np.cos(np.deg2rad(angle)) + y_masked * np.sin(np.deg2rad(angle))
        phase = 2 * np.pi * g_amp_Hz * np.outer(spatial_proj, t_arr)
        
        # Build the system matrix A
        A = b1_masked[:, None] * np.exp(1j * phase) * dt * 2 * np.pi * gambar
        A_H = A.conj().T

        regularizer = lambda_reg * np.eye(nt)
        
        # Solve for RF weights
        rf_weights = np.linalg.solve(A_H @ A + regularizer, A_H @ target_b1)
        
        # Calculate cost (residual + penalty)
        residual = (A @ rf_weights) - target_b1
        cost[jj] = np.linalg.norm(residual, 2)**2 + lambda_reg * np.linalg.norm(rf_weights, 2)**2
    
    idx = np.argmin(cost)
    angle0 = angles[idx]
    print(f"Optimal angle found: {angle0} degrees")
    
    # Recalculate RF weights at the absolute best angle
    spatial_proj = x_masked * np.cos(np.deg2rad(angle0)) + y_masked * np.sin(np.deg2rad(angle0))
    phase = 2 * np.pi * g_amp_Hz * np.outer(spatial_proj, t_arr)
    A = b1_masked[:, None] * np.exp(1j * phase) * dt * 2 * np.pi * gambar
    A_H = A.conj().T
    regularizer = lambda_reg * np.eye(nt)
    final_rf_weights = np.linalg.solve(A.conj().T @ A + regularizer, A.conj().T @ target_b1)
    
    # Generate final PyPulseq Gradients by copying the exact rasterized timing
    gx_lobe1 = pp.make_trapezoid(channel='x', system=sys, 
                                 amplitude=base_trap.amplitude * np.cos(np.deg2rad(angle0)),
                                 rise_time=base_trap.rise_time, 
                                 flat_time=base_trap.flat_time, 
                                 fall_time=base_trap.fall_time)
    
    gy_lobe1 = pp.make_trapezoid(channel='y', system=sys, 
                                 amplitude=base_trap.amplitude * np.sin(np.deg2rad(angle0)),
                                 rise_time=base_trap.rise_time, 
                                 flat_time=base_trap.flat_time, 
                                 fall_time=base_trap.fall_time)
    
    # Generate Rewinders
    gx_rewind = pp.make_trapezoid(channel='x', system=sys, 
                                  amplitude=rewinder.amplitude * np.cos(np.deg2rad(angle0)),
                                  rise_time=rewinder.rise_time, 
                                  flat_time=rewinder.flat_time,
                                  fall_time=rewinder.fall_time)
    
    gy_rewind = pp.make_trapezoid(channel='y', system=sys, 
                                  amplitude=rewinder.amplitude * np.sin(np.deg2rad(angle0)),
                                  rise_time=rewinder.rise_time, 
                                  flat_time=rewinder.flat_time, 
                                  fall_time=rewinder.fall_time)
    
    total_duration = pp.calc_duration(gx_lobe1) + pp.calc_duration(gx_rewind)
    
    # Pulse time (for Gauss)
    dur = tp

    # Calculate how many subpulses fit in Gaussian pulse
    num_subpulses = int(np.floor(dur / total_duration))
    
    # --- We actually have to generate special gradients (with special delays) for the beginning of the pulse train
    # Write special prewinders first
    prewinder_x = pp.make_trapezoid(channel='x', area=-gx_lobe1.area / 2, system=sys)
    prewinder_y = pp.make_trapezoid(channel='y', area=-gy_lobe1.area / 2, system=sys)
    # Check duration
    prewinder_dur = max(pp.calc_duration(prewinder_x), pp.calc_duration(prewinder_y))
    # Rewrite with common duration
    prewinder_x = pp.make_trapezoid(channel='x', area=-gx_lobe1.area / 2, duration=prewinder_dur, system=sys)
    prewinder_y = pp.make_trapezoid(channel='y', area=-gy_lobe1.area / 2, duration=prewinder_dur, system=sys)
    # Calculate delay
    if prewinder_dur + base_trap.rise_time <= sys.rf_dead_time:
        prewinder_grad_delay = sys.rf_dead_time - (prewinder_dur + base_trap.rise_time)
        prewinder_rf_delay = 0
    else:
        prewinder_grad_delay = 0
        prewinder_rf_delay = prewinder_dur + base_trap.rise_time - sys.rf_dead_time
    # Align delays
    raster = sys.grad_raster_time
    prewinder_grad_delay = np.round(prewinder_grad_delay / raster) * raster
    prewinder_rf_delay = np.round(prewinder_rf_delay / raster) * raster
    
    # Concatenate an arbitrary number of waveforms AND/OR time delays
    def make_combined_times_amps(*items):
        times = [0.0]
        amps = [0.0]
        current_time = 0.0
        raster = sys.grad_raster_time  # capture from outer scope
        
        for item in items:
            if isinstance(item, (int, float, np.number)):
                if item > 0:
                    current_time += item
                    current_time = np.round(current_time / raster) * raster
                    times.append(current_time)
                    amps.append(0.0)
            else:
                for dt_seg, target_amp in [(item.rise_time, item.amplitude),
                                            (item.flat_time, item.amplitude),
                                            (item.fall_time, 0.0)]:
                    if dt_seg > 0:
                        current_time += dt_seg
                        current_time = np.round(current_time / raster) * raster
                        times.append(current_time)
                        amps.append(target_amp)
                        
        return np.array(times), np.array(amps)
    
    gcx_times, gcx_amp = make_combined_times_amps(prewinder_x, gx_lobe1)
    gcy_times, gcy_amp = make_combined_times_amps(prewinder_y, gy_lobe1)
    gx_lobe0 = pp.make_extended_trapezoid(channel='x', amplitudes=gcx_amp, times=gcx_times, system=sys)
    gy_lobe0 = pp.make_extended_trapezoid(channel='y', amplitudes=gcy_amp, times=gcy_times, system=sys)
    
    # --- Re-generate gradients with the delay ---
    # If the rise time is shorter than the dead time, we pad the gradient start
    grad_delay = max(0, sys.rf_dead_time - base_trap.rise_time)

    gx_lobe1_del = pp.make_trapezoid(channel='x', system=sys, 
                                 amplitude=base_trap.amplitude * np.cos(np.deg2rad(angle0)),
                                 rise_time=base_trap.rise_time, 
                                 flat_time=base_trap.flat_time, 
                                 fall_time=base_trap.fall_time,
                                 delay=grad_delay) # <-- Shift gradient right
    
    gy_lobe1_del = pp.make_trapezoid(channel='y', system=sys, 
                                 amplitude=base_trap.amplitude * np.sin(np.deg2rad(angle0)),
                                 rise_time=base_trap.rise_time, 
                                 flat_time=base_trap.flat_time, 
                                 fall_time=base_trap.fall_time,
                                 delay=grad_delay) # <-- Shift gradient right
    
    # Make final rewinders
    final_rewinder_x = pp.make_trapezoid(channel='x', area=-gx_lobe1.area / 2, system=sys)
    final_rewinder_y = pp.make_trapezoid(channel='y', area=-gy_lobe1.area / 2, system=sys)
    final_rewind_dur = max(pp.calc_duration(final_rewinder_x), pp.calc_duration(final_rewinder_y))
    final_rewind_dur = align_grad_raster(final_rewind_dur, sys)
    final_rewinder_x = pp.make_trapezoid(channel='x', area=-gx_lobe1.area / 2, duration=final_rewind_dur, system=sys)
    final_rewinder_y = pp.make_trapezoid(channel='y', area=-gy_lobe1.area / 2, duration=final_rewind_dur, system=sys)

    
    # --- Generate the Gaussian Envelope ---
    gauss_pulse = pp.make_gauss_pulse(flip_angle=np.pi, 
                               duration=num_subpulses*sys.rf_raster_time, 
                               time_bw_product=0.2,
                               apodization=0.5, 
                               delay=100e-6, 
                               system=sys) # We know raster is 1e-6 s
    weights = gauss_pulse.signal / np.max(np.abs(gauss_pulse.signal))
    
    # --- Make concatenated waveforms --- 
    max_magnitude = np.max(np.abs(final_rf_weights))
    norm_subpulse = final_rf_weights / max_magnitude

    # intrapulse_delay = pp.calc_duration(gx_rewind) + gx_lobe1.rise_time + gx_lobe1.fall_time
    # intrapulse_delay = int(np.ceil(intrapulse_delay / sys.grad_raster_time) * sys.grad_raster_time / dt)
    intrapulse_delay_s = pp.calc_duration(gx_rewind) + gx_lobe1.rise_time + gx_lobe1.fall_time
    intrapulse_delay_s = align_grad_raster(intrapulse_delay_s, sys)
    intrapulse_delay = int(np.round(intrapulse_delay_s / dt))

    pulse_shape = []
    grad_x_list = []
    grad_y_list = []

    for n, weight in enumerate(weights):
        # Add initial prewinder before the first subpulse
        if n == 0:
            # RF delay is an array of zeros 
            # pulse_shape.append(np.zeros(int(prewinder_rf_delay / dt)))
            pulse_shape.append(np.zeros(int(np.round(prewinder_rf_delay / dt))))
            
            # Gradient delay is just the scalar time in seconds
            grad_x_list.append(prewinder_grad_delay)
            grad_y_list.append(prewinder_grad_delay)
            
            grad_x_list.append(prewinder_x)
            grad_y_list.append(prewinder_y)
            
        # Add the main subpulse and corresponding gradient lobe
        scaled_subpulse = norm_subpulse * weight
        pulse_shape.append(scaled_subpulse)
        grad_x_list.append(gx_lobe1)
        grad_y_list.append(gy_lobe1)
        
        # Add the delay/rewinder between subpulses, or the final rewinder at the end
        if n < len(weights) - 1:
            pulse_shape.append(np.zeros(intrapulse_delay))
            grad_x_list.append(gx_rewind)
            grad_y_list.append(gy_rewind)
        elif n == len(weights) - 1:

            # pulse_shape.append(np.zeros(int(final_rewind_dur / dt))) 
            pulse_shape.append(np.zeros(int(np.round(final_rewind_dur / dt))))
            grad_x_list.append(final_rewinder_x)
            grad_y_list.append(final_rewinder_y)

    # Concatenate the list of RF arrays into a single continuous 1D array
    full_rf = np.concatenate(pulse_shape)
        
    # Unpack the accumulated lists of gradient objects/delays into our modified helper function
    gcx_times, gcx_amp = make_combined_times_amps(*grad_x_list)
    gcy_times, gcy_amp = make_combined_times_amps(*grad_y_list)
    
    # Create the final, single extended trapezoids for the entire pulse train
    full_gx = pp.make_extended_trapezoid(channel='x', amplitudes=gcx_amp, times=gcx_times, system=sys)
    full_gy = pp.make_extended_trapezoid(channel='y', amplitudes=gcy_amp, times=gcy_times, system=sys)
        
    # Package outputs
    spsp_objects = {
        'gx_lobe0': gx_lobe0,
        'gy_lobe0': gy_lobe0,
        'gx_lobe1': gx_lobe1_del,
        'gy_lobe1': gy_lobe1_del,
        'gx_rewind': gx_rewind,
        'gy_rewind': gy_rewind,
        'full_gx': full_gx,
        'full_gy': full_gy, 
        'full_rf': full_rf,
        'prewinder_rf_delay': prewinder_rf_delay,
        'num_subpulses': num_subpulses,
        'weights': weights,
        'final_rf_weights': final_rf_weights
        }
    
    # Apply the RF weights to the entire spatial grid, not just the mask
    spatial_proj_full = x * np.cos(np.deg2rad(angle0)) + y * np.sin(np.deg2rad(angle0))
    phase_full = 2 * np.pi * g_amp_Hz * np.outer(spatial_proj_full.ravel(), t_arr)
    A_full = b1_map.ravel()[:, None] * np.exp(1j * phase_full) * dt * 2 * np.pi * gambar
    
    # Reshape the 1D result back into a 2D image
    b1_full_sim = np.abs(A_full @ final_rf_weights).reshape(b1_map.shape)

    grad_times = [
        0, 
        base_trap.rise_time, 
        base_trap.rise_time + base_trap.flat_time, 
        base_trap.rise_time + base_trap.flat_time + base_trap.fall_time,
        base_trap.rise_time + base_trap.flat_time + base_trap.fall_time + rewinder.rise_time,
        total_duration
    ]
    # Get the amplitudes at those corners
    gx_points = [0, gx_lobe1.amplitude, gx_lobe1.amplitude, 0, gx_rewind.amplitude, 0]
    gy_points = [0, gy_lobe1.amplitude, gy_lobe1.amplitude, 0, gy_rewind.amplitude, 0]
    
    plot_b1_roi_only(b1_map, mask, final_rf_weights, A, b1_full_sim, t_arr, grad_times, gx_points, gy_points)
        
    return spsp_objects
    
def plot_b1_roi_only(b1_map, mask, final_rf_weights, A, b1_full_sim, t_arr, grad_times, gx_points, gy_points):
    # In the design: A * rf = target_b1 (ideally ones)
    roi_result = np.abs(A @ final_rf_weights)
    
    # Map the ROI result back into a 2D image
    tailored_roi_map = np.zeros_like(b1_map)
    tailored_roi_map[mask] = roi_result
    
    # Create the plots (made slightly larger to accommodate 6 panels)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    
    # --- Top Left: Experimental Full B1 ---
    ax_experimental = axs[0, 0]
    im1 = ax_experimental.imshow(b1_map)
    ax_experimental.set_title("Experimental B1 Map")
    plt.colorbar(im1, ax=ax_experimental, fraction=0.046, pad=0.04)
    ax_experimental.axis('off')
    
    # --- Top Middle: Simulated Full B1 ---
    ax_tailored = axs[0, 1]
    im2 = ax_tailored.imshow(b1_full_sim)
    ax_tailored.set_title("Simulated B1 (tailored)")
    plt.colorbar(im2, ax=ax_tailored, fraction=0.046, pad=0.04)
    ax_tailored.axis('off')
    
    # --- Top Right: RF Subpulse ---
    ax_rf = axs[0, 2]
    
    grad_times_us = np.array(grad_times) * 1e6
    flat_start_us = grad_times_us[1]  
    flat_end_us   = grad_times_us[2]  
    
    rf_t_us = np.linspace(flat_start_us, flat_end_us, len(final_rf_weights))
    rf_t_step = rf_t_us[1] - rf_t_us[0]

    num_prefix = int(np.round(flat_start_us / rf_t_step))
    num_suffix = int(np.round((grad_times_us[-1] - flat_end_us) / rf_t_step))
    
    t_prefix = np.linspace(0, flat_start_us, num_prefix, endpoint=False)
    t_suffix = np.linspace(flat_end_us, grad_times_us[-1], num_suffix + 1)[1:] # Start just after flat_end
    
    fine_t_axis = np.concatenate([t_prefix, rf_t_us, t_suffix])
    
    rf_mag = np.concatenate([
        np.zeros(len(t_prefix)), 
        np.abs(final_rf_weights), 
        np.zeros(len(t_suffix))
    ])
    
    rf_phase = np.concatenate([
        np.zeros(len(t_prefix)), 
        np.angle(final_rf_weights), 
        np.zeros(len(t_suffix))
    ])
    
    ax_rf.plot(fine_t_axis, rf_mag, linewidth=2.5, label='Magnitude')
    ax_rf.plot(fine_t_axis, rf_phase, linewidth=2.5, label='Phase (rad)')
    
    ax_rf.set_title('RF Subpulse (Matched Spacing)')
    ax_rf.set_xlabel('Time (μs)')
    ax_rf.set_xlim([0, grad_times_us[-1]])
    ax_rf.legend()
    ax_rf.grid(True, linestyle='--', alpha=0.6)
    
    # --- Bottom Left: Original B1 in ROI ---
    ax_exp_roi = axs[1, 0]
    im3 = ax_exp_roi.imshow(b1_map * mask)
    ax_exp_roi.set_title(f"Original B1 (ROI)\nMean: {np.mean(b1_map[mask]):.3f} ± {np.std(b1_map[mask]):.3f}")
    plt.colorbar(im3, ax=ax_exp_roi, fraction=0.046, pad=0.04)
    ax_exp_roi.axis('off')
    vmin, vmax = np.min(b1_map*mask), np.max(b1_map*mask)
    
    # --- Bottom Middle: Tailored Result in ROI ---
    ax_tail_roi = axs[1, 1]
    im4 = ax_tail_roi.imshow(tailored_roi_map, vmin=vmin, vmax=vmax) 
    ax_tail_roi.set_title(f"Tailored Result (ROI)\nMean: {np.mean(roi_result):.3f} ± {np.std(roi_result):.3f}")
    plt.colorbar(im4, ax=ax_tail_roi, fraction=0.046, pad=0.04)
    ax_tail_roi.axis('off')

    # --- Bottom Right: Gradient Waveforms ---
    ax_grad = axs[1, 2]
    grad_times_us = np.array(grad_times) * 1e6
    # Converting Hz/m to kHz/m for cleaner y-axis numbers
    gx_kHz_m = np.array(gx_points) / 1000 
    gy_kHz_m = np.array(gy_points) / 1000
    
    ax_grad.plot(grad_times_us, gx_kHz_m, linewidth=2.5, label='gx')
    ax_grad.plot(grad_times_us, gy_kHz_m, linewidth=2.5, label='gy')
    ax_grad.set_title('Gradients')
    ax_grad.set_xlabel('Time (μs)')
    ax_grad.set_ylabel('Amplitude (kHz/m)')
    ax_grad.legend()
    ax_grad.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()