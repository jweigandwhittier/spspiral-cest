#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Siemens TWIX (R=2 GRAPPA + Readout OS) Reconstruction Pipeline
Author: Jonah 
Date: July 7, 2026

Reconstructs an inversion-recovery T1 series, then lets you pick each phantom
vial (utils.vial_roi) on the longest-TI image and plots/saves the per-vial
signal vs TI. Run with an interactive backend (e.g. `%matplotlib qt` in Spyder);
picker controls are listed in utils/vial_roi.py.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu
import twixtools

# Make the project's utils package importable when run from a subfolder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import vial_roi

try:
    import pygrappa
    HAS_PYGRAPPA = True
except ImportError as err:
    HAS_PYGRAPPA = False
    print(f"WARNING: could not import pygrappa ({err!r})")
    print(f"         Python: {sys.executable}")
    print("         Falling back to line interpolation (expect R=2 aliasing).")


def load_and_sort_twix(filename, full_matrix_size=64):
    twix_data = twixtools.read_twix(filename, parse_pmu=False)
    mdbs = twix_data[-1]['mdb']
    
    first_scan = next(mdb for mdb in mdbs if mdb.is_image_scan())
    n_coils, n_readout = first_scan.data.shape
    
    kspace_sub = np.zeros((full_matrix_size, n_coils, n_readout), dtype=np.complex128)
    acs_dict = {}
    lines_mapped = 0
    
    for mdb in mdbs:
        # Capture ALL calibration lines by their spatial row index (cLin)
        if mdb.is_flag_set('PATREFSCAN') or mdb.is_flag_set('PATREFANDIMASCAN'):
            acs_dict[mdb.cLin] = mdb.data.astype(np.complex128)
            
        if mdb.is_image_scan():
            line_idx = mdb.cLin
            if line_idx < full_matrix_size:
                kspace_sub[line_idx, :, :] = mdb.data.astype(np.complex128)
                lines_mapped += 1
                
    if acs_dict:
        sorted_acs_indices = sorted(acs_dict.keys())
        calib_data = np.array([acs_dict[idx] for idx in sorted_acs_indices], dtype=np.complex128)
    else:
        calib_data = None
        
    return kspace_sub, calib_data, acs_dict


def reconstruct_grappa(kspace_sub, calib_data, acs_dict):
    print("\n--- 2. Parallel Imaging (GRAPPA) Synthesis ---")
    if HAS_PYGRAPPA and calib_data is not None and len(calib_data) > 4:
        print("Running pygrappa.cgrappa with tuned (3, 3) kernel and low regularization...")
        # 1. TUNE KERNEL: Use (3, 3) kernel and lower lamda to prevent overfitting on 26 lines
        kspace_recon = pygrappa.cgrappa(
            kspace=kspace_sub,
            calib=calib_data,
            kernel_size=(3, 3),  # Swapped from (5, 5) for tighter fit
            lamda=0.0001,        # Swapped from 0.01
            coil_axis=1
        )
        
        # 2. ACS DATA SUBSTITUTION
        # Overwrite the synthesized center lines with true measured ground-truth data
        print(f"Executing Data Substitution: Inserting {len(acs_dict)} measured ACS lines into k-space center...")
        for idx, true_data in acs_dict.items():
            if idx < kspace_recon.shape[0]:
                kspace_recon[idx, :, :] = true_data
                
    else:
        print("Falling back to 1D linear interpolation across un-sampled lines...")
        kspace_recon = kspace_sub.copy()
        for i in range(1, kspace_recon.shape[0] - 1, 2):
            kspace_recon[i] = 0.5 * (kspace_recon[i - 1] + kspace_recon[i + 1])
            
    return kspace_recon


def perform_fft_and_crop(kspace_recon, target_resolution=64):
    """
    Applies 2D Inverse FFT across Phase Encode and Readout axes, then removes
    Readout Oversampling by cropping the central spatial pixels.
    """
    print("\n--- 3. 2D Inverse FFT & Readout OS Removal ---")
    img_coils_oversampled = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace_recon, axes=(0, 2)), 
            axes=(0, 2)
        ), 
        axes=(0, 2)
    )
    print(f"Oversampled Image Shape  : {img_coils_oversampled.shape}")
    
    current_readout = img_coils_oversampled.shape[2]
    crop_start = (current_readout - target_resolution) // 2
    crop_end = crop_start + target_resolution
    
    img_coils_cropped = img_coils_oversampled[:, :, crop_start:crop_end]
    print(f"Cropped Image Shape      : {img_coils_cropped.shape} [Y, Coils, X]")
    
    return img_coils_cropped


def combine_coils(img_coils):
    """
    Combines multi-channel coil arrays using Sum-of-Squares (SoS).
    Returns both magnitude SoS and the complex array (for Phase-Sensitive IR).
    """
    print("\n--- 4. Coil Combination ---")
    img_sos = np.sqrt(np.sum(np.abs(img_coils)**2, axis=1))
    print(f"Final Image Shape        : {img_sos.shape} [Y, X]")
    return img_sos, img_coils


def restore_signed_t1_series(img_coils_series, ti_list, ref_idx=None):
    """
    Phase-sensitive IR: use the coil images at the reference TI (default: longest,
    fully recovered) as complex sensitivities, combine every TI against them and
    keep the real part, so inverted magnetization stays negative.
    img_coils_series: [n_TI, Y, Coils, X] -> returns ([n_TI, Y, X], ref_idx)
    """
    ti_arr = np.array(ti_list)
    if ref_idx is None:
        ref_idx = int(np.argmax(ti_arr))

    sens = img_coils_series[ref_idx]
    sens_power = np.sum(np.abs(sens) ** 2, axis=1)
    sens_power_safe = np.where(sens_power < 1e-9, 1e-9, sens_power)

    n_ti = img_coils_series.shape[0]
    signed = np.zeros((n_ti,) + sens_power.shape, dtype=np.float64)
    for t in range(n_ti):
        combined = np.sum(np.conj(sens) * img_coils_series[t], axis=1) / sens_power_safe
        signed[t] = np.real(combined)
    return signed, ref_idx


def ir_signed_model(ti, a, b, t1):
    return a - b * np.exp(-ti / t1)


def _fit_ir(tis, signal, max_t1=5000.0, maxfev=2000):
    """Signed IR fit. Returns (popt [a, b, T1], T1 standard error) or (None, nan) on failure."""
    a_guess = signal[np.argmax(tis)]
    p0 = [a_guess, 2.0 * a_guess, 1000.0]
    bounds = ([-np.inf, 0.0, 1.0], [np.inf, np.inf, max_t1])
    try:
        popt, pcov = curve_fit(ir_signed_model, tis, signal, p0=p0, bounds=bounds, maxfev=maxfev)
        t1_se = float(np.sqrt(pcov[2, 2])) if np.isfinite(pcov[2, 2]) else np.nan
        return popt, t1_se
    except (RuntimeError, ValueError):
        return None, np.nan


def calculate_t1_map(signed_series, mag_series, inversion_times, max_t1=5000.0):
    """
    Voxelwise signed-IR T1 fit inside an Otsu mask of the mean magnitude.
    signed_series, mag_series: [n_TI, Y, X]. Returns (t1_map [Y, X] in ms, mask).
    """
    print("\n--- Voxel-by-Voxel T1 Mapping (signed IR) ---")
    mean_mag = mag_series.mean(axis=0)
    mask = mean_mag > threshold_otsu(mean_mag)
    tis = np.asarray(inversion_times, dtype=np.float64)

    t1_map = np.zeros(mean_mag.shape, dtype=np.float32)
    for r, c in zip(*np.nonzero(mask)):
        popt, _ = _fit_ir(tis, signed_series[:, r, c].astype(np.float64), max_t1=max_t1, maxfev=1000)
        if popt is not None:
            t1_map[r, c] = popt[2]
    print(f"Mapped {np.count_nonzero(t1_map)} / {mask.sum()} masked voxels.")
    return t1_map, mask


def batch_reconstruct_t1_series(data_dir, file_list, ti_list, target_res=64):
    """
    Loops through the specified TWIX files and reconstructs each inversion time.
    Returns magnitude [n_TIs, Y, X] and complex coil images [n_TIs, Y, Coils, X].
    """
    n_times = len(file_list)
    all_mag_images = None
    all_complex_images = None
    
    print(f"Starting batch reconstruction of {n_times} datasets...")
    print("--------------------------------------------------")
    
    for idx, (filename, ti) in enumerate(zip(file_list, ti_list)):
        full_path = os.path.join(data_dir, filename)
        print(f"\n[{idx + 1}/{n_times}] Processing: {filename} (TI = {ti} ms)")
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Target raw file not found at: {full_path}")
            
        kspace_sub, calib_data, acs_dict = load_and_sort_twix(full_path, full_matrix_size=target_res)
        kspace_recon = reconstruct_grappa(kspace_sub, calib_data, acs_dict)
        img_coils = perform_fft_and_crop(kspace_recon, target_resolution=target_res)
        img_mag, img_complex = combine_coils(img_coils)
        img_mag = np.fliplr(np.rot90(img_mag, k=1))
        img_complex = np.flip(np.rot90(img_complex, k=1, axes=(0, 2)), axis=2)
        
        if all_mag_images is None:  # allocate on the first file so the coil count comes from the data
            all_mag_images = np.zeros((n_times,) + img_mag.shape)
            all_complex_images = np.zeros((n_times,) + img_complex.shape, dtype=np.complex128)
        all_mag_images[idx] = img_mag
        all_complex_images[idx] = img_complex
        
    print("\n--------------------------------------------------")
    print("All reconstructions complete!")
    return all_mag_images, all_complex_images


if __name__ == '__main__':
    base_data_dir = '/Volumes/JWW-BIC/Phantom_Zspec/Siemens'
    out_dir = 'results/t1_series'
    MAX_VIAL_RADIUS_PX = 15   # Longest ray for wall detection [px]; set from your FOV / vial size
    MARGIN_PX = 1.5           # Initial ROI margin (adjust live with +/-)
    
    target_files = [
        'meas_MID00128_FID56460_tir_rs_1mm_TI50.dat',
        'meas_MID00129_FID56461_tir_rs_1mm_TI200.dat',
        'meas_MID00130_FID56462_tir_rs_1mm_TI400.dat',
        'meas_MID00131_FID56463_tir_rs_1mm_TI1350.dat',
        'meas_MID00133_FID56465_tir_rs_1mm_TI2700.dat',
        'meas_MID00134_FID56466_tir_rs_1mm_TI5200.dat'
    ]
    inversion_times = np.array([50, 200, 400, 1350, 2700, 5200])  # ms
    
    # Step 1: Reconstruct
    images_mag, images_complex = batch_reconstruct_t1_series(
        base_data_dir, target_files, inversion_times, target_res=64)
    os.makedirs(out_dir, exist_ok=True)
    
    # Step 2: Pick vials on the longest TI (closest to fully recovered, best vial contrast)
    ref_idx = int(np.argmax(inversion_times))
    ref_image = images_mag[ref_idx]
    res = vial_roi.pick_vials(ref_image, MAX_VIAL_RADIUS_PX, margin=MARGIN_PX,
                              title_prefix=f'TI = {inversion_times[ref_idx]} ms')
    vial_roi.save_roi_overlay(ref_image, res, os.path.join(out_dir, 'vial_rois.png'))
    
    # Step 3: Signed (phase-sensitive) IR series
    images_signed, phase_ref_idx = restore_signed_t1_series(images_complex, inversion_times)
    print(f"Phase reference: TI = {inversion_times[phase_ref_idx]} ms")
    
    # Step 4: Voxelwise T1 map
    T1_FIT_MAX = 5000.0   # Upper bound on T1 in the fit [ms]
    T1_DISPLAY_MAX = 4000.0
    t1_map, t1_mask = calculate_t1_map(images_signed, images_mag, inversion_times, max_t1=T1_FIT_MAX)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(t1_map, cmap='inferno', vmin=0, vmax=T1_DISPLAY_MAX)
    for i, (v, r) in enumerate(zip(res['vials'], res['roi_radii'])):
        rc, cc = v['center']
        ax.add_patch(Circle((cc, rc), r, fill=False, ec='cyan', lw=1))
        ax.text(cc, rc, str(i + 1), color='cyan', ha='center', va='center', fontsize=9, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('T1 [ms]')
    ax.set_title('T1 map (signed IR fit)')
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 't1_map.png'), dpi=150, bbox_inches='tight')
    
    # Step 5: Per-vial fits on the ROI-mean signed signal (+ voxelwise map stats in the ROI)
    tis = inversion_times.astype(np.float64)
    sig_mean, sig_std = vial_roi.roi_stats(images_signed, res['masks'])
    n_vials = len(res['vials'])
    roi_popt = np.full((n_vials, 3), np.nan)   # [a, b, T1]
    roi_t1_se = np.full(n_vials, np.nan)
    map_t1_mean = np.full(n_vials, np.nan)
    map_t1_std = np.full(n_vials, np.nan)
    print("\n--- Per-vial T1 ---")
    for i, m in enumerate(res['masks']):
        popt, se = _fit_ir(tis, sig_mean[i], max_t1=T1_FIT_MAX)
        if popt is not None:
            roi_popt[i], roi_t1_se[i] = popt, se
        vals = t1_map[m & (t1_map > 0)]
        if vals.size:
            map_t1_mean[i], map_t1_std[i] = vals.mean(), vals.std()
        print(f"Vial {i + 1}: ROI-mean fit T1 = {roi_popt[i, 2]:.0f} ± {roi_t1_se[i]:.0f} ms | "
              f"voxelwise map {map_t1_mean[i]:.0f} ± {map_t1_std[i]:.0f} ms ({vals.size} px)")
    
    fig = vial_roi.plot_vial_grid(tis, sig_mean, sig_std, res['colors'],
                                  xlabel='TI [ms]', ylabel='Signed signal [a.u.]',
                                  title='Signed IR signal per vial (mean ± pixelwise SD) with fit',
                                  marker='o')
    ti_fine = np.linspace(0, tis.max(), 300)
    for i in range(n_vials):
        ax = fig.axes[i]
        if np.all(np.isfinite(roi_popt[i])):
            ax.plot(ti_fine, ir_signed_model(ti_fine, *roi_popt[i]), 'k-', lw=1)
            ax.set_title(f'Vial {i + 1}: T1 = {roi_popt[i, 2]:.0f} ms', color=res['colors'][i],
                         fontweight='bold')
        ax.axhline(0, color='0.5', lw=0.8)
    fig.savefig(os.path.join(out_dir, 't1_signal_grid.png'), dpi=150)
    plt.show()
    
    # Step 6: Save images, T1 map, per-vial curves/fits and ROIs
    output_fn = os.path.join(out_dir, 'recon_t1_series.npz')
    np.savez_compressed(
        output_fn,
        ti=inversion_times,                # [n_TIs] ms
        images_mag=images_mag,             # [n_TIs, Y, X]
        images_complex=images_complex,     # [n_TIs, Y, Coils, X]
        images_signed=images_signed,       # [n_TIs, Y, X]  phase-sensitive real part
        phase_ref_ti=inversion_times[phase_ref_idx],
        t1_map=t1_map,                     # [Y, X] ms, 0 = outside mask / failed fit
        t1_mask=t1_mask,                   # [Y, X] Otsu mask used for the map
        signal_mean=sig_mean,              # [n_vials, n_TIs]  signed ROI mean
        signal_std=sig_std,                # [n_vials, n_TIs]  pixelwise SD
        roi_fit_params=roi_popt,           # [n_vials, 3]  a, b, T1 [ms] from the ROI-mean fit
        roi_t1_ms=roi_popt[:, 2],          # [n_vials]
        roi_t1_se_ms=roi_t1_se,            # [n_vials]  fit standard error
        map_t1_mean_ms=map_t1_mean,        # [n_vials]  voxelwise map, mean in ROI
        map_t1_std_ms=map_t1_std,          # [n_vials]
        ref_ti=inversion_times[ref_idx],
        img_ref=ref_image,
        data_dir=base_data_dir, file_list=np.array(target_files),
        **vial_roi.roi_geometry(res))
    np.savetxt(os.path.join(out_dir, 'vial_t1.csv'),
               np.column_stack([np.arange(1, n_vials + 1), roi_popt[:, 2], roi_t1_se,
                                map_t1_mean, map_t1_std]),
               delimiter=',', header='vial,roi_fit_t1_ms,roi_fit_t1_se_ms,map_t1_mean_ms,map_t1_std_ms',
               comments='', fmt=['%d', '%.2f', '%.2f', '%.2f', '%.2f'])
    print(f"Saved T1 map, per-vial T1 and ROIs to: {out_dir}/")