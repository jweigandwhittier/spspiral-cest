#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siemens Z-spectrum phantom analysis.

Takes the single combined Siemens ZSPEC acquisition (FLAG_ZSPEC=True, one
.dat with every offset back-to-back, recovery delays in between), reconstructs
one image per offset, lets you pick each phantom vial on the M0 image
(utils.vial_roi), and plots/saves the Z-spectrum per vial.

Run with an interactive backend (e.g. `%matplotlib qt` in Spyder); picker
controls are listed in utils/vial_roi.py.
"""

# %% Step 0: Parameters
import os
import numpy as np
import matplotlib.pyplot as plt

data_fn = '/Volumes/JWW-BIC/Phantom_Zspec/SPSP/meas_MID00070_FID57059_zspec_gauss.dat'  # combined ZSPEC .dat
seq_fn = '/Volumes/JWW-BIC/Phantom_Zspec/SPSP/continuous_spiral_zspec_spsp.seq'  # matching .seq

out_dir = 'results/gauss_saturation'
run_label = 'compare_gauss'  # Used in the saved spectra filename and as the plot label when comparing
cache_fn = os.path.join(out_dir, 'zspec_recon_cache.npz')
USE_CACHE = False        # Reuse images + trajectory from a previous run (recon is the slow part)
N_SKIP_READOUTS = 0     # Leading readouts in the .dat to discard (dummy/noise scans), if any

M0_THRESH_PPM = 50.0    # Offsets with |offset| >= this are treated as M0 / S0 references
NORM_REF = 'm0'        # 'm0': divide by the M0 image | 'far': divide by the mean of the far offsets (diagnostic)
FAR_PPM = 9.5          # 'far' reference = saturated offsets with |offset| >= this (here +/-10 ppm)

M0_THRESH_PPM = 50.0    # Offsets with |offset| >= this are treated as M0 / S0 references
MAX_VIAL_RADIUS_MM = 100 # Longest ray [mm]; keeps rays from running into neighbouring vials
MARGIN_PX = 1.5         # Initial ROI margin (adjust live with +/-)

# %% Step 1: Sequence info + trajectory, raw data, recon (cached)
from utils import extract_raw, recon, vial_roi
 
os.makedirs(out_dir, exist_ok=True)
seq = extract_raw.load_seq(seq_fn)
defs = seq.definitions
offsets_acq = np.atleast_1d(np.asarray(defs['Offsets_ppm'], dtype=float))  # acquisition order
nx = int(defs['Nx'])
fov = float(np.atleast_1d(defs['FOV'])[0])
n_adc = int(defs['MaxAdcSegmentLength'])
n_trs = int(defs['N_TRs_per_offset'] if 'N_TRs_per_offset' in defs else defs['N_TRs'])
n_offsets = len(offsets_acq)
print(f'{n_offsets} offsets x {n_trs} TRs, {n_adc} ADC samples, Nx={nx}, FOV={fov*1e3:.0f} mm')
 
cache_ok = False
if USE_CACHE and os.path.exists(cache_fn):
    c = np.load(cache_fn)
    if (c['image_stack'].shape[0] == n_offsets and np.allclose(c['offsets_acq'], offsets_acq)
            and str(c['data_fn']) == data_fn):
        image_stack = c['image_stack']
        cache_ok = True
        print(f'Loaded cached recon from {cache_fn}')
    else:
        print('Cache does not match this data/sequence; reconstructing.')
 
if not cache_ok:
    # Trajectory: the golden-angle train restarts at every offset, so the first
    # offset's n_trs readouts define the trajectory for all offsets.
    # (calculate_kspace over the full ZSPEC sequence is slow; that's why we cache.)
    print('Computing k-space trajectory (full sequence, can take a while)...')
    k_traj_adc, _, _, _, _ = seq.calculate_kspace(trajectory_delay=0)
    k_traj_adc = k_traj_adc[:, :n_adc * n_trs]
    kx_full, ky_full = recon.get_rotated_trajectory(seq, k_traj_adc, n_adc, n_trs, 0)
 
    # Raw data: extract_kspace returns [samples, readouts, coils] for both vendors
    _, ksp_raw, _, vendor = extract_raw.extract_kspace(data_fn)
    ksp_raw = np.asarray(ksp_raw)
    n_samp, n_ro, n_coils = ksp_raw.shape
    if n_samp != n_adc:
        print(f'WARNING: raw has {n_samp} samples/readout, sequence says {n_adc}.')
    expected = n_offsets * n_trs
    ksp_raw = ksp_raw[:, N_SKIP_READOUTS:, :]
    if ksp_raw.shape[1] != expected:
        raise ValueError(
            f'Raw has {ksp_raw.shape[1]} readouts after skipping {N_SKIP_READOUTS}, '
            f'expected {n_offsets} offsets x {n_trs} TRs = {expected}. '
            f'Check N_SKIP_READOUTS / that data_fn matches seq_fn.')
    # -> [offsets, samples, TRs, coils] (offsets acquired back-to-back, n_trs readouts each)
    ksp_all = ksp_raw.reshape(n_samp, n_offsets, n_trs, n_coils).transpose(1, 0, 2, 3)
 
    image_stack = []
    for i in range(n_offsets):
        print(f'  Recon offset {i + 1}/{n_offsets} ({offsets_acq[i]:+.2f} ppm)')
        image_stack.append(recon.adjoint_nufft_from_traj(ksp_all[i], kx_full, ky_full, nx, fov))
    image_stack = np.array(image_stack)
    np.savez(cache_fn, image_stack=image_stack, offsets_acq=offsets_acq, data_fn=data_fn)
    print(f'Saved recon cache to {cache_fn}')
 
mag_stack = np.abs(image_stack)
 
# %% Step 2: Pick vials on the M0 image
m0_idx = np.flatnonzero(np.abs(offsets_acq) >= M0_THRESH_PPM)
if len(m0_idx) == 0:
    raise ValueError(f'No M0 offset (|offset| >= {M0_THRESH_PPM} ppm) in the sequence.')
img_m0 = mag_stack[m0_idx].mean(axis=0)
max_r_px = MAX_VIAL_RADIUS_MM * 1e-3 / (fov / nx)
 
res = vial_roi.pick_vials(img_m0, max_r_px, margin=MARGIN_PX,
                          title_prefix=f'M0 ({offsets_acq[m0_idx[0]]:.0f} ppm)')
vial_roi.save_roi_overlay(img_m0, res, os.path.join(out_dir, 'vial_rois.png'))
 
# %% Step 3: Z-spectra per vial
# Repeated saturation offsets (e.g. the +2/-2 ppm scans prepended to the sweep):
# keep the LAST acquisition of each offset (the one inside the sweep) for the
# spectra; earlier repeats are "extras", used only as a drift check.
non_m0 = np.setdiff1d(np.arange(n_offsets), m0_idx)
extra_idx, extra_match = [], []
for i in non_m0:
    later = [j for j in non_m0 if j > i and np.isclose(offsets_acq[j], offsets_acq[i])]
    if later:
        extra_idx.append(i)
        extra_match.append(later[-1])
extra_idx = np.array(extra_idx, dtype=int)
if len(extra_idx):
    print(f'Repeated offsets: scans {extra_idx.tolist()} '
          f'({", ".join(f"{o:+.2f}" for o in offsets_acq[extra_idx])} ppm) left out of the spectra')
 
sat_idx = np.setdiff1d(non_m0, extra_idx)
order = sat_idx[np.argsort(offsets_acq[sat_idx])]
offs = offsets_acq[order]
 
m0_mean = np.array([img_m0[m].mean() for m in res['masks']])
 
# Normalisation reference
far_idx = sat_idx[np.abs(offsets_acq[sat_idx]) >= FAR_PPM]
if len(far_idx) == 0:
    raise ValueError(f'No saturated offsets with |offset| >= {FAR_PPM} ppm for the far reference.')
img_far = mag_stack[far_idx].mean(axis=0)
far_mean = np.array([img_far[m].mean() for m in res['masks']])
print(f'\nFar reference: {", ".join(f"{o:+.1f}" for o in offsets_acq[far_idx])} ppm')
for i in range(len(res['masks'])):
    print(f'Vial {i + 1}: M0 / far = {m0_mean[i] / far_mean[i]:.3f}')
 
if NORM_REF == 'm0':
    ref_img, ref_label = img_m0, r'$M_{sat}/M_0$'
elif NORM_REF == 'far':
    ref_img, ref_label = img_far, rf'$M_{{sat}}/M_{{\pm{FAR_PPM:g}\,ppm}}$'
else:
    raise ValueError("NORM_REF must be 'm0' or 'far'")
 
z_mean, z_std = vial_roi.roi_stats(mag_stack[order], res['masks'], ref=ref_img)
sig_mean, _ = vial_roi.roi_stats(mag_stack[order], res['masks'])
 
# Drift check: Z of each early repeat vs the same offset in the sweep
for e, j in zip(extra_idx, extra_match):
    for i, m in enumerate(res['masks']):
        z_early = mag_stack[e][m].mean() / m0_mean[i]
        z_late = mag_stack[j][m].mean() / m0_mean[i]
        print(f'Vial {i + 1}, {offsets_acq[e]:+.2f} ppm: Z(scan {e}) = {z_early:.3f}, '
              f'Z(scan {j}) = {z_late:.3f}, diff = {z_late - z_early:+.3f}')
 
vial_roi.plot_vial_grid(offs, z_mean, z_std, res['colors'],
                        xlabel='Saturation offset [ppm]', ylabel=ref_label,
                        title=f'Z-spectra per vial ({run_label}, norm: {NORM_REF}, mean ± pixelwise SD)',
                        out_fn=os.path.join(out_dir, f'zspectra_grid_{NORM_REF}.png'), invert_x=True,
                        ylim=(0, max(1.05, np.nanmax(z_mean + z_std) * 1.02)))
 
# Export
n_vials = len(res['vials'])
header = ['offset_ppm'] + [f'vial{i + 1}_{s}' for i in range(n_vials) for s in ('Z', 'SD')]
cols = [offs] + [a for i in range(n_vials) for a in (z_mean[i], z_std[i])]
np.savetxt(os.path.join(out_dir, f'zspectra_{NORM_REF}.csv'), np.column_stack(cols),
           delimiter=',', header=','.join(header), comments='', fmt='%.6f')
# Single .npz with every vial's spectrum (rows = vials, columns = offsets_ppm)
spectra_fn = os.path.join(out_dir, f'{run_label}_zspectra_{NORM_REF}.npz')
np.savez(spectra_fn,
         run_label=run_label,
         norm_ref=NORM_REF,                       # 'm0' or 'far'
         offsets_ppm=offs,                        # [n_offsets], sorted, M0 + repeats excluded
         z_mean=z_mean,                           # [n_vials, n_offsets]  M_sat / reference
         z_std=z_std,                             # [n_vials, n_offsets]  pixelwise SD of M_sat / reference
         signal_mean=sig_mean,                    # [n_vials, n_offsets]  un-normalised ROI mean
         m0_mean=m0_mean,                         # [n_vials]
         far_mean=far_mean,                       # [n_vials]  mean of the far offsets
         far_offsets_ppm=offsets_acq[far_idx],
         m0_offsets_ppm=offsets_acq[m0_idx],
         extra_idx=extra_idx,                     # acquisition indices of repeats left out
         extra_offsets_ppm=offsets_acq[extra_idx],
         img_m0=img_m0,
         data_fn=data_fn, seq_fn=seq_fn,
         **vial_roi.roi_geometry(res))
print(f'Saved spectra for {n_vials} vial(s) to {spectra_fn}')
print(f'Saved ROI overlay, Z-spectra grid and CSV to {out_dir}/')
plt.show()