#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-spectrum phantom recon + per-vial spectra for GE or Siemens.

  GE       one scan (folder with one .h5) per offset, all sharing one trajectory;
           offsets in acquisition order are listed in CONFIG (from the pgeNN.entry files).
  Siemens  one combined ZSPEC .dat with every offset back-to-back; offsets read from
           the .seq definition 'Offsets_ppm'.

Both: one image per offset (cached, recon is the slow part) -> pick vials on the
M0 image (utils.vial_roi) -> Z-spectra per vial -> grid plot, CSV and
<run_label>_zspectra.npz (including the per-offset images as 'images_sat' for
pixelwise B0 maps) -> optional GIF.

Offsets with |offset| >= M0_THRESH_PPM are the M0 reference. Repeated saturation
offsets (e.g. the +2/-2 ppm scans before the sweep) are detected automatically:
the last acquisition of each offset is kept for the spectra, earlier ones are used
as a drift check.

Run with an interactive backend (e.g. `%matplotlib qt` in Spyder); picker
controls are listed in utils/vial_roi.py.
"""

# %% Step 0: Parameters
import os
import re
import numpy as np
import matplotlib.pyplot as plt

SCANNER = 'ge'  # 'ge' or 'siemens'

_ge_sweep = np.concatenate((np.arange(-10, -5, 1), np.arange(-5, 5.01, 0.2), np.arange(6, 10.01, 1)))
CONFIG = {
    'ge': dict(
        data='data/raw/ge/Exam807',                                          # one subfolder (one .h5) per offset
        seq='sequences/ge/continuous_spiral_zspec_gauss_75p0_ppm_ge.seq',     # any offset's .seq: same trajectory
        # Acquisition order (pge10..pge73.entry): 75 ppm M0, extra +2/-2 ppm, then the -10..+10 ppm sweep
        offsets_ppm=np.round(np.concatenate(([75.0, 2.0, -2.0], _ge_sweep)), 2),
        out_dir='results/zspec_ge',
        run_label='ge',
    ),
    'siemens': dict(
        data='/Volumes/JWW-BIC/Phantom_Zspec/Siemens/meas_MID00187_FID56519_zspec_gauss.dat',
        seq='/Volumes/JWW-BIC/Phantom_Zspec/Siemens/continuous_spiral_zspec_gauss.seq',
        offsets_ppm=None,                     # None -> seq definition 'Offsets_ppm'
        out_dir='results/zspec_siemens',
        run_label='compare_ge',
        n_skip_readouts=0,                    # leading readouts in the .dat to discard, if any
    ),
}

USE_CACHE = True          # reuse the reconstructed images if the cache matches data, seq and offsets
M0_THRESH_PPM = 50.0      # |offset| >= this is the M0 reference
NORM_REF = 'far'          # 'm0': divide by the M0 image | 'far': divide by the mean of the far offsets
FAR_PPM = 9.5             # 'far' reference = saturated offsets with |offset| >= this (+/-10 ppm)
MAX_VIAL_RADIUS_MM = 20   # longest ray for wall detection [mm]
MARGIN_PX = 1.5           # initial ROI margin (adjust live with +/-)
MAKE_GIF = True

assert SCANNER in CONFIG, f"SCANNER must be one of {list(CONFIG)}"
cfg = CONFIG[SCANNER]
out_dir, run_label = cfg['out_dir'], cfg['run_label']
cache_fn = os.path.join(out_dir, 'zspec_recon_cache.npz')
os.makedirs(out_dir, exist_ok=True)


# %% Step 1: Sequence info
from utils import extract_raw, recon, vial_roi

seq = extract_raw.load_seq(cfg['seq'])
defs = seq.definitions
nx = int(defs['Nx'])
fov = float(np.atleast_1d(defs['FOV'])[0])
n_adc = int(defs['MaxAdcSegmentLength'])
n_trs = int(defs['N_TRs_per_offset'] if 'N_TRs_per_offset' in defs else defs['N_TRs'])
pislquant = int(defs.get('pislquant', 0))          # GE receive-gain calibration TRs (0 on Siemens)
if cfg['offsets_ppm'] is not None:
    offsets_acq = np.asarray(cfg['offsets_ppm'], dtype=float)
else:
    offsets_acq = np.atleast_1d(np.asarray(defs['Offsets_ppm'], dtype=float))
n_offsets = len(offsets_acq)
print(f'{SCANNER}: {n_offsets} offsets x {n_trs} TRs, {n_adc} ADC samples, Nx={nx}, FOV={fov*1e3:.0f} mm')


# %% Step 2: Raw data -> images (cached)
def _natural_key(s):
    # numeric parts compare as numbers, so '2' < '10'
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def load_kspace_ge(data_dir):
    """[offsets, samples, TRs, coils] from one .h5 per subfolder, calibration TRs dropped."""
    folders = sorted((f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))),
                     key=_natural_key)
    print('Folder order:', folders)
    if len(folders) != n_offsets:
        raise ValueError(f'{len(folders)} scan folders but {n_offsets} offsets in CONFIG.')
    main_start = 1 + pislquant          # one extra view + calibration TRs before the real train
    ksp = []
    for folder in folders:
        path = os.path.join(data_dir, folder)
        h5 = [f for f in os.listdir(path) if f.endswith('.h5')]
        if len(h5) != 1:
            raise ValueError(f'Expected exactly one .h5 file in {path}, found {len(h5)}')
        _, k, _, _ = extract_raw.extract_kspace(os.path.join(path, h5[0]))
        ksp.append(np.asarray(k)[:, main_start:main_start + n_trs, :])
    return np.array(ksp)


def load_kspace_siemens(data_fn, n_skip):
    """[offsets, samples, TRs, coils] from one combined .dat (offsets back-to-back)."""
    _, k, _, _ = extract_raw.extract_kspace(data_fn)
    k = np.asarray(k)[:, n_skip:, :]                  # [samples, readouts, coils]
    n_samp, n_ro, n_coils = k.shape
    if n_samp != n_adc:
        print(f'WARNING: raw has {n_samp} samples/readout, sequence says {n_adc}.')
    if n_ro != n_offsets * n_trs:
        raise ValueError(f'Raw has {n_ro} readouts after skipping {n_skip}, expected '
                         f'{n_offsets} offsets x {n_trs} TRs = {n_offsets * n_trs}. '
                         f'Check n_skip_readouts / that the .dat matches the .seq.')
    return k.reshape(n_samp, n_offsets, n_trs, n_coils).transpose(1, 0, 2, 3)


def trajectory():
    """Rotated trajectory [n_trs, n_adc] shared by every offset."""
    print('Computing k-space trajectory (can take a while for the combined Siemens sequence)...')
    k_traj_adc, _, _, _, _ = seq.calculate_kspace(trajectory_delay=0)
    if SCANNER == 'siemens':   # golden-angle train restarts at every offset: first offset defines it
        k_traj_adc = k_traj_adc[:, :n_adc * n_trs]
    return recon.get_rotated_trajectory(seq, k_traj_adc, n_adc, n_trs, pislquant)


cache_ok = False
if USE_CACHE and os.path.exists(cache_fn):
    c = np.load(cache_fn, allow_pickle=True)
    if (c['image_stack'].shape[0] == n_offsets and np.allclose(c['offsets_acq'], offsets_acq)
            and str(c['data_fn']) == str(cfg['data'])
            and ('seq_fn' not in c.files or str(c['seq_fn']) == str(cfg['seq']))):
        image_stack = c['image_stack']
        cache_ok = True
        print(f'Loaded cached recon from {cache_fn}')
    else:
        print('Cache does not match this data/sequence/offsets; reconstructing.')

if not cache_ok:
    kx_full, ky_full = trajectory()
    ksp_all = (load_kspace_ge(cfg['data']) if SCANNER == 'ge'
               else load_kspace_siemens(cfg['data'], cfg.get('n_skip_readouts', 0)))
    image_stack = []
    for i in range(n_offsets):
        print(f'  Recon offset {i + 1}/{n_offsets} ({offsets_acq[i]:+.2f} ppm)')
        image_stack.append(recon.adjoint_nufft_from_traj(ksp_all[i], kx_full, ky_full, nx, fov))
    image_stack = np.array(image_stack)
    np.savez(cache_fn, image_stack=image_stack, offsets_acq=offsets_acq,
             data_fn=str(cfg['data']), seq_fn=str(cfg['seq']))
    print(f'Saved recon cache to {cache_fn}')

mag_stack = np.abs(image_stack)


# %% Step 3: Pick vials on the M0 image
m0_idx = np.flatnonzero(np.abs(offsets_acq) >= M0_THRESH_PPM)
if len(m0_idx) == 0:
    raise ValueError(f'No M0 offset (|offset| >= {M0_THRESH_PPM} ppm).')
img_m0 = mag_stack[m0_idx].mean(axis=0)
max_r_px = MAX_VIAL_RADIUS_MM * 1e-3 / (fov / nx)

res = vial_roi.pick_vials(img_m0, max_r_px, margin=MARGIN_PX,
                          title_prefix=f'{SCANNER} M0 ({offsets_acq[m0_idx[0]]:.0f} ppm)')
vial_roi.save_roi_overlay(img_m0, res, os.path.join(out_dir, 'vial_rois.png'))


# %% Step 4: Z-spectra per vial
# Repeated saturation offsets: keep the LAST acquisition (inside the sweep); earlier ones are drift checks
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
masks = res['masks']
m0_mean = np.array([img_m0[m].mean() for m in masks])

far_idx = sat_idx[np.abs(offsets_acq[sat_idx]) >= FAR_PPM]
if len(far_idx) == 0:
    raise ValueError(f'No saturated offsets with |offset| >= {FAR_PPM} ppm for the far reference.')
img_far = mag_stack[far_idx].mean(axis=0)
far_mean = np.array([img_far[m].mean() for m in masks])
print(f'\nFar reference: {", ".join(f"{o:+.1f}" for o in offsets_acq[far_idx])} ppm')
for i in range(len(masks)):
    print(f'Vial {i + 1}: M0 / far = {m0_mean[i] / far_mean[i]:.3f}')

if NORM_REF == 'm0':
    ref_img, ref_label = img_m0, r'$M_{sat}/M_0$'
elif NORM_REF == 'far':
    ref_img, ref_label = img_far, rf'$M_{{sat}}/M_{{\pm{FAR_PPM:g}\,ppm}}$'
else:
    raise ValueError("NORM_REF must be 'm0' or 'far'")

z_mean, z_std = vial_roi.roi_stats(mag_stack[order], masks, ref=ref_img)
sig_mean, _ = vial_roi.roi_stats(mag_stack[order], masks)

# Drift check: each early repeat vs the same offset in the sweep (same reference image)
for e, j in zip(extra_idx, extra_match):
    for i, m in enumerate(masks):
        ref = ref_img[m].mean()
        z_early, z_late = mag_stack[e][m].mean() / ref, mag_stack[j][m].mean() / ref
        print(f'Vial {i + 1}, {offsets_acq[e]:+.2f} ppm: Z(scan {e}) = {z_early:.3f}, '
              f'Z(scan {j}) = {z_late:.3f}, diff = {z_late - z_early:+.3f}')

vial_roi.plot_vial_grid(offs, z_mean, z_std, res['colors'],
                        xlabel='Saturation offset [ppm]', ylabel=ref_label,
                        title=f'Z-spectra per vial ({SCANNER}, {run_label}, norm: {NORM_REF}, mean ± pixelwise SD)',
                        out_fn=os.path.join(out_dir, f'zspectra_grid_{NORM_REF}.png'), invert_x=True,
                        ylim=(0, max(1.05, np.nanmax(z_mean + z_std) * 1.02)))

n_vials = len(res['vials'])
header = ['offset_ppm'] + [f'vial{i + 1}_{s}' for i in range(n_vials) for s in ('Z', 'SD')]
cols = [offs] + [a for i in range(n_vials) for a in (z_mean[i], z_std[i])]
np.savetxt(os.path.join(out_dir, f'zspectra_{NORM_REF}.csv'), np.column_stack(cols),
           delimiter=',', header=','.join(header), comments='', fmt='%.6f')

spectra_fn = os.path.join(out_dir, f'{run_label}_zspectra.npz')
np.savez(spectra_fn,
         scanner=SCANNER,
         run_label=run_label,
         norm_ref=NORM_REF,                       # 'm0' or 'far'
         offsets_ppm=offs,                        # [n_offsets], sorted, M0 + repeats excluded
         z_mean=z_mean,                           # [n_vials, n_offsets]  M_sat / reference
         z_std=z_std,                             # [n_vials, n_offsets]  pixelwise SD of M_sat / reference
         signal_mean=sig_mean,                    # [n_vials, n_offsets]  un-normalised ROI mean
         m0_mean=m0_mean,                         # [n_vials]
         far_mean=far_mean,                       # [n_vials]
         far_offsets_ppm=offsets_acq[far_idx],
         m0_offsets_ppm=offsets_acq[m0_idx],
         extra_idx=extra_idx,                     # acquisition indices of repeats left out
         extra_offsets_ppm=offsets_acq[extra_idx],
         img_m0=img_m0,
         img_far=img_far,
         images_sat=mag_stack[order].astype(np.float32),   # [n_offsets, ny, nx], same order as offsets_ppm
         data_fn=str(cfg['data']), seq_fn=str(cfg['seq']),
         **vial_roi.roi_geometry(res))
print(f'Saved spectra for {n_vials} vial(s) to {spectra_fn}')
plt.show()


# %% Step 5: GIF of the offset images (acquisition order)
if MAKE_GIF:
    from matplotlib.animation import FuncAnimation, PillowWriter

    gif_fn = os.path.join(out_dir, f'offset_images_{run_label}.gif')
    vmax = np.percentile(mag_stack, 99.5)       # fixed window so frames are comparable
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(mag_stack[0], cmap='gray', vmin=0, vmax=vmax)
    ax.axis('off')
    label = ax.text(0.03, 0.97, '', transform=ax.transAxes, color='yellow', fontsize=14, fontweight='bold',
                    va='top', ha='left', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3))

    def _label(i):
        tag = ' (M0)' if i in m0_idx else ' (extra)' if i in extra_idx else ''
        return f'{offsets_acq[i]:.2f} ppm{tag}'

    def _update(i):
        im.set_data(mag_stack[i])
        label.set_text(_label(i))
        return im, label

    anim = FuncAnimation(fig, _update, frames=n_offsets, interval=250, blit=True)
    anim.save(gif_fn, writer=PillowWriter(fps=4), dpi=100)
    plt.close(fig)
    print(f'Saved {gif_fn} ({n_offsets} frames)')