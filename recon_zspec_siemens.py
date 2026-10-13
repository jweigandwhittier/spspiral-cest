#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siemens Z-spectrum phantom analysis.

Takes the single combined Siemens ZSPEC acquisition (FLAG_ZSPEC=True, one
.dat with every offset back-to-back, recovery delays in between), reconstructs
one image per offset, lets you click inside each phantom vial, grows each click
out to the vial wall, fits a circle, shrinks it by a margin, and plots the
Z-spectrum per vial.

Vial picker controls (run with an interactive backend, e.g. `%matplotlib qt`
in Spyder):
    left click        add a vial (click anywhere inside it; the wall is found automatically)
    left drag         draw a vial by hand: press at its centre, release on its wall
    right click       remove the vial under the cursor (or the last one)
    u / backspace     undo last vial
    + / -             grow / shrink the margin for all vials
    enter             finish (closing the window also finishes)
Zoom/pan with the toolbar is fine: clicks are ignored while a tool is active.
"""

# %% Step 0: Parameters
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import find_peaks

data_fn = '/Volumes/JWW-BIC/lipids/meas_MID00047_FID54534_spspiral_spsp_exc.dat'  # combined ZSPEC .dat
seq_fn = '/Volumes/JWW-BIC/lipids/continuous_spiral_zspec_gauss_spspexc.seq'  # matching .seq

out_dir = 'results/zspec_siemens'
run_label = 'ff_spsp_on'  # Used in the saved spectra filename and as the plot label when comparing
cache_fn = os.path.join(out_dir, 'zspec_recon_cache.npz')
USE_CACHE = False        # Reuse images + trajectory from a previous run (recon is the slow part)
N_SKIP_READOUTS = 0     # Leading readouts in the .dat to discard (dummy/noise scans), if any

# Vial segmentation
M0_THRESH_PPM = 50.0    # Offsets with |offset| >= this are treated as M0 / S0 references
SMOOTH_SIGMA = 0.5      # Gaussian smoothing [px] for the segmentation image (more blurs the thin walls away)
DIP_FRAC = 0.10         # Vial wall = first dip along a ray at least this deep, as a fraction of the click value
N_RAYS = 90             # Rays cast from the click point
MAX_VIAL_RADIUS_MM = 20 # Longest ray [mm]; keeps rays from running into neighbouring vials
MIN_VIAL_RADIUS_PX = 1.5 # Ignore dips closer to the click than this [px]
MARGIN_PX = 1.5         # Shrink fitted circle (wall centre) by this many pixels (adjust live with +/-)
MIN_ROI_RADIUS_PX = 1.0



# ========================================== #
# Helpers
# ========================================== #
def _fit_circle(rows, cols):
    """Algebraic (Kasa) least-squares circle fit. Returns (row_c, col_c, radius)."""
    A = np.column_stack([2 * cols, 2 * rows, np.ones_like(rows)])
    b = cols ** 2 + rows ** 2
    (xc, yc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return yc, xc, np.sqrt(c + xc ** 2 + yc ** 2)


def find_vial(img_seg, seed_rc, max_r, n_rays=N_RAYS, dip_frac=DIP_FRAC, min_r=MIN_VIAL_RADIUS_PX):
    """
    From the click, find the vial wall along radial rays as the first local
    minimum that dips at least dip_frac * (click value) below its surroundings,
    then fit a circle through the wall points with outlier rejection.
    Returns dict or None.
    """
    r0, c0 = seed_rc
    ri, ci = int(round(r0)), int(round(c0))
    seed_val = np.median(img_seg[max(ri - 1, 0):ri + 2, max(ci - 1, 0):ci + 2])
    if seed_val <= 0:
        return None

    step = 0.25
    thetas = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(0, max_r, step)
    rows = r0 + radii[None, :] * np.sin(thetas)[:, None]
    cols = c0 + radii[None, :] * np.cos(thetas)[:, None]
    prof = map_coordinates(img_seg, [rows, cols], order=1, mode='nearest')

    edge_r = np.full(n_rays, np.nan)
    for i in range(n_rays):
        dips, _ = find_peaks(-prof[i], prominence=dip_frac * seed_val)
        dips = dips[radii[dips] >= min_r]
        if len(dips):
            j = dips[0]
            # sub-sample position of the wall minimum (parabola through 3 points)
            if 0 < j < len(radii) - 1:
                y0, y1, y2 = prof[i, j - 1], prof[i, j], prof[i, j + 1]
                den = y0 - 2 * y1 + y2
                edge_r[i] = radii[j] + (0.5 * (y0 - y2) / den * step if den != 0 else 0.0)
            else:
                edge_r[i] = radii[j]

    # Reject rays that went through a gap in the wall or stopped at a bubble/inner structure
    good = np.isfinite(edge_r)
    if good.sum() < 8:
        return None
    med = np.median(edge_r[good])
    mad = 1.4826 * np.median(np.abs(edge_r[good] - med))
    good &= np.abs(edge_r - med) <= 3 * mad + 0.75
    if good.sum() < 8:
        return None

    er = r0 + edge_r[good] * np.sin(thetas[good])
    ec = c0 + edge_r[good] * np.cos(thetas[good])
    rc, cc, R = _fit_circle(er, ec)
    resid = np.abs(np.hypot(er - rc, ec - cc) - R)
    keep = resid <= max(1.0, 2.5 * np.median(resid))
    if keep.sum() >= 8:
        er, ec = er[keep], ec[keep]
        rc, cc, R = _fit_circle(er, ec)

    return dict(center=(rc, cc), radius=R, edge_rows=er, edge_cols=ec,
                n_rays_used=int(len(er)), seed=(r0, c0))


def disk_mask(shape, center, radius):
    rr, cc = np.mgrid[:shape[0], :shape[1]]
    return (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2


class VialPicker:
    def __init__(self, img_disp, img_seg, max_r, margin=MARGIN_PX, title_prefix='S0'):
        self.img_seg = img_seg
        self.max_r = max_r
        self.press = None
        self.margin = margin
        self.title_prefix = title_prefix
        self.vials = []
        self.artists = []
        self.done = False
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        vmax = np.percentile(img_disp, 99.5)
        self.ax.imshow(img_disp, cmap='gray', vmin=0, vmax=vmax)
        self.ax.set_axis_off()
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', lambda ev: setattr(self, 'done', True))
        self._redraw()

    def color(self, i):
        return self.colors[i % len(self.colors)]

    def roi_radius(self, v):
        return max(v['radius'] - self.margin, MIN_ROI_RADIUS_PX)

    def _usable(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return False
        tb = getattr(self.fig.canvas, 'toolbar', None)
        return not (tb is not None and getattr(tb, 'mode', ''))  # ignore while zoom/pan active

    def _on_press(self, ev):
        if self._usable(ev) and ev.button == 1:
            self.press = (ev.ydata, ev.xdata)

    def _on_release(self, ev):
        if ev.button == 1:
            start, self.press = self.press, None
            if start is None or not self._usable(ev):
                return
            r, c = start
            drag = np.hypot(ev.ydata - r, ev.xdata - c)
            if drag > 2:   # dragged: manual circle, centre at press, edge at release
                v = dict(center=(r, c), radius=drag, edge_rows=np.array([]), edge_cols=np.array([]),
                         n_rays_used=0, seed=(r, c))
                print(f'Vial {len(self.vials) + 1}: manual circle, centre (r={r:.1f}, c={c:.1f}), R={drag:.2f} px')
            else:          # click: automatic wall detection
                v = find_vial(self.img_seg, (r, c), self.max_r)
                if v is None:
                    print(f'No vial wall found from ({r:.1f}, {c:.1f}); drag from the centre to the wall '
                          f'to draw it, or adjust DIP_FRAC / SMOOTH_SIGMA.')
                    return
                print(f'Vial {len(self.vials) + 1}: centre (r={v["center"][0]:.1f}, c={v["center"][1]:.1f}), '
                      f'R={v["radius"]:.2f} px, {v["n_rays_used"]}/{N_RAYS} rays')
            self.vials.append(v)
            self._redraw()
        elif ev.button == 3 and self._usable(ev):
            self._on_right_click(ev)

    def _on_right_click(self, ev):
        r, c = ev.ydata, ev.xdata
        if self.vials:
            inside = [i for i, v in enumerate(self.vials)
                      if np.hypot(r - v['center'][0], c - v['center'][1]) <= v['radius']]
            self.vials.pop(inside[-1] if inside else -1)
        self._redraw()

    def _on_key(self, ev):
        if ev.key in ('u', 'backspace') and self.vials:
            self.vials.pop()
        elif ev.key in ('+', '='):
            self.margin += 0.5
        elif ev.key == '-':
            self.margin = max(self.margin - 0.5, 0.0)
        elif ev.key == 'enter':
            self.done = True
            return
        self._redraw()

    def _redraw(self):
        for a in self.artists:
            a.remove()
        self.artists = []
        for i, v in enumerate(self.vials):
            col = self.color(i)
            (rc, cc), R = v['center'], v['radius']
            outer = Circle((cc, rc), R, fill=False, ec=col, ls='--', lw=1)
            inner = Circle((cc, rc), self.roi_radius(v), fill=False, ec=col, lw=2)
            self.ax.add_patch(outer)
            self.ax.add_patch(inner)
            pts, = self.ax.plot(v['edge_cols'], v['edge_rows'], '.', ms=2, color=col)
            txt = self.ax.text(cc, rc, str(i + 1), color=col, ha='center', va='center',
                               fontsize=11, fontweight='bold')
            self.artists += [outer, inner, pts, txt]
        self.ax.set_title(f'{self.title_prefix} | {len(self.vials)} vial(s) | margin {self.margin:.1f} px\n'
                          'Click = auto, drag centre→wall = manual, R-click/u remove, +/- margin, Enter done',
                          fontsize=10)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show(block=False)
        while not self.done and plt.fignum_exists(self.fig.number):
            plt.pause(0.1)
        return self.vials, self.margin


# %% Step 1: Sequence info + trajectory, raw data, recon (cached)
from utils import extract_raw, recon

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
img_seg = gaussian_filter(img_m0, SMOOTH_SIGMA)

max_r_px = MAX_VIAL_RADIUS_MM * 1e-3 / (fov / nx)
picker = VialPicker(img_m0, img_seg, max_r_px, margin=MARGIN_PX,
                    title_prefix=f'M0 ({offsets_acq[m0_idx[0]]:.0f} ppm)')
vials, margin_final = picker.run()
if not vials:
    raise SystemExit('No vials selected.')

masks = np.array([disk_mask(img_m0.shape, v['center'], picker.roi_radius(v)) for v in vials])
for i, m in enumerate(masks):
    print(f'Vial {i + 1}: ROI radius {picker.roi_radius(vials[i]):.2f} px, {m.sum()} px')

# Save ROI overlay
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_m0, cmap='gray', vmin=0, vmax=np.percentile(img_m0, 99.5))
for i, v in enumerate(vials):
    (rc, cc) = v['center']
    ax.add_patch(Circle((cc, rc), picker.roi_radius(v), fill=False, ec=picker.color(i), lw=2))
    ax.text(cc, rc, str(i + 1), color=picker.color(i), ha='center', va='center', fontweight='bold')
ax.set_axis_off()
ax.set_title(f'Vial ROIs (margin {margin_final:.1f} px)')
fig.savefig(os.path.join(out_dir, 'vial_rois.png'), dpi=150, bbox_inches='tight')

# %% Step 3: Z-spectra per vial
sat_idx = np.setdiff1d(np.arange(n_offsets), m0_idx)
order = sat_idx[np.argsort(offsets_acq[sat_idx])]
offs = offsets_acq[order]

z_mean = np.zeros((len(vials), len(order)))
z_std = np.zeros_like(z_mean)
sig_mean = np.zeros_like(z_mean)
m0_mean = np.zeros(len(vials))
for i, m in enumerate(masks):
    m0_pix = img_m0[m]
    m0_mean[i] = m0_pix.mean()
    sig_mean[i] = mag_stack[order][:, m].mean(axis=1)
    z_mean[i] = sig_mean[i] / m0_mean[i]                      # ratio of means
    z_std[i] = (mag_stack[order][:, m] / m0_pix).std(axis=1)  # pixelwise spread

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(vials)):
    col = picker.color(i)
    ax.plot(offs, z_mean[i], 'o-', ms=3, lw=1.2, color=col, label=f'Vial {i + 1}')
    ax.fill_between(offs, z_mean[i] - z_std[i], z_mean[i] + z_std[i], color=col, alpha=0.15)
ax.invert_xaxis()
ax.set_xlabel('Saturation offset [ppm]')
ax.set_ylabel(r'$M_{sat}/M_0$')
ax.set_ylim(0, max(1.05, np.nanmax(z_mean + z_std) * 1.02))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
ax.set_title('Z-spectra per vial (mean ± pixelwise SD)')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'zspectra.png'), dpi=150)

# Export
header = ['offset_ppm'] + [f'vial{i + 1}_{s}' for i in range(len(vials)) for s in ('Z', 'SD')]
cols = [offs] + [a for i in range(len(vials)) for a in (z_mean[i], z_std[i])]
np.savetxt(os.path.join(out_dir, 'zspectra.csv'), np.column_stack(cols),
           delimiter=',', header=','.join(header), comments='', fmt='%.6f')
# Single .npz with every vial's spectrum (rows = vials, columns = offsets_ppm)
spectra_fn = os.path.join(out_dir, f'{run_label}_zspectra.npz')
np.savez(spectra_fn,
         run_label=run_label,
         offsets_ppm=offs,                        # [n_offsets], sorted, M0 excluded
         z_mean=z_mean,                           # [n_vials, n_offsets]  M_sat / M0
         z_std=z_std,                             # [n_vials, n_offsets]  pixelwise SD of M_sat / M0
         signal_mean=sig_mean,                    # [n_vials, n_offsets]  un-normalised ROI mean
         m0_mean=m0_mean,                         # [n_vials]
         vial_labels=np.array([f'Vial {i + 1}' for i in range(len(vials))]),
         n_pixels=masks.sum(axis=(1, 2)),         # [n_vials]
         masks=masks,                             # [n_vials, ny, nx]
         centers_rc=np.array([v['center'] for v in vials]),
         fitted_radii_px=np.array([v['radius'] for v in vials]),
         roi_radii_px=np.array([picker.roi_radius(v) for v in vials]),
         margin_px=margin_final,
         m0_offsets_ppm=offsets_acq[m0_idx],
         img_m0=img_m0,
         data_fn=data_fn, seq_fn=seq_fn)
print(f'Saved spectra for {len(vials)} vial(s) to {spectra_fn}')
print(f'Saved ROI overlay, Z-spectra plot and CSV to {out_dir}/')
plt.show()