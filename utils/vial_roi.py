#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive phantom vial segmentation + per-vial ROI statistics and plotting.

Click inside each vial; rays are cast from the click, the vial wall is found as
the first sufficiently deep dip along each ray, a circle is fitted through the
wall points (with outlier rejection), and the circle is shrunk by a margin to
give the ROI.

Picker controls (needs an interactive backend, e.g. `%matplotlib qt` in Spyder):
    left click        add a vial (click anywhere inside it; the wall is found automatically)
    left drag         draw a vial by hand: press at its centre, release on its wall
    right click       remove the vial under the cursor (or the last one)
    u / backspace     undo last vial
    + / -             grow / shrink the margin for all vials
    enter             finish (closing the window also finishes)
Zoom/pan with the toolbar is fine: clicks are ignored while a tool is active.

Typical use:
    from utils import vial_roi
    res = vial_roi.pick_vials(ref_image, max_radius_px=..., title_prefix='S0')
    mean, sd = vial_roi.roi_stats(stack, res['masks'], ref=ref_image)
    vial_roi.save_roi_overlay(ref_image, res, 'results/vial_rois.png')
    vial_roi.plot_vial_grid(x, mean, sd, res['colors'], ..., out_fn='results/grid.png')
    np.savez(fn, ..., **vial_roi.roi_geometry(res))
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import find_peaks

# Defaults
SMOOTH_SIGMA = 0.5        # Gaussian smoothing [px] for the segmentation image (more blurs thin walls away)
DIP_FRAC = 0.10           # Vial wall = first dip along a ray at least this deep, as a fraction of the click value
N_RAYS = 90               # Rays cast from the click point
MIN_VIAL_RADIUS_PX = 1.5  # Ignore dips closer to the click than this [px]
MARGIN_PX = 1.5           # Shrink fitted circle (wall centre) by this many pixels (adjust live with +/-)
MIN_ROI_RADIUS_PX = 1.0


# ========================================== #
# Geometry
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


# ========================================== #
# Interactive picker
# ========================================== #
class VialPicker:
    def __init__(self, img_disp, img_seg, max_r, margin=MARGIN_PX, title_prefix='',
                 n_rays=N_RAYS, dip_frac=DIP_FRAC, min_r=MIN_VIAL_RADIUS_PX,
                 min_roi_r=MIN_ROI_RADIUS_PX):
        self.img_seg = img_seg
        self.max_r = max_r
        self.n_rays = n_rays
        self.dip_frac = dip_frac
        self.min_r = min_r
        self.min_roi_r = min_roi_r
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
        return max(v['radius'] - self.margin, self.min_roi_r)

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
                v = find_vial(self.img_seg, (r, c), self.max_r, n_rays=self.n_rays,
                              dip_frac=self.dip_frac, min_r=self.min_r)
                if v is None:
                    print(f'No vial wall found from ({r:.1f}, {c:.1f}); drag from the centre to the wall '
                          f'to draw it, or adjust dip_frac / smooth_sigma.')
                    return
                print(f'Vial {len(self.vials) + 1}: centre (r={v["center"][0]:.1f}, c={v["center"][1]:.1f}), '
                      f'R={v["radius"]:.2f} px, {v["n_rays_used"]}/{self.n_rays} rays')
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
        prefix = f'{self.title_prefix} | ' if self.title_prefix else ''
        self.ax.set_title(f'{prefix}{len(self.vials)} vial(s) | margin {self.margin:.1f} px\n'
                          'Click = auto, drag centre→wall = manual, R-click/u remove, +/- margin, Enter done',
                          fontsize=10)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show(block=False)
        while not self.done and plt.fignum_exists(self.fig.number):
            plt.pause(0.1)
        return self.vials, self.margin


def pick_vials(img, max_radius_px, smooth_sigma=SMOOTH_SIGMA, margin=MARGIN_PX, title_prefix='',
               **picker_kwargs):
    """
    Run the interactive picker on a (magnitude) image.
    Returns dict(vials, masks [n_vials, ny, nx], roi_radii, margin, colors).
    Raises SystemExit if no vials were selected.
    """
    img = np.abs(np.asarray(img))
    img_seg = gaussian_filter(img, smooth_sigma)
    picker = VialPicker(img, img_seg, max_radius_px, margin=margin, title_prefix=title_prefix,
                        **picker_kwargs)
    vials, margin_final = picker.run()
    plt.close(picker.fig)
    if not vials:
        raise SystemExit('No vials selected.')

    roi_radii = np.array([picker.roi_radius(v) for v in vials])
    masks = np.array([disk_mask(img.shape, v['center'], r) for v, r in zip(vials, roi_radii)])
    colors = [picker.color(i) for i in range(len(vials))]
    for i, m in enumerate(masks):
        print(f'Vial {i + 1}: ROI radius {roi_radii[i]:.2f} px, {m.sum()} px')
    return dict(vials=vials, masks=masks, roi_radii=roi_radii, margin=margin_final, colors=colors)


# ========================================== #
# Stats, plotting, export
# ========================================== #
def roi_stats(stack, masks, ref=None):
    """
    Per-vial ROI statistics over a stack of images [n_images, ny, nx].
    Without ref: returns (mean, pixelwise SD) of the stack, each [n_vials, n_images].
    With ref [ny, nx]: returns (ratio of means stack/ref, pixelwise SD of stack/ref).
    """
    stack = np.asarray(stack)
    mean = np.array([stack[:, m].mean(axis=1) for m in masks])
    if ref is None:
        sd = np.array([stack[:, m].std(axis=1) for m in masks])
        return mean, sd
    ref_mean = np.array([ref[m].mean() for m in masks])
    norm = mean / ref_mean[:, None]
    sd = np.array([(stack[:, m] / ref[m]).std(axis=1) for m in masks])
    return norm, sd


def save_roi_overlay(img, res, out_fn, title=None):
    """Save the image with the final ROI circles and vial numbers drawn on it."""
    img = np.abs(img)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray', vmin=0, vmax=np.percentile(img, 99.5))
    for i, (v, r) in enumerate(zip(res['vials'], res['roi_radii'])):
        rc, cc = v['center']
        col = res['colors'][i]
        ax.add_patch(Circle((cc, rc), r, fill=False, ec=col, lw=2))
        ax.text(cc, rc, str(i + 1), color=col, ha='center', va='center', fontweight='bold')
    ax.set_axis_off()
    ax.set_title(title or f'Vial ROIs (margin {res["margin"]:.1f} px)')
    fig.savefig(out_fn, dpi=150, bbox_inches='tight')
    return fig


def plot_vial_grid(x, y_mean, y_std, colors, xlabel, ylabel, title=None, out_fn=None,
                   invert_x=False, ylim=None, marker='o-'):
    """One panel per vial (shared axes), mean line with a ±SD band."""
    n_vials = len(y_mean)
    ncols = int(np.ceil(np.sqrt(n_vials)))
    nrows = int(np.ceil(n_vials / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n_vials:
            ax.set_visible(False)
            continue
        col = colors[i]
        ax.plot(x, y_mean[i], marker, ms=3, lw=1.2, color=col)
        ax.fill_between(x, y_mean[i] - y_std[i], y_mean[i] + y_std[i], color=col, alpha=0.15)
        ax.set_title(f'Vial {i + 1}', color=col, fontweight='bold')
        ax.grid(alpha=0.3)
    if ylim is not None:
        axes[0, 0].set_ylim(*ylim)
    if invert_x:
        axes[0, 0].invert_xaxis()  # shared x, so this flips every panel
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out_fn:
        fig.savefig(out_fn, dpi=150)
    return fig


def roi_geometry(res):
    """ROI geometry as a dict ready to splat into np.savez(..., **roi_geometry(res))."""
    n = len(res['vials'])
    return dict(
        vial_labels=np.array([f'Vial {i + 1}' for i in range(n)]),
        n_pixels=res['masks'].sum(axis=(1, 2)),           # [n_vials]
        masks=res['masks'],                               # [n_vials, ny, nx]
        centers_rc=np.array([v['center'] for v in res['vials']]),
        fitted_radii_px=np.array([v['radius'] for v in res['vials']]),
        roi_radii_px=res['roi_radii'],
        margin_px=res['margin'],
    )
