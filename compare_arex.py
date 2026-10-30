#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-vendor AREX comparison of the creatine (Cr) effect, per vial, alongside the
per-vial B0 shift (mean +/- SD over pixels).

  AREX(dw) = R1 * (1/Z(+dw) - 1/Z(-dw))           (Zaiss et al., NMR Biomed 2014)

* Z is the ROI-mean Z-spectrum, normalised so the far offsets (|ppm| >= FAR_PPM)
  match the qMT MT-model prediction ('far_model'), or so they equal 1 ('far').
* B0: each ROI-mean spectrum is shifted so its water dip (Lorentzian fit) sits at
  0 ppm before taking +dw / -dw, by linear interpolation.
* AREX is averaged over CR_WINDOW (ppm, positive side). Error bars: the pixelwise
  SD of Z (z_std from the recon scripts) propagated through AREX, i.e. spread
  within the ROI, not the uncertainty of the mean.
* R1: the acquired IR T1 per vial. With READOUT_CORRECTION, R1_app = R1 + R_ro,
  R_ro = -ln(cos(alpha))/TR (the spoiled readout every TR acts as an extra
  longitudinal loss, so the pulsed-imaging steady state behaves like a CW Zss
  with R1 -> R1_app). Set False for the plain R1 definition.
* The mirror reference assumes the background (water, MT) is symmetric about
  water: fine for the Gaussian MT model used here, and rNOE at -3.5 ppm sits
  outside a 1.6-2.2 ppm window.

B0 per vial: the water frequency is fitted pixel by pixel (Lorentzian fit of each
pixel's dip) from the per-offset images ('images_sat' in the spectra npz, or a
Siemens zspec_recon_cache.npz). If a vendor has no images, its B0 falls back to
the single ROI-mean shift (no SD) and this is printed.
"""

# %% Step 0: Parameters
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq

GAMMA_HZ_T = 42.577478e6
B0_PROGRAMMED = 3.00     # B0 used by continuous_spiral.py to convert ppm -> Hz

out_dir = 'results/arex_compare'

# 'seq': None -> seq_fn stored in the npz. 'b0': field strength [T] used to convert Hz -> ppm.
# 'images': None -> 'images_sat' in the npz (written by recon_zspec.py); or an older
# Siemens zspec_recon_cache.npz.
DATASETS = {
    'GE': dict(npz='results/zspec_ge/ge_zspectra.npz', seq=None, b0=3.00,
               vial_map=[1, 2, 3, 4, 5], images=None),
    'Siemens': dict(npz='results/zspec_siemens/compare_ge_zspectra.npz', seq=None, b0=3.00,
                    vial_map=[1, 2, 3, 4, 5],
                    images='results/zspec_siemens/zspec_recon_cache.npz'),  # npz predates recon_zspec.py (no images_sat)
}
NORM = 'far'                 # 'far': Z(+/-10 ppm) = 1 | 'far_model': far offsets set to the qMT MT-model prediction
FAR_PPM = 9.5
M0_THRESH_PPM = 50.0
CR_WINDOW = (1.6, 2.2)       # ppm, AREX averaged over this (water-referenced)
READOUT_CORRECTION = False
B0_FIT_HALF_WIDTH = 0.6      # ppm either side of the dip used in the Lorentzian water fit

# T1 per reference vial (inversion recovery)
T1_CSV = 'results/t1_series/vial_t1.csv'
T1_COLUMN = 'roi_fit_t1_ms'

# qMT per reference vial (only used for NORM = 'far_model'; ignored with 'far')
R1B = 1.0
QMT = {
    1: dict(f=0.0141, k=1.0,  T2b=18.3e-6),
    2: dict(f=0.0290, k=15.9, T2b=17.9e-6),
    3: dict(f=0.0400, k=15.9, T2b=17.9e-6),
    4: dict(f=0.0162, k=1.0,  T2b=18.3e-6),
    5: dict(f=0.0122, k=1.0,  T2b=18.3e-6),
}


# %% Step 1: Helpers
def read_seq_sat(seq_fn):
    """Saturation-pulse power (CW equivalent over one TR), TR and flip from the .seq text."""
    sec, defs, rf, shapes, cur = None, {}, {}, {}, None
    with open(seq_fn) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('['):
                sec = s
                continue
            if sec == '[DEFINITIONS]':
                k, *v = s.split()
                defs[k] = v
            elif sec == '[RF]':
                p = s.split()
                rf[int(p[0])] = p
            elif sec == '[SHAPES]':
                if s.startswith('shape_id'):
                    cur = int(s.split()[1])
                    shapes[cur] = dict(n=None, v=[])
                elif s.startswith('num_samples'):
                    shapes[cur]['n'] = int(s.split()[1])
                else:
                    shapes[cur]['v'].append(float(s))

    def decompress(v, n):
        v = np.asarray(v)
        if len(v) == n:
            return v
        w, i = [], 0
        while i < len(v):
            if i < len(v) - 2 and v[i] == v[i + 1]:
                w += [v[i]] * (int(v[i + 2]) + 2)
                i += 3
            else:
                w.append(v[i])
                i += 1
        return np.cumsum(w)

    prep = [p for p in rf.values() if p[-1] == 'p']
    if not prep:
        raise ValueError(f'No preparation RF pulse found in {seq_fn}')
    p = prep[0]
    amp = float(p[1])
    mag = np.abs(decompress(shapes[int(p[2])]['v'], shapes[int(p[2])]['n']))
    rf_raster = float(defs.get('RadiofrequencyRasterTime', ['1e-6'])[0])
    if int(p[4]) == 0:
        dt = np.full(len(mag), rf_raster)
    else:
        t = decompress(shapes[int(p[4])]['v'], shapes[int(p[4])]['n']) * rf_raster
        dt = np.diff(np.append(t, t[-1] + rf_raster))
    tr = float(defs['TR'][0])
    energy = np.sum((amp * mag) ** 2 * dt)
    return dict(w1_cw=2 * np.pi * np.sqrt(energy / tr), b1rms_tr_uT=np.sqrt(energy / tr) / GAMMA_HZ_T * 1e6,
                tr=tr, flip_deg=float(defs['FlipAngle'][0]))


def r1a_from_r1obs(r1obs, f, k, r1b=R1B):
    kf, kr = k * f, k
    if kf < 1e-6:
        return r1obs

    def slow_rate(r1a):
        a = np.array([[-(r1a + kf), kr], [kf, -(r1b + kr)]])
        return -np.max(np.linalg.eigvals(a).real) - r1obs
    return brentq(slow_rate, 1e-4, 50.0)


def z_mt(dw, w1, r1a, f, k, t2b, r_ro, r1b=R1B):
    """Two-pool (water + Gaussian semisolid) steady state, normalised to no saturation."""
    kf, kr = k * f, k
    rrfb = w1 ** 2 * t2b * np.sqrt(np.pi / 2) * np.exp(-0.5 * (dw * t2b) ** 2)

    def ma(rrf):
        a11, a12, a21, a22 = -(r1a + kf + r_ro), kr, kf, -(r1b + kr + rrf)
        return (-r1a * a22 + a12 * r1b * f) / (a11 * a22 - a12 * a21)
    return ma(rrfb) / ma(0.0)


def water_shift(ppm, z, half_width=B0_FIT_HALF_WIDTH):
    """Water frequency [ppm] from a Lorentzian fit of the dip (nan if it fails)."""
    x0 = ppm[np.argmin(z)]
    sel = np.abs(ppm - x0) <= half_width
    if sel.sum() < 5:
        return np.nan
    xs, zs = ppm[sel], z[sel]

    def res(p):
        d, a, g, c = p
        return c - a * g ** 2 / (g ** 2 + (xs - d) ** 2) - zs
    try:
        return least_squares(res, [x0, max(1 - zs.min(), 0.05), 0.4, 1.0],
                             bounds=([x0 - half_width, 0, 0.02, 0.5], [x0 + half_width, 1.5, 3.0, 1.5])).x[0]
    except ValueError:
        return np.nan


def load_sat_images(cfg, d, offs_nom):
    """Saturated-offset magnitude images [n_off, ny, nx] in the npz's sorted offset order, or None."""
    if cfg.get('images') is None:
        return np.abs(d['images_sat']).astype(float) if 'images_sat' in d.files else None
    if not os.path.exists(cfg['images']):
        return None
    c = np.load(cfg['images'], allow_pickle=True)
    if 'data_fn' in c.files and 'data_fn' in d.files and str(c['data_fn']) != str(d['data_fn']):
        raise ValueError(f"{cfg['images']} is from {c['data_fn']}, spectra are from {d['data_fn']}")
    offs_acq = np.asarray(c['offsets_acq'], float)
    sat = [i for i in range(len(offs_acq)) if abs(offs_acq[i]) < M0_THRESH_PPM]
    keep = [i for i in sat if not any(j > i and np.isclose(offs_acq[j], offs_acq[i]) for j in sat)]
    keep = np.array(keep)[np.argsort(offs_acq[keep])]
    if not np.allclose(offs_acq[keep], offs_nom):
        raise ValueError('Offsets in the image cache do not match the spectra file.')
    return np.abs(c['image_stack'][keep]).astype(float)


def load_t1(fn, column):
    data = np.genfromtxt(fn, delimiter=',', names=True)
    return {int(v): float(t) * 1e-3 for v, t in zip(np.atleast_1d(data['vial']), np.atleast_1d(data[column]))}


# %% Step 2: AREX and B0 per vial
os.makedirs(out_dir, exist_ok=True)
t1 = load_t1(T1_CSV, T1_COLUMN)
xw = np.linspace(*CR_WINDOW, 61)
x_spec = np.linspace(0.5, 5.0, 181)        # for the AREX spectra plot
res = {}

for vendor, cfg in DATASETS.items():
    d = np.load(cfg['npz'], allow_pickle=True)
    seq_fn = cfg['seq'] or str(d['seq_fn'])
    acq = read_seq_sat(seq_fn)
    r_ro = -np.log(np.cos(np.deg2rad(acq['flip_deg']))) / acq['tr'] if READOUT_CORRECTION else 0.0
    print(f"\n=== {vendor} ({cfg['b0']:.2f} T): {seq_fn}\n    TR {acq['tr']*1e3:.2f} ms, "
          f"B1rms over TR {acq['b1rms_tr_uT']:.3f} uT, flip {acq['flip_deg']:.0f} deg -> R_ro {r_ro:.3f} s^-1")
    offs_nom = np.asarray(d['offsets_ppm'], float)
    ppm = offs_nom * B0_PROGRAMMED / cfg['b0']
    ppm_per_rad = 1e6 / (2 * np.pi * GAMMA_HZ_T * cfg['b0'])
    far = np.abs(offs_nom) >= FAR_PPM
    masks = d['masks'].astype(bool)
    stack = load_sat_images(cfg, d, offs_nom)
    if stack is None:
        print(f"    No per-offset images for {vendor}: B0 SD unavailable, using the ROI-mean shift only.")

    res[vendor] = {}
    for i, ref in enumerate(cfg['vial_map']):
        r1 = 1 / t1[ref] + r_ro
        z, z_sd = d['z_mean'][i].astype(float), d['z_std'][i].astype(float)
        if NORM == 'far_model':
            mt = QMT[ref]
            r1a = r1a_from_r1obs(1 / t1[ref], mt['f'], mt['k'])
            target = np.mean(z_mt(ppm[far] / ppm_per_rad, acq['w1_cw'], r1a, mt['f'], mt['k'], mt['T2b'], r_ro))
        elif NORM == 'far':
            target = 1.0
        else:
            raise ValueError("NORM must be 'far_model' or 'far'")
        scale = target / z[far].mean()
        z, z_sd = z * scale, z_sd * scale

        d0 = water_shift(ppm, z)
        zp, zm = np.interp(d0 + xw, ppm, z), np.interp(d0 - xw, ppm, z)
        sp, sm = np.interp(d0 + xw, ppm, z_sd), np.interp(d0 - xw, ppm, z_sd)
        arex_w = r1 * (1 / zp - 1 / zm)
        arex_sd = r1 * np.sqrt((sp / zp ** 2) ** 2 + (sm / zm ** 2) ** 2)
        arex_spec = r1 * (1 / np.interp(d0 + x_spec, ppm, z) - 1 / np.interp(d0 - x_spec, ppm, z))

        if stack is not None:
            pix = stack[:, masks[i]]
            zpix = pix / pix[far].mean(axis=0)
            shifts = np.array([water_shift(ppm, zpix[:, j]) for j in range(zpix.shape[1])])
            shifts = shifts[np.isfinite(shifts)]
            b0_mean, b0_sd, npx = shifts.mean(), shifts.std(ddof=1), len(shifts)
        else:
            b0_mean, b0_sd, npx = d0, np.nan, int(masks[i].sum())

        res[vendor][ref] = dict(arex=arex_w.mean(), arex_sd=np.sqrt(np.mean(arex_sd ** 2)), arex_spec=arex_spec,
                                d0=d0, b0_mean=b0_mean, b0_sd=b0_sd, n_pix=npx, r1=r1)
        print(f"  vial {i+1} -> ref {ref}: AREX {arex_w.mean():.4f} ± {res[vendor][ref]['arex_sd']:.4f} s^-1 "
              f"(R1 {r1:.3f} s^-1) | ROI water shift {d0:+.2f} ppm | "
              + (f"pixel B0 {b0_mean:+.2f} ± {b0_sd:.2f} ppm ({b0_mean / ppm_per_rad / (2 * np.pi):+.0f} Hz, "
                 f"{npx} px)" if np.isfinite(b0_sd) else "no pixel B0 (no images)"))

# %% Step 3: Table + figures
vendors = list(res)
refs = sorted(set.intersection(*[set(v) for v in res.values()]))
with open(os.path.join(out_dir, 'arex_b0.csv'), 'w') as fh:
    fh.write('vendor,ref_vial,arex_s,arex_sd_s,r1_used_s,roi_water_shift_ppm,b0_mean_ppm,b0_sd_ppm,n_pix\n')
    for v in vendors:
        for r in refs:
            q = res[v][r]
            fh.write(f"{v},{r},{q['arex']:.6g},{q['arex_sd']:.6g},{q['r1']:.5g},{q['d0']:.4g},"
                     f"{q['b0_mean']:.4g},{q['b0_sd']:.4g},{q['n_pix']}\n")

fig = plt.figure(figsize=(3.2 * len(refs), 7.5))
gs = fig.add_gridspec(2, len(refs), height_ratios=[1, 1.1])
x = np.arange(len(refs))
width = 0.8 / len(vendors)

ax = fig.add_subplot(gs[0, :len(refs) // 2 + len(refs) % 2])
for k, v in enumerate(vendors):
    ax.bar(x + (k - (len(vendors) - 1) / 2) * width, [res[v][r]['arex'] for r in refs], width,
           yerr=[res[v][r]['arex_sd'] for r in refs], capsize=3, color=f'C{k}', label=v)
ax.set_xticks(x, [str(r) for r in refs]); ax.set_xlabel('Vial')
ax.set_ylabel(f'AREX, {CR_WINDOW[0]}-{CR_WINDOW[1]} ppm [s$^{{-1}}$]')
ax.set_title('Cr AREX (mean ± pixelwise SD)', fontsize=10); ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, len(refs) // 2 + len(refs) % 2:])
for k, v in enumerate(vendors):
    ax.errorbar(x + (k - (len(vendors) - 1) / 2) * width, [res[v][r]['b0_mean'] for r in refs],
                yerr=[0 if np.isnan(res[v][r]['b0_sd']) else res[v][r]['b0_sd'] for r in refs],
                fmt='o', capsize=4, color=f'C{k}', label=v)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xticks(x, [str(r) for r in refs]); ax.set_xlabel('Vial'); ax.set_ylabel('water shift [ppm]')
ax.set_title('B0 shift per vial (pixel mean ± SD)', fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)

for j, r in enumerate(refs):
    ax = fig.add_subplot(gs[1, j])
    for k, v in enumerate(vendors):
        ax.plot(x_spec, res[v][r]['arex_spec'], color=f'C{k}', lw=1.5, label=v)
    ax.axvspan(*CR_WINDOW, color='0.85', zorder=0)
    ax.axhline(0, color='0.5', lw=0.8)
    vals = np.concatenate([res[v][r]['arex_spec'][x_spec >= 1.0] for v in vendors])
    pad = 0.15 * (vals.max() - vals.min())
    ax.set_ylim(vals.min() - pad, vals.max() + pad)   # scale to > 1 ppm; the water edge below runs off
    ax.set_xlim(5, 0.5); ax.set_xlabel('ppm'); ax.grid(alpha=0.3)
    ax.set_title(f'Vial {r}', fontsize=10)
    if j == 0:
        ax.set_ylabel('AREX [s$^{-1}$]'); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'arex_b0_comparison.png'), dpi=150)
print(f'\nSaved arex_b0.csv and arex_b0_comparison.png to {out_dir}/')
plt.show()