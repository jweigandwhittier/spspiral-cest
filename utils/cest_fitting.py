#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:30:15 2026
@author: jonah
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# --- Model ---------------------------------------------------------------

def lorentzian(x, amp, fwhm, offset):
    num = amp * 0.25 * fwhm ** 2
    den = 0.25 * fwhm ** 2 + (x - offset) ** 2
    return num / den


def step_1_fit(x, *p):
    """Water + MT two-pool baseline."""
    water_fit = lorentzian(x, p[0], p[1], p[2])
    mt_fit = lorentzian(x, p[3], p[4], p[5])
    return 1 - water_fit - mt_fit


# --- Starting points / bounds (post-correction values from the app) ------

P0_WATER, LB_WATER, UB_WATER = [0.8, 0.2, 0], [0.02, 0.01, -1e-6], [1, 10, 1e-6]
P0_MT, LB_MT, UB_MT = [0.15, 40, -1], [0.0, 30, -2.5], [0.5, 60, 0]

P0_1 = P0_WATER + P0_MT
LB_1 = LB_WATER + LB_MT
UB_1 = UB_WATER + UB_MT

# --- Pre-correction water+MT bounds, used only for the B0 shift fit ------
P0_WATER_PRE, LB_WATER_PRE, UB_WATER_PRE = [0.8, 1.8, 0], [0.02, 0.3, -10], [1, 10, 10]
P0_MT_PRE, LB_MT_PRE, UB_MT_PRE = [0.15, 40, -1], [0.0, 30, -2.5], [0.5, 60, 0]

P0_CORR = P0_WATER_PRE + P0_MT_PRE
LB_CORR = LB_WATER_PRE + LB_MT_PRE
UB_CORR = UB_WATER_PRE + UB_MT_PRE

# Name -> (p0, lb, ub) as [amplitude, FWHM, offset_ppm]
CONTRAST_PARAMS = {
    'NOE (-3.5 ppm)':  ([0.05, 1, -3.50],  [0.0, 0.5, -4.0], [0.25, 5, -1.5]),
    'Creatine':        ([0.05, 1.5, 2.0],  [0.0, 0.5, 1.6],  [0.5, 8, 2.6]),
    'Amide':           ([0.05, 1.5, 3.5],  [0.0, 0.5, 3.2],  [0.3, 5, 4.0]),
    'Amine':           ([0.05, 1.5, 2.5],  [0.0, 0.1, 2.2],  [0.3, 5, 2.8]),
    'Hydroxyl':        ([0.05, 1.5, 0.6],  [0.0, 0.1, 0.4],  [0.3, 5, 1.2]),
    'NOE (-1.6 ppm)':  ([0.05, 1, -1.6],   [0.0, 0.5, -1.8], [.25, 5, -1.2]),
    'Salicylic acid':  ([0.05, 1.5, 9.3],  [0.0, 0.5, 8.0],  [0.3, 5, 10.0]),
}

DEFAULT_CONTRASTS = ['Amide', 'Creatine', 'NOE (-3.5 ppm)']

CONTRAST_COLORS = {
    'Water_Fit': '#0072BD',
    'MT_Fit': '#EDB120',
    'NOE (-3.5 ppm)_Fit': '#A6761D',
    'Amide_Fit': '#7E2F8E',
    'Amine_Fit': '#F8961E',
    'Creatine_Fit': '#6F1D1B',
    'Hydroxyl_Fit': '#4DBEEE',
    'NOE (-1.6 ppm)_Fit': '#E144C4',
}

CUTOFFS = [-4, -1.4, 1.4, 4]
FIT_OPTIONS = {'xtol': 1e-10, 'ftol': 1e-4, 'maxfev': 50}


# --- B0 correction -----------------------------------------------------

def b0_correct(spectrum, offsets):
    """
    Fits a water+MT baseline with loose (pre-correction) bounds to find
    the water peak center, then shifts offsets so the peak sits at 0 ppm.

    Parameters
    ----------
    spectrum : array-like, S/S0 values (no unsaturated reference point).
    offsets : array-like, saturation offsets in ppm, same order as spectrum.
        Does NOT need to be pre-sorted - this flips to descending order
        internally, matching two_step's convention.

    Returns
    -------
    offsets_corrected : ndarray
        `offsets` (descending order) shifted by the fitted water peak center.
    shift : float
        The fitted peak center (ppm) that was subtracted off.
    fit_corr : ndarray
        Raw curve_fit output [water_amp, water_fwhm, water_center,
        mt_amp, mt_fwhm, mt_center] from the correction fit.
    """
    offsets = np.asarray(offsets, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    if offsets[0] > 0:
        offsets = np.flip(offsets)
        spectrum = np.flip(spectrum)

    fit_corr, _ = curve_fit(step_1_fit, offsets, spectrum, p0=P0_CORR,
                             bounds=(LB_CORR, UB_CORR), **FIT_OPTIONS)
    shift = fit_corr[2]
    offsets_corrected = offsets - shift
    return offsets_corrected, shift, fit_corr


# --- Fitting ---------------------------------------------------------------

def two_step(spectrum, offsets, custom_contrasts=None, n_interp=4000,
             correct_b0=True, cutoffs=None):
    """
    Two-step Lorentzian Z-spectrum fit.

    Step 1 fits a water + MT two-pool baseline across the full spectrum.
    Step 2 fits the named contrasts (Lorentzian pools) on the residual
    ("Lorentzian difference") between the data and the step-1 baseline.

    Parameters
    ----------
    spectrum : array-like
        S/S0 values, one per offset. Should NOT include an unsaturated
        reference point (e.g. drop the 75 ppm / M0 entry before calling).
    offsets : array-like
        Saturation offsets in ppm, same length/order as `spectrum`.
        Raw (uncorrected) offsets are fine - see `correct_b0`.
    custom_contrasts : list[str], optional
        Names from CONTRAST_PARAMS to fit in step 2.
        Defaults to DEFAULT_CONTRASTS.
    n_interp : int
        Number of points in the interpolated fit curves.
    correct_b0 : bool, default True
        If True, runs b0_correct() first (loose water+MT fit) and shifts
        `offsets` so the fitted water peak sits at 0 ppm before doing the
        two-step fit. Set False to skip and fit on raw offsets as-is.
    cutoffs : list[float], optional
        [neg_far, neg_near, pos_near, pos_far] band edges (ppm) defining
        which offsets are excluded from the step-1 water+MT fit (the
        shoulder bands, where contrast peaks live). Defaults to
        module-level CUTOFFS = [-4, -1.4, 1.4, 4]. Widen this (e.g.
        [-6, -1.4, 1.4, 6]) if a contrast peak (like a broad Creatine
        pool) has real signal extending past the default shoulder edges -
        otherwise its wings leak into the "baseline" fit and distort the
        water peak the same way an un-cropped fit would.

    Returns
    -------
    dict with keys:
        'Fit_Params' : [fit_1, fit_2] raw curve_fit outputs (post-correction
                       water+MT fit, then contrast fit)
        'B0_Shift'   : float, the ppm shift applied (0.0 if correct_b0=False)
        'Data_Dict'  : dict of arrays for plotting (Zspec, Offsets,
                       Offsets_Interp, Water_Fit, MT_Fit, <name>_Fit...,
                       Lorentzian_Difference). 'Offsets' is post-correction
                       if correct_b0=True.
        'Contrasts'  : dict of {name: percent contrast}
        'RMSE'       : float, fit residual over the full spectrum

    Raises
    ------
    RuntimeError
        If either curve_fit stage fails to converge (propagated from
        scipy - no silent zero-fallback, since this is meant for
        single-spectrum debugging/reuse rather than batch pixel fits).
    """
    if custom_contrasts is None:
        custom_contrasts = DEFAULT_CONTRASTS

    p0_2, lb_2, ub_2 = [], [], []
    for c in custom_contrasts:
        p, lb, ub = CONTRAST_PARAMS[c]
        p0_2 += p; lb_2 += lb; ub_2 += ub

    def step_2_fit(x, *params):
        fit_sum = np.zeros_like(x)
        idx = 0
        for _ in custom_contrasts:
            fit_sum += lorentzian(x, params[idx], params[idx + 1], params[idx + 2])
            idx += 3
        return fit_sum

    offsets = np.asarray(offsets, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    if offsets[0] > 0:
        offsets = np.flip(offsets)
        spectrum = np.flip(spectrum)

    if correct_b0:
        offsets, b0_shift, _ = b0_correct(spectrum, offsets)
    else:
        b0_shift = 0.0

    # --- Step 1: water + MT baseline, fit on cropped offsets only ---
    # Exclude the shoulder bands (where real contrast peaks - Amide,
    # Creatine, NOE, etc. - live) so they don't get absorbed into the
    # water/MT baseline fit; keep the central band and far tails, which
    # is genuine water+MT signal. Without this, contrast-driven dips in
    # the shoulder get fit as if they were baseline, and the optimizer's
    # only way to compensate is to over-narrow the water peak to nail the
    # deep central point at the shoulders' expense.
    cutoffs = list(cutoffs) if cutoffs is not None else list(CUTOFFS)
    if 'Hydroxyl' in custom_contrasts:
        cutoffs[2] = 0.4  # Hydroxyl sits close to water; shrink the kept band
    crop_condition = (offsets <= cutoffs[0]) | (offsets >= cutoffs[3]) | \
                      ((offsets >= cutoffs[1]) & (offsets <= cutoffs[2]))
    if not np.any(crop_condition):
        raise RuntimeError("No offsets remain after cropping for the step-1 baseline fit "
                            "- check that `offsets` spans beyond the CUTOFFS band.")
    offsets_cropped = offsets[crop_condition]
    spectrum_cropped = spectrum[crop_condition]

    fit_1, _ = curve_fit(step_1_fit, offsets_cropped, spectrum_cropped, p0=P0_1,
                          bounds=(LB_1, UB_1), **FIT_OPTIONS)

    offsets_interp = np.linspace(offsets[0], offsets[-1], n_interp)
    water_fit = lorentzian(offsets_interp, fit_1[0], fit_1[1], fit_1[2])
    mt_fit = lorentzian(offsets_interp, fit_1[3], fit_1[4], fit_1[5])
    background = lorentzian(offsets, fit_1[0], fit_1[1], fit_1[2]) + \
                 lorentzian(offsets, fit_1[3], fit_1[4], fit_1[5])
    lorentzian_difference = 1 - (spectrum + background)

    # --- Step 2: named contrasts on the Lorentzian difference ---
    fit_2, _ = curve_fit(step_2_fit, offsets, lorentzian_difference, p0=p0_2,
                          bounds=(lb_2, ub_2), **FIT_OPTIONS)

    fit_curves = {}
    idx = 0
    for c in custom_contrasts:
        fit_curves[c] = lorentzian(offsets_interp, fit_2[idx], fit_2[idx + 1], fit_2[idx + 2])
        idx += 3

    step_1_vals = step_1_fit(offsets, *fit_1)
    step_2_vals = step_2_fit(offsets, *fit_2)
    total_fit = step_1_vals - step_2_vals

    # RMSE over the shoulder bands only (where contrasts live), matching the
    # app's convention - the full-spectrum RMSE is dominated by the deep
    # central water dip and doesn't reflect contrast-fit quality.
    rmse_condition = ((offsets <= cutoffs[1]) & (offsets >= cutoffs[0])) | \
                      ((offsets >= cutoffs[2]) & (offsets <= cutoffs[3]))
    if np.any(rmse_condition):
        rmse = np.sqrt(mean_squared_error(spectrum[rmse_condition], total_fit[rmse_condition]))
    else:
        rmse = np.sqrt(mean_squared_error(spectrum, total_fit))

    # Flip back to display convention (descending -> ascending ppm was
    # flipped above; flip fit curves back to match original ordering)
    offsets_interp = np.flip(offsets_interp)
    water_fit = np.flip(water_fit)
    mt_fit = np.flip(mt_fit)
    fit_curves_named = {f"{c}_Fit": np.flip(fit_curves[c]) for c in fit_curves}

    contrasts = {'Water': 100 * fit_1[0], 'MT': 100 * fit_1[3]}
    for i, c in enumerate(custom_contrasts):
        contrasts[c] = 100 * fit_2[i * 3]

    data_dict = {
        'Zspec': spectrum, 'Offsets': offsets, 'Offsets_Interp': offsets_interp,
        'Water_Fit': water_fit, 'MT_Fit': mt_fit,
        **fit_curves_named, 'Lorentzian_Difference': lorentzian_difference,
    }
    return {'Fit_Params': [fit_1, fit_2], 'B0_Shift': b0_shift,
            'Data_Dict': data_dict, 'Contrasts': contrasts, 'RMSE': rmse}


# --- Plotting ----------------------------------------------------------

def plot_fit(fit, title="Z-Spectrum", ax=None, save_path=None):
    """
    Plots a two_step() fit result in the app's plot_zspec style: raw data
    as black open circles, each individual pool (water, MT, and every
    fitted contrast) as its own colored curve, and the summed total fit
    as a solid orange line.

    Parameters
    ----------
    fit : dict
        The dict returned by two_step().
    title : str
        Plot title / ROI label (e.g. an ROI name).
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into (e.g. for a multi-ROI subplot grid).
        If None, creates a new figure/axes at figsize=(12, 10).
    save_path : str, optional
        If given, saves the figure as a PNG at this path
        (dpi=300, bbox_inches='tight'). Only applies when ax is None,
        since a shared multi-panel figure should be saved by the caller.

    Returns
    -------
    matplotlib.axes.Axes
    """
    data = fit['Data_Dict']
    offsets = data['Offsets']
    offsets_interp = data['Offsets_Interp']
    spectrum = data['Zspec']

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    ax.plot(offsets, spectrum, '.', markersize=15, fillstyle='none',
            color='black', label="Raw")

    fit_curves = {k: v for k, v in data.items() if k.endswith('_Fit')}
    color_cycle = itertools.cycle(plt.get_cmap('viridis').colors)
    total_fit = np.zeros_like(offsets_interp)
    for name, curve in fit_curves.items():
        color = CONTRAST_COLORS.get(name, next(color_cycle))
        label = name.replace('_Fit', '')
        ax.plot(offsets_interp, 1 - curve, linewidth=4, color=color, label=label)
        total_fit += curve
    ax.plot(offsets_interp, 1 - total_fit, linewidth=4, color='#D95319', label="Fit")

    ax.legend(fontsize=16)
    ax.invert_xaxis()
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim([0, 1])
    ax.set_xlabel("Offset frequency (ppm)", fontsize=18, fontname='Arial')
    ax.set_ylabel("$S/S_0$", fontsize=18, fontname='Arial')
    ax.set_title(title, fontsize=28, fontweight='bold', fontname='Arial')
    ax.grid(False)

    if fig is not None and save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_fits(fits_by_roi, n_cols=3, save_path=None):
    """
    Plots multiple ROI fits in a grid, one panel per ROI, matching the
    app's multi-ROI plot_zspec layout.

    Parameters
    ----------
    fits_by_roi : dict[str, dict]
        Mapping of ROI name -> two_step() result dict.
    n_cols : int
        Number of columns in the subplot grid.
    save_path : str, optional
        If given, saves the whole grid figure as a PNG
        (dpi=300, bbox_inches='tight').

    Returns
    -------
    matplotlib.figure.Figure
    """
    roi_names = list(fits_by_roi.keys())
    n_rows = -(-len(roi_names) // n_cols)  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 10 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, roi in zip(axes, roi_names):
        plot_fit(fits_by_roi[roi], title=roi, ax=ax)
    for ax in axes[len(roi_names):]:
        ax.axis('off')

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig