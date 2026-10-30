#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spsp_exc.py -- spectral-spatial (SPSP) water excitation with an exact lipid null.

Design summary
--------------
* Flyback gradient train, RF played on the excitation-lobe plateaus only.
* Sub-pulse period T_s is chosen so lipid sits midway between spectral
  replicas: f_fat * T_s = k + 1/2 (smallest k whose lobes fit).
* Spectral envelope: symmetric least-squares filter with a Lagrange
  constraint forcing beta(f_fat) = 0, converted to rotations with SLR (b2rf).
  In the hard-pulse SPSP model this is an exact Bloch null at f_fat.
* Spatial sub-pulse: SigPy small-tip SLR (dzrf) on the plateau.
* Whole train is ONE extended-trapezoid gradient and ONE arbitrary RF event,
  so rf_dead_time / ringdown are applied once and T_s stays exact.

The above is mode='slr' (default). mode='binomial' is the short design of
MRM 2004 (doi:10.1002/mrm.20053):
* 3 sub-pulses with a fixed 1-2-1 binomial envelope (exact fat null at any
  flip in the hard-pulse model; double zero -> flat null).
* T_s fixed at k = 0 (f_fat * T_s = 1/2, first spectral null on lipid).
* Excitation lobe and flyback each get ~T_s/2; RF is played over the WHOLE
  excitation lobe, ramps included, using VERSE: b(t) = p0(s(t)) * G(t),
  s(t) = k(t) / A_lobe. Spatial TBW defaults to 3.5.

Public API
----------
design_spsp_exc(system, flip_angle, slice_thickness, mode='slr', ...) -> (rf, gz, info)
design_matched_subpulse(system, flip_angle, slice_thickness, info) -> (rf, gz, gzr)
bloch_sim(rf, gz, z, df, ...)                            -> Mxy
verify_spsp(rf, gz, info, ...)                           -> metrics dict

Units follow PyPulseq: gradients in Hz/m, k-space in 1/m, RF in Hz, seconds.

@author: jonah
"""

from types import SimpleNamespace

import numpy as np
import pypulseq as pp
import sigpy.mri.rf as sprf


# ----------------------------------------------------------------------------
# Raster helpers
# ----------------------------------------------------------------------------
def _ceil_to(x, dt):
    return np.ceil(x / dt - 1e-6) * dt


def _round_to(x, dt):
    return np.round(x / dt) * dt


def _trap_for_area(area, system, duration=None):
    """Trapezoid (ramp, flat, amplitude) on the gradient raster.

    duration=None -> minimum-time trapezoid for `area`.
    duration given -> lowest-amplitude trapezoid with exactly that duration.
    Returns None if infeasible.
    """
    dt = system.grad_raster_time
    S, Gmax = system.max_slew, system.max_grad
    area = abs(area)
    if duration is None:
        best = None
        for n_r in range(1, 10000):
            tr = n_r * dt
            G_r = min(S * tr, Gmax)
            if G_r * tr >= area:                      # triangle suffices
                G = area / tr
                return tr, 0.0, G
            tf = _ceil_to((area - G_r * tr) / G_r, dt)
            G = area / (tr + tf)
            cand = (2 * tr + tf, tr, tf, G)
            if best is None or cand[0] < best[0] - 1e-12:
                best = cand
            if S * tr >= Gmax:                        # ramps can't get more useful
                break
        return best[1], best[2], best[3]
    D = _round_to(duration, dt)
    for n_r in range(1, int(round(D / dt / 2)) + 1):
        tr = n_r * dt
        tf = D - 2 * tr
        G = area / (tr + tf)
        if G <= S * tr + 1e-6 and G <= Gmax:
            return tr, tf, G
    return None


# ----------------------------------------------------------------------------
# Gradient train
# ----------------------------------------------------------------------------
def _exc_lobe_geometry(system, A_flat, T_s):
    """Plateau-only excitation lobe + flyback that fit exactly in T_s.

    Chooses the longest plateau that still fits (lowest gradient amplitude and
    lowest peak B1). Returns SimpleNamespace or None if nothing fits.
    """
    dt = system.grad_raster_time
    S, Gmax = system.max_slew, system.max_grad
    best = None
    for n_f in range(1, int(T_s / dt)):
        tf = n_f * dt
        G = A_flat / tf
        if G > Gmax:
            continue
        tr = _ceil_to(G / S, dt)
        lobe = 2 * tr + tf
        A_tot = G * (tf + tr)
        fb_min = _trap_for_area(A_tot, system)
        if lobe + 2 * fb_min[0] + fb_min[1] > T_s + 1e-9:
            continue
        fb = _trap_for_area(A_tot, system, duration=T_s - lobe)
        if fb is None:
            continue
        best = SimpleNamespace(t_ramp=tr, t_flat=tf, amp=G, A_tot=A_tot,
                               lobe_dur=lobe,
                               fb_ramp=fb[0], fb_flat=fb[1], fb_amp=fb[2])
    return best


def _verse_lobe_geometry(system, A_lobe, T_s):
    """Binomial/VERSE lobe + flyback splitting T_s roughly in half.

    RF covers the whole excitation lobe, so A_lobe is the TOTAL lobe area
    (ramps included). Each half is the lowest-amplitude trapezoid with that
    area and duration (equal split maximises reachable area). None if it
    doesn't fit.
    """
    dt = system.grad_raster_time
    D_exc = np.floor(T_s / 2 / dt + 1e-6) * dt
    D_fb = _round_to(T_s - D_exc, dt)
    exc = _trap_for_area(A_lobe, system, duration=D_exc)
    fb = _trap_for_area(A_lobe, system, duration=D_fb)
    if exc is None or fb is None:
        return None
    return SimpleNamespace(t_ramp=exc[0], t_flat=exc[1], amp=exc[2], A_tot=A_lobe,
                           lobe_dur=D_exc,
                           fb_ramp=fb[0], fb_flat=fb[1], fb_amp=fb[2])


def make_spsp_grad(system, slice_thickness, fwhm_tbw, n_sub, f_fat, k_max=6,
                   mode='slr'):
    """Flyback SPSP gradient as a single extended trapezoid.

    fwhm_tbw: FWHM of the sub-pulse profile in cycles (from _profile_fwhm), so the
    RF-bearing area makes the *measured* FWHM equal slice_thickness.
    mode='slr'      : plateau-only RF, smallest k that fits.
    mode='binomial' : RF over the whole lobe (VERSE), k = 0 enforced.
    Returns (gz, geom) where geom holds T_s and lobe timings.
    """
    dt = system.grad_raster_time
    A_rf = fwhm_tbw / slice_thickness                 # 1/m under the RF

    if mode == 'binomial':
        k = 0
        T_s = _round_to(0.5 / abs(f_fat), dt)         # first null on lipid
        lobe = _verse_lobe_geometry(system, A_rf, T_s)
        if lobe is None:
            raise ValueError(
                f'Binomial: lobe area {A_rf:.0f} 1/m does not fit in T_s/2 = '
                f'{T_s/2*1e6:.0f} us; use a thicker slice, lower tbw, or mode="slr".')
    else:
        for k in range(k_max + 1):                    # fat midway between replicas
            T_s = _round_to((k + 0.5) / abs(f_fat), dt)
            lobe = _exc_lobe_geometry(system, A_rf, T_s)
            if lobe is not None:
                break
        else:
            raise ValueError('No flyback geometry fits; lower tbw or raise limits.')

    tr, tf, G = lobe.t_ramp, lobe.t_flat, lobe.amp
    times, amps, t = [0.0], [0.0], 0.0
    plateau_starts, lobe_starts = [], []
    for n in range(n_sub):
        lobe_starts.append(t)
        plateau_starts.append(t + tr)
        pts_t = [t + tr, t + tr + tf, t + 2 * tr + tf]
        pts_a = [G, G, 0.0]
        if tf == 0:                                   # triangle (binomial only)
            pts_t, pts_a = [pts_t[0], pts_t[2]], [pts_a[0], pts_a[2]]
        times += pts_t
        amps += pts_a
        t += 2 * tr + tf
        if n < n_sub - 1:
            fr, ff, fG = lobe.fb_ramp, lobe.fb_flat, lobe.fb_amp
            pts_t = [t + fr, t + fr + ff, t + 2 * fr + ff]
            pts_a = [-fG, -fG, 0.0]
            if ff == 0:                               # triangle: drop duplicate corner
                pts_t, pts_a = [pts_t[0], pts_t[2]], [pts_a[0], pts_a[2]]
            times += pts_t
            amps += pts_a
            t += 2 * fr + ff

    # slice refocus: -(area after the RF centre of the last lobe) = -A_tot/2
    rr, rf_, rG = _trap_for_area(lobe.A_tot / 2, system)
    pts_t = [t + rr, t + rr + rf_, t + 2 * rr + rf_]
    pts_a = [-rG, -rG, 0.0]
    if rf_ == 0:
        pts_t, pts_a = [pts_t[0], pts_t[2]], [pts_a[0], pts_a[2]]
    times += pts_t
    amps += pts_a

    gz = pp.make_extended_trapezoid('z', amplitudes=np.array(amps),
                                    times=_round_to(np.array(times), dt),
                                    system=system)
    geom = SimpleNamespace(T_s=T_s, k=k, plateau_starts=np.array(plateau_starts),
                           lobe_starts=np.array(lobe_starts), **vars(lobe))
    return gz, geom


# ----------------------------------------------------------------------------
# Spectral envelope
# ----------------------------------------------------------------------------
def design_spectral_env(n_sub, T_s, flip_angle, f_fat, pb_hw=100.0, sb_hw=100.0,
                        n_grid=8001):
    """Real spectral envelope (rad per sub-pulse) with beta(f_fat) = 0 exactly.

    pb_hw: water passband half-width (Hz); sb_hw: stopband half-width about fat.
    """
    if n_sub % 2 == 0:
        raise ValueError('n_sub must be odd (symmetric, linear-phase envelope).')
    M = (n_sub - 1) // 2
    f = np.linspace(0, 0.5, n_grid)                   # cycles/sample; x 1/T_s = Hz
    fp = pb_hw * T_s
    fs = (abs(f_fat) - sb_hw) * T_s
    if fs <= fp:
        raise ValueError('Passband and fat stopband overlap for this T_s.')
    sc = np.r_[1, 2 * np.ones(M)]
    C = np.cos(2 * np.pi * np.outer(f, np.arange(M + 1))) * sc
    w = np.where((f <= fp) | (f >= fs), 1.0, 0.0)
    d = (f <= fp).astype(float)
    zrow = (np.cos(2 * np.pi * abs(f_fat) * T_s * np.arange(M + 1)) * sc)[None, :]
    K = np.block([[C.T @ (w[:, None] * C), zrow.T], [zrow, np.zeros((1, 1))]])
    x = np.linalg.solve(K, np.r_[C.T @ (w * d), 0.0])[:M + 1]
    h = np.r_[x[:0:-1], x]
    h /= h.sum()                                      # unit gain at water
    b = np.sin(flip_angle / 2) * h                    # SLR beta polynomial
    return np.real(sprf.b2rf(b))                      # imag part is ~1e-11 noise


def binomial_env(n_sub, flip_angle):
    """Binomial spectral envelope (rad per sub-pulse), e.g. 1-2-1 for n_sub=3.

    With f_fat * T_s = 1/2, fat precesses pi between sub-pulses, so the
    rotations alternate about +x/-x and cancel exactly at any flip angle;
    water rotations add to flip_angle exactly. Zero order n_sub-1 at fat.
    """
    w = np.array([1.0])
    for _ in range(n_sub - 1):
        w = np.convolve(w, [1.0, 1.0])
    return flip_angle * w / w.sum()


# ----------------------------------------------------------------------------
# RF
# ----------------------------------------------------------------------------
def _profile_fwhm(tbw, d1, d2, n=512, pad=64):
    """FWHM of the dzrf small-tip profile, in cycles across the pulse.

    SLR 'ls' designs at low TBW have FWHM < tbw (about 0.95x at tbw=2), so the
    gradient is scaled with this instead of tbw to hit the nominal thickness.
    """
    p = np.real(sprf.dzrf(n=n, tb=tbw, ptype='st', ftype='ls', d1=d1, d2=d2))
    P = np.abs(np.fft.fft(p, n * pad))
    f = np.fft.fftfreq(n * pad) * n
    above = f[P >= P.max() / 2]
    return above.max() - above.min()


def _k_of_t(gz, t):
    """k(t) = integral of gz from the start of gz (t relative to gz start)."""
    tt = np.linspace(0, gz.shape_dur, int(round(gz.shape_dur / 1e-7)) + 1)
    gg = np.interp(tt, gz.tt, gz.waveform)
    kk = np.concatenate([[0], np.cumsum(0.5 * (gg[1:] + gg[:-1]) * np.diff(tt))])
    return np.interp(t, tt, kk)


def make_spsp_rf(system, gz, geom, env, tbw, flip_angle, slice_offset=0.0,
                 d1=0.01, d2=0.01):
    """One arbitrary RF event spanning plateau 1 .. plateau N (zero on flybacks).

    RF timing is expressed relative to the gz event start; rf.delay is set later
    by design_spsp_exc once gz.delay is known.
    """
    dt_rf = system.rf_raster_time
    n_pts = int(round(geom.t_flat / dt_rf))
    p = np.real(sprf.dzrf(n=n_pts, tb=tbw, ptype='st', ftype='ls', d1=d1, d2=d2))
    p /= p.sum() * dt_rf * 2 * np.pi                  # unit area -> 1 rad

    t0 = geom.plateau_starts[0]
    t_end = geom.plateau_starts[-1] + geom.t_flat
    n_tot = int(round((t_end - t0) / dt_rf))
    signal = np.zeros(n_tot, dtype=complex)
    for n, w in enumerate(env):
        i0 = int(round((geom.plateau_starts[n] - t0) / dt_rf))
        signal[i0:i0 + n_pts] = w * p                 # sub-pulse n flips w rad

    t_c = geom.plateau_starts[len(env) // 2] + geom.t_flat / 2   # RF centre
    if slice_offset != 0.0:
        # gradient is not constant across the event -> phase-modulate with k(t)
        t_abs = t0 + (np.arange(n_tot) + 0.5) * dt_rf
        dk = _k_of_t(gz, t_abs) - _k_of_t(gz, np.array([t_c]))[0]
        signal *= np.exp(1j * 2 * np.pi * slice_offset * dk)

    rf = pp.make_arbitrary_rf(signal, flip_angle, system=system,
                              delay=system.rf_dead_time,   # overwritten by caller
                              no_signal_scaling=True, use='excitation')
    return rf, t0, t_c


def _verse_subpulse(geom, tbw, dt_rf, d1=0.01, d2=0.01, n_nom=512):
    """VERSE sub-pulse spanning one whole excitation lobe, unit area (1 rad).

    p0 is the dzrf pulse for a constant gradient, i.e. uniform in k. It is
    resampled at s(t) = k(t)/A_lobe and scaled by G(t) so the RF per unit k
    is unchanged: b(t) = p0(s(t)) * G(t).
    """
    tr, tf, G = geom.t_ramp, geom.t_flat, geom.amp
    n_pts = int(round(geom.lobe_dur / dt_rf))
    t = (np.arange(n_pts) + 0.5) * dt_rf
    knots_t = [0.0, tr, tr + tf, 2 * tr + tf]
    knots_g = [0.0, G, G, 0.0]
    g = np.interp(t, knots_t, knots_g)

    tt = np.linspace(0.0, geom.lobe_dur, 50 * n_pts + 1)     # fine grid for k(t)
    gg = np.interp(tt, knots_t, knots_g)
    kk = np.concatenate([[0], np.cumsum(0.5 * (gg[1:] + gg[:-1]) * np.diff(tt))])
    s = np.interp(t, tt, kk) / kk[-1]                         # 0 -> 1 across lobe

    p0 = np.real(sprf.dzrf(n=n_nom, tb=tbw, ptype='st', ftype='ls', d1=d1, d2=d2))
    s_nom = (np.arange(n_nom) + 0.5) / n_nom
    b = np.interp(s, s_nom, p0) * g
    return b / (b.sum() * dt_rf * 2 * np.pi)


def make_spsp_rf_verse(system, gz, geom, env, tbw, flip_angle, slice_offset=0.0,
                       d1=0.01, d2=0.01):
    """Binomial-mode RF: one arbitrary event, VERSE sub-pulse on every lobe.

    Same contract as make_spsp_rf (timing relative to gz start; caller sets
    rf.delay). Slice offset uses the same k(t) phase modulation, which is
    already correct for a gradient that varies under the RF.
    """
    dt_rf = system.rf_raster_time
    b = _verse_subpulse(geom, tbw, dt_rf, d1, d2)
    n_pts = b.size

    t0 = geom.lobe_starts[0]
    t_end = geom.lobe_starts[-1] + geom.lobe_dur
    n_tot = int(round((t_end - t0) / dt_rf))
    signal = np.zeros(n_tot, dtype=complex)
    for n, w in enumerate(env):
        i0 = int(round((geom.lobe_starts[n] - t0) / dt_rf))
        signal[i0:i0 + n_pts] = w * b

    t_c = 0.5 * (t0 + t_end)                          # symmetric train -> RF centre
    if slice_offset != 0.0:
        t_abs = t0 + (np.arange(n_tot) + 0.5) * dt_rf
        dk = _k_of_t(gz, t_abs) - _k_of_t(gz, np.array([t_c]))[0]
        signal *= np.exp(1j * 2 * np.pi * slice_offset * dk)

    rf = pp.make_arbitrary_rf(signal, flip_angle, system=system,
                              delay=system.rf_dead_time,   # overwritten by caller
                              no_signal_scaling=True, use='excitation')
    return rf, t0, t_c


# ----------------------------------------------------------------------------
# Top-level design
# ----------------------------------------------------------------------------
_MODE_DEFAULTS = {                    # tbw, n_sub, pb_hw, sb_hw
    'slr':      (2.0, 7, 100.0, 100.0),
    'binomial': (3.5, 3, 50.0, 50.0),   # pb/sb_hw only used by verify_spsp here
}


def design_spsp_exc(system, flip_angle, slice_thickness, mode='slr', tbw=None,
                    n_sub=None, fat_ppm=-3.4, pb_hw=None, sb_hw=None,
                    slice_offset=0.0, d1=0.01, d2=0.01):
    """SPSP water excitation with an exact null at the lipid resonance.

    mode='slr'      : least-squares/SLR envelope, plateau-only RF (default;
                      tbw=2, n_sub=7, pb_hw=sb_hw=100 Hz unless given).
    mode='binomial' : 1-2-1 envelope, k=0, VERSE RF over whole lobes
                      (tbw=3.5, n_sub=3 unless given).

    Returns
    -------
    rf, gz : PyPulseq events, delays already aligned; add as seq.add_block(rf, gz)
    info   : SimpleNamespace (T_s, env, rf_center from block start, timings, ...)
    """
    if mode not in _MODE_DEFAULTS:
        raise ValueError(f"mode must be 'slr' or 'binomial', got {mode!r}")
    d_tbw, d_nsub, d_pb, d_sb = _MODE_DEFAULTS[mode]
    tbw = d_tbw if tbw is None else tbw
    n_sub = d_nsub if n_sub is None else n_sub
    pb_hw = d_pb if pb_hw is None else pb_hw
    sb_hw = d_sb if sb_hw is None else sb_hw

    f_fat = fat_ppm * 1e-6 * system.gamma * system.B0          # Hz
    fwhm_tbw = _profile_fwhm(tbw, d1, d2)
    gz, geom = make_spsp_grad(system, slice_thickness, fwhm_tbw, n_sub, f_fat,
                              mode=mode)
    if mode == 'binomial':
        env = binomial_env(n_sub, flip_angle)
        rf, t0, t_c = make_spsp_rf_verse(system, gz, geom, env, tbw, flip_angle,
                                         slice_offset, d1, d2)
    else:
        env = design_spectral_env(n_sub, geom.T_s, flip_angle, f_fat, pb_hw, sb_hw)
        rf, t0, t_c = make_spsp_rf(system, gz, geom, env, tbw, flip_angle,
                                   slice_offset, d1, d2)

    # align: RF must start >= rf_dead_time after block start
    dt = system.grad_raster_time
    gz.delay = _ceil_to(max(0.0, system.rf_dead_time - t0), dt)
    rf.delay = gz.delay + t0

    info = SimpleNamespace(
        mode=mode, k=geom.k, lobe_dur=geom.lobe_dur, A_lobe=geom.A_tot,
        f_fat=f_fat, T_s=geom.T_s, fat_cycles_per_Ts=f_fat * geom.T_s,
        n_sub=n_sub, tbw=tbw, env=env, flip_angle=flip_angle,
        slice_thickness=slice_thickness, slice_offset=slice_offset,
        rf_center=gz.delay + t_c,                              # from block start
        t_flat=geom.t_flat, t_ramp=geom.t_ramp, g_plateau=geom.amp,
        fb_ramp=geom.fb_ramp, fb_flat=geom.fb_flat, fb_amp=geom.fb_amp,
        block_duration=pp.calc_duration(rf, gz),
        peak_b1_uT=np.max(np.abs(rf.signal)) / system.gamma * 1e6,
        pb_hw=pb_hw, sb_hw=sb_hw,
    )
    return rf, gz, info


# ----------------------------------------------------------------------------
# Profile-matched single sub-pulse (non-SPSP reference excitation)
# ----------------------------------------------------------------------------
def design_matched_subpulse(system, flip_angle, slice_thickness, spsp_info,
                            slice_offset=0.0, d1=0.01, d2=0.01, verse=False):
    """ONE copy of the SPSP spatial sub-pulse, at the full flip angle.

    Same dzrf design (tbw, d1, d2), same plateau duration and gradient
    amplitude as the SPSP lobes, so in the small-tip regime its slice profile
    is identical to the SPSP's water profile -- but with no spectral selectivity.
    Use as the non-lipid-suppressed reference to isolate the effect of the
    lipid null from the slice-profile shape.

    RF is played on the plateau only (as in the SPSP). Slice offset is applied
    as RF phase modulation about the RF centre (constant gradient under RF).

    Returns rf, gz, gzr in the same form as pp.make_sinc_pulse(return_gz=True):
    gz is the slice-select trapezoid, gzr the slice refocus trapezoid.

    For spsp_info.mode == 'binomial' the SPSP sub-pulses are VERSE'd onto
    their ramps, but the reference does not need to be: the small-tip profile
    depends only on RF per unit k and the total area. By default it is a
    plateau-only dzrf sub-pulse (same tbw, d1, d2 and area A_lobe) whose
    plateau equals the SPSP lobe duration, built with `system`'s limits
    (lower slew/peak B1 than the VERSE lobe). verse=True instead plays the
    SPSP's own VERSE lobe; pass the excitation limits as `system` then.
    """
    binomial = getattr(spsp_info, 'mode', 'slr') == 'binomial'
    if binomial and verse:
        return _matched_subpulse_verse(system, flip_angle, spsp_info,
                                       slice_offset, d1, d2)
    dt = system.grad_raster_time
    dt_rf = system.rf_raster_time
    t_flat = spsp_info.lobe_dur if binomial else spsp_info.t_flat
    tbw = spsp_info.tbw

    fwhm_tbw = _profile_fwhm(tbw, d1, d2)
    A_flat = fwhm_tbw / slice_thickness                     # 1/m, same as SPSP lobe
    gz = pp.make_trapezoid('z', flat_area=A_flat, flat_time=t_flat, system=system)

    n_pts = int(round(t_flat / dt_rf))
    p = np.real(sprf.dzrf(n=n_pts, tb=tbw, ptype='st', ftype='ls', d1=d1, d2=d2))
    p /= p.sum() * dt_rf * 2 * np.pi                        # unit area -> 1 rad
    signal = flip_angle * p.astype(complex)
    if slice_offset != 0.0:
        t_rel = (np.arange(n_pts) + 0.5) * dt_rf - t_flat / 2   # about RF centre
        signal *= np.exp(1j * 2 * np.pi * gz.amplitude * slice_offset * t_rel)

    # RF must start >= rf_dead_time after block start.
    # no_signal_scaling: pypulseq would divide by |sum(signal)|, which the
    # offset phase ramp shrinks -> wrong flip off-centre.
    gz.delay = _ceil_to(max(0.0, system.rf_dead_time - gz.rise_time), dt)
    rf = pp.make_arbitrary_rf(signal, flip_angle, system=system,
                              delay=gz.delay + gz.rise_time,
                              no_signal_scaling=True, use='excitation')

    # Refocus: -(area from RF centre to end of slice select) = -gz.area/2
    gzr = pp.make_trapezoid('z', area=-gz.area / 2, system=system)
    return rf, gz, gzr


def _matched_subpulse_verse(system, flip_angle, spsp_info, slice_offset=0.0,
                            d1=0.01, d2=0.01):
    """Binomial-mode reference: one VERSE sub-pulse at the full flip angle.

    Gradient is the identical lobe (same ramps/plateau/amplitude) as the SPSP
    train. Peak B1 is 2x the SPSP centre sub-pulse (flip vs flip/2).
    """
    dt = system.grad_raster_time
    dt_rf = system.rf_raster_time
    tr, tf, G = spsp_info.t_ramp, spsp_info.t_flat, spsp_info.g_plateau
    geom = SimpleNamespace(t_ramp=tr, t_flat=tf, amp=G, lobe_dur=spsp_info.lobe_dur)

    if tf == 0:
        times, amps = [0.0, tr, 2 * tr], [0.0, G, 0.0]
    else:
        times, amps = [0.0, tr, tr + tf, 2 * tr + tf], [0.0, G, G, 0.0]
    gz = pp.make_extended_trapezoid('z', amplitudes=np.array(amps),
                                    times=_round_to(np.array(times), dt),
                                    system=system)

    b = _verse_subpulse(geom, spsp_info.tbw, dt_rf, d1, d2)
    signal = flip_angle * b.astype(complex)           # b is 1 rad -> flip_angle rad
    if slice_offset != 0.0:
        t = (np.arange(b.size) + 0.5) * dt_rf
        k = _k_of_t(gz, t)
        dk = k - _k_of_t(gz, np.array([spsp_info.lobe_dur / 2]))[0]
        signal *= np.exp(1j * 2 * np.pi * slice_offset * dk)

    # RF starts with the lobe; both must start >= rf_dead_time.
    # no_signal_scaling: pypulseq would divide by |sum(signal)|, which the
    # offset phase ramp shrinks -> wrong flip off-centre.
    gz.delay = _ceil_to(system.rf_dead_time, dt)
    rf = pp.make_arbitrary_rf(signal, flip_angle, system=system,
                              delay=gz.delay, no_signal_scaling=True,
                              use='excitation')

    area = G * (tr + tf)
    gzr = pp.make_trapezoid('z', area=-area / 2, system=system)
    return rf, gz, gzr


# ----------------------------------------------------------------------------
# Bloch simulation (spinor, rotating frame, no relaxation)
# ----------------------------------------------------------------------------
def bloch_sim(rf, gz, z, df, dt=1e-6, grad_delay=0.0, b1_scale=1.0):
    """Transverse magnetisation after the block, starting from Mz = 1.

    z, df broadcast against each other (m, Hz). grad_delay shifts the actual
    gradient later than nominal (s), to test RF/gradient timing errors.
    """
    z, df = np.broadcast_arrays(np.asarray(z, float), np.asarray(df, float))
    T = pp.calc_duration(rf, gz) + abs(grad_delay)
    t = (np.arange(int(round(T / dt))) + 0.5) * dt
    G = np.interp(t, gz.delay + grad_delay + gz.tt, gz.waveform, left=0, right=0)
    B = b1_scale * (np.interp(t, rf.delay + rf.t, rf.signal.real, left=0, right=0)
                    + 1j * np.interp(t, rf.delay + rf.t, rf.signal.imag, left=0, right=0))
    a = np.ones(z.shape, complex)
    b = np.zeros(z.shape, complex)
    for Gi, Bi in zip(G, B):
        w = 2 * np.pi * (Gi * z + df) * dt           # rad about z
        b1 = 2 * np.pi * Bi * dt                      # rad, complex (phase = axis)
        phi = np.sqrt(w ** 2 + abs(b1) ** 2)
        s = np.sinc(phi / (2 * np.pi)) * 0.5          # sin(phi/2)/phi, safe at 0
        al = np.cos(phi / 2) - 1j * w * s
        be = -1j * b1 * s
        a, b = al * a - np.conj(be) * b, be * a + np.conj(al) * b
    return 2 * np.conj(a) * b


def _flip_deg(mxy):
    return np.rad2deg(np.arcsin(np.clip(np.abs(mxy), 0, 1)))


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
def verify_spsp(rf, gz, info, delays=(-4e-6, -2e-6, 2e-6, 4e-6),
                b1_scales=(0.9, 1.1), plot=True, dt=1e-6):
    """Bloch-sim checks of the designed pulse. Prints a report, returns metrics."""
    thk, z0, ff = info.slice_thickness, info.slice_offset, info.f_fat
    fa = np.rad2deg(info.flip_angle)
    zc = z0 + np.linspace(-1.5 * thk, 1.5 * thk, 241)
    in_slice = np.abs(zc - z0) <= thk / 2
    m = {}

    # 1) slice profile at water and at fat
    p_w = _flip_deg(bloch_sim(rf, gz, zc, 0.0, dt))
    p_f = _flip_deg(bloch_sim(rf, gz, zc, ff, dt))
    half = zc[p_w >= p_w.max() / 2]
    m['water_center_deg'] = float(_flip_deg(bloch_sim(rf, gz, z0, 0.0, dt)))
    m['fwhm_mm'] = float((half.max() - half.min()) * 1e3)
    m['slice_center_mm'] = float(0.5 * (half.max() + half.min()) * 1e3)
    m['fat_max_in_slice_deg'] = float(p_f[in_slice].max())

    # 2) spectral profile at slice centre
    fr = np.linspace(-1000, 1000, 801)
    s_c = _flip_deg(bloch_sim(rf, gz, z0, fr, dt))
    pb = np.abs(fr) <= info.pb_hw
    sb = np.abs(fr - ff) <= info.sb_hw
    m['passband_min_max_deg'] = (float(s_c[pb].min()), float(s_c[pb].max()))
    m['fat_band_max_deg'] = float(s_c[sb].max())

    # 3) water/fat phase across the slice (refocus check)
    ph = np.angle(bloch_sim(rf, gz, zc[in_slice], 0.0, dt), deg=True)
    m['water_phase_spread_deg'] = float(np.ptp(np.unwrap(np.deg2rad(ph))) * 180 / np.pi)

    # 4) robustness: RF/gradient delay and B1 scaling -> fat leak in slice
    zi = zc[in_slice][::4]
    rob = {}
    for d in delays:
        rob[f'delay {d*1e6:+.0f} us'] = (
            float(_flip_deg(bloch_sim(rf, gz, zi, 0.0, dt, grad_delay=d)).max()),
            float(_flip_deg(bloch_sim(rf, gz, zi, ff, dt, grad_delay=d)).max()))
    for s in b1_scales:
        rob[f'B1 x{s:.2f}'] = (
            float(_flip_deg(bloch_sim(rf, gz, zi, 0.0, dt, b1_scale=s)).max()),
            float(_flip_deg(bloch_sim(rf, gz, zi, ff, dt, b1_scale=s)).max()))
    m['robustness'] = rob

    print(f'--- SPSP verification ({fa:.1f} deg, {thk*1e3:.1f} mm, '
          f'fat {ff:.1f} Hz, T_s {info.T_s*1e3:.2f} ms) ---')
    print(f'water flip @ centre      : {m["water_center_deg"]:.3f} deg')
    print(f'FWHM / centre            : {m["fwhm_mm"]:.2f} mm @ {m["slice_center_mm"]:+.2f} mm')
    print(f'water phase across slice : {m["water_phase_spread_deg"]:.2f} deg (peak-peak)')
    print(f'passband +/-{info.pb_hw:.0f} Hz       : '
          f'{m["passband_min_max_deg"][0]:.2f} - {m["passband_min_max_deg"][1]:.2f} deg')
    print(f'fat max in slice         : {m["fat_max_in_slice_deg"]:.2e} deg')
    print(f'fat +/-{info.sb_hw:.0f} Hz max (ctr)  : {m["fat_band_max_deg"]:.3f} deg')
    for k_, (w_, f_) in rob.items():
        print(f'{k_:<25}: water max {w_:.2f} deg, fat max {f_:.3f} deg')

    if plot:
        import matplotlib.pyplot as plt
        zz = np.linspace(z0 - 1.5 * thk, z0 + 1.5 * thk, 121)
        ffm = np.linspace(-1000, 1000, 161)
        mp = _flip_deg(bloch_sim(rf, gz, zz[:, None], ffm[None, :], dt))
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(zc * 1e3, p_w, label='water')
        ax[0].plot(zc * 1e3, p_f, label=f'fat ({ff:.0f} Hz)')
        ax[0].set(xlabel='z (mm)', ylabel='flip (deg)', title='Slice profile')
        ax[0].legend()
        ax[1].plot(fr, s_c)
        ax[1].axvline(ff, ls='--', c='k', lw=0.8)
        ax[1].set(xlabel='off-resonance (Hz)', ylabel='flip (deg)',
                  title='Spectral profile @ slice centre')
        im = ax[2].imshow(mp, aspect='auto', origin='lower',
                          extent=[ffm[0], ffm[-1], zz[0] * 1e3, zz[-1] * 1e3])
        ax[2].axvline(ff, ls='--', c='w', lw=0.8)
        ax[2].set(xlabel='off-resonance (Hz)', ylabel='z (mm)', title='|flip| (deg)')
        fig.colorbar(im, ax=ax[2])
        fig.tight_layout()
        m['figure'] = fig
    return m


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=130, slew_unit='mT/m/ms',
                     rf_ringdown_time=60e-6, rf_dead_time=100e-6,
                     adc_dead_time=20e-6, adc_raster_time=10e-6, B0=3.00)

    MODE = 'slr'                                     # 'slr' or 'binomial'
    rf_exc, gz_exc, info = design_spsp_exc(system, np.deg2rad(16.5), 8e-3, mode=MODE)
    print(f'T_s {info.T_s*1e3:.2f} ms (fat*T_s = {info.fat_cycles_per_Ts:.4f}), '
          f'plateau {info.t_flat*1e3:.2f} ms @ {info.g_plateau/system.gamma*1e3:.1f} mT/m, '
          f'ramps {info.t_ramp*1e3:.2f} ms, flyback {2*info.fb_ramp*1e3+info.fb_flat*1e3:.2f} ms')
    print(f'envelope (deg): {np.round(np.rad2deg(info.env), 3)}')
    print(f'block {info.block_duration*1e3:.2f} ms, RF centre {info.rf_center*1e3:.3f} ms, '
          f'peak B1 {info.peak_b1_uT:.2f} uT')

    seq = pp.Sequence(system)
    seq.add_block(rf_exc, gz_exc)
    ok, err = seq.check_timing()
    print('timing ok:', ok)
    seq.plot()

    verify_spsp(rf_exc, gz_exc, info, plot=True)