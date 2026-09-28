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

Public API
----------
design_spsp_exc(system, flip_angle, slice_thickness, ...) -> (rf, gz, info)
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
                               fb_ramp=fb[0], fb_flat=fb[1], fb_amp=fb[2])
    return best


def make_spsp_grad(system, slice_thickness, fwhm_tbw, n_sub, f_fat, k_max=6):
    """Flyback SPSP gradient as a single extended trapezoid.

    fwhm_tbw: FWHM of the sub-pulse profile in cycles (from _profile_fwhm), so the
    plateau area makes the *measured* FWHM equal slice_thickness.
    Returns (gz, geom) where geom holds T_s and lobe timings.
    """
    dt = system.grad_raster_time
    A_flat = fwhm_tbw / slice_thickness               # 1/m under the RF plateau

    for k in range(k_max + 1):                        # fat midway between replicas
        T_s = _round_to((k + 0.5) / abs(f_fat), dt)
        lobe = _exc_lobe_geometry(system, A_flat, T_s)
        if lobe is not None:
            break
    else:
        raise ValueError('No flyback geometry fits; lower tbw or raise limits.')

    tr, tf, G = lobe.t_ramp, lobe.t_flat, lobe.amp
    times, amps, t = [0.0], [0.0], 0.0
    plateau_starts = []
    for n in range(n_sub):
        plateau_starts.append(t + tr)
        times += [t + tr, t + tr + tf, t + 2 * tr + tf]
        amps += [G, G, 0.0]
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
                           **vars(lobe))
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


# ----------------------------------------------------------------------------
# Top-level design
# ----------------------------------------------------------------------------
def design_spsp_exc(system, flip_angle, slice_thickness, tbw=2.0, n_sub=7,
                    fat_ppm=-3.4, pb_hw=100.0, sb_hw=100.0, slice_offset=0.0,
                    d1=0.01, d2=0.01):
    """SPSP water excitation with an exact null at the lipid resonance.

    Returns
    -------
    rf, gz : PyPulseq events, delays already aligned; add as seq.add_block(rf, gz)
    info   : SimpleNamespace (T_s, env, rf_center from block start, timings, ...)
    """
    f_fat = fat_ppm * 1e-6 * system.gamma * system.B0          # Hz
    fwhm_tbw = _profile_fwhm(tbw, d1, d2)
    gz, geom = make_spsp_grad(system, slice_thickness, fwhm_tbw, n_sub, f_fat)
    env = design_spectral_env(n_sub, geom.T_s, flip_angle, f_fat, pb_hw, sb_hw)
    rf, t0, t_c = make_spsp_rf(system, gz, geom, env, tbw, flip_angle, slice_offset,
                               d1, d2)

    # align: RF must start >= rf_dead_time after block start
    dt = system.grad_raster_time
    gz.delay = _ceil_to(max(0.0, system.rf_dead_time - t0), dt)
    rf.delay = gz.delay + t0

    info = SimpleNamespace(
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

    rf_exc, gz_exc, info = design_spsp_exc(system, np.deg2rad(16.5), 8e-3)
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