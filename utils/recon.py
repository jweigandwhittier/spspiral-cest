#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:59:04 2026

@author: jonah
"""
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch
import torchkbnufft as tkbn
import matplotlib.pyplot as plt
from packaging.version import Version

def _adjoint_nufft_core(ksp, kx_full, ky_full, nx, fov):
    """
    Shared adjoint-NUFFT + DCF + RSS coil combination, given kx/ky already
    shaped as [n_trs, n_adc] and correctly rotated per-TR.

    Parameters
    ----------
    ksp: k-space [n_adc, n_trs, n_coils]
    kx_full, ky_full: [n_trs, n_adc] k-space coordinates, same units as
        traj from seq.calculate_kspace (1/m, prior to pixel_size scaling)
    nx: Nominal matrix size (isotropic), from sequence definitions
    fov: FOV from sequence definitions

    Returns
    -------
    combined: Image with RSS coil combination
    """
    nx = int(nx)
    n_coils = ksp.shape[2]
    pixel_size = fov / nx
    im_size = [nx, nx]
    omega = np.stack([kx_full.ravel(), ky_full.ravel()]) * pixel_size * 2 * np.pi
    omega_t = torch.from_numpy(omega).to(torch.float32).unsqueeze(0)
    data_flat = ksp.transpose(2, 1, 0).reshape(n_coils, -1)
    sig_t = torch.from_numpy(data_flat).unsqueeze(0).to(torch.complex64)
    dcomp = tkbn.calc_density_compensation_function(omega_t, im_size)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)
    img = adjnufft_ob(sig_t * dcomp, omega_t)
    img = img[0].cpu().numpy()
    combined = np.sqrt(np.sum(np.abs(img) ** 2, axis=0))
    combined = np.rot90(combined, k=3)
    return combined


def adjoint_nufft(ksp, traj, nx, fov, tr_offset=0):
    """
    Simple adjoint NUFFT for VDS acquisition with DCF. Can be used for phantom data with
    no cardiac motion.

    Assumes `traj` (from seq.calculate_kspace) already contains correctly
    rotated per-TR trajectory samples in chronological order -- do NOT use
    this on a trajectory where calculate_kspace() dropped the rotation
    extension (see adjoint_nufft_from_traj for that case).

    Parameters
    ----------
    ksp: k-space from full acquisition like [n_samples, n_trs, n_coils]
    traj: Trajectory for full sequences directly from Pypulseq (seq.calculate_kspace)
    nx: Nominal matrix size (isotropic), from sequence definitions
    fov: FOV from sequence definitions

    Returns
    -------
    combined: Image with RSS coil combination
    """
    n_adc = ksp.shape[0]
    n_trs = ksp.shape[1]
    total_samples = n_adc * n_trs
    start = tr_offset * n_adc
    end = start + total_samples
    kx_full = traj[0, start:end].reshape(n_trs, n_adc)
    ky_full = traj[1, start:end].reshape(n_trs, n_adc)
    return _adjoint_nufft_core(ksp, kx_full, ky_full, nx, fov)

def seq_uses_rotation_extension(seq):
    """
    calculate_kspace() ignores the rotext ROTATION extension entirely --
    it returns the same unrotated reference arm for every TR when a
    sequence was written using pp.make_rotation() (PyPulseq >=1.5.0 rotext
    fork). Sequences written with plain PyPulseq 1.4.2 instead bake the
    per-TR rotation directly into each block's gradient waveform, so
    calculate_kspace() already returns correctly rotated per-TR trajectory.

    Detected from the Pulseq spec version recorded in the .seq file itself
    (mirrors the same 1.5.0 threshold continuous_spiral.py uses to decide
    USE_ROTATION at write time) -- NOT the currently-installed PyPulseq
    version, which may differ from whatever wrote this particular file.
    """
    version = Version(f"{seq.version_major}.{seq.version_minor}.{seq.version_revision}")
    return version >= Version("1.5.0")


def get_rotated_trajectory(seq, k_traj_adc, n_adc, n_trs, pislquant):
    """
    Returns kx_full, ky_full as [n_trs, n_adc], correctly rotated per-TR,
    for the n_trs MAIN TRs only (calibration TRs already excluded from the
    output, though still accounted for via `pislquant` when indexing into
    k_traj_adc).

    NOTE: reads seq.definitions['RotationAngles_rad'], which is only set by
    build_and_save_single_offset_seq() -- i.e. the non-ZSPEC offsets_ppm loop
    and the per-offset GE ZSPEC branch. The combined-Siemens-ZSPEC sequence
    (one .seq sweeping all zspec_offsets_ppm) does NOT set this definition;
    reconstructing that file needs angles rebuilt from
    golden_angle * np.arange(n_trs) instead, and this function will raise
    a KeyError as a signal to do that rather than silently misbehaving.
    """
    start = pislquant * n_adc  # skip calibration TRs (present in k_traj_adc, no dead TR here)

    if seq_uses_rotation_extension(seq):
        angles = np.array(seq.definitions['RotationAngles_rad'])
        assert len(angles) == n_trs, \
            f"RotationAngles_rad length {len(angles)} != N_TRs {n_trs}"

        kx_ref = k_traj_adc[0, :n_adc]
        ky_ref = k_traj_adc[1, :n_adc]

        c = np.cos(angles)[:, None]
        s = np.sin(angles)[:, None]
        kx_full = kx_ref[None, :] * c - ky_ref[None, :] * s
        ky_full = kx_ref[None, :] * s + ky_ref[None, :] * c
    else:
        total_samples = n_adc * n_trs
        kx_full = k_traj_adc[0, start:start + total_samples].reshape(n_trs, n_adc)
        ky_full = k_traj_adc[1, start:start + total_samples].reshape(n_trs, n_adc)

    return kx_full, ky_full


def adjoint_nufft_from_traj(ksp, kx_full, ky_full, nx, fov):
    """
    Adjoint NUFFT variant that takes kx/ky trajectory arrays directly,
    already shaped [n_trs, n_adc] and rotated per-TR -- for use when
    seq.calculate_kspace() can't be trusted to apply per-TR rotation
    (see seq_uses_rotation_extension / get_rotated_trajectory).

    Parameters
    ----------
    ksp: k-space from full acquisition like [n_samples, n_trs, n_coils]
    kx_full, ky_full: [n_trs, n_adc] per-TR rotated trajectory
    nx: Nominal matrix size (isotropic), from sequence definitions
    fov: FOV from sequence definitions

    Returns
    -------
    combined: Image with RSS coil combination
    """
    n_trs = ksp.shape[1]
    assert kx_full.shape == (n_trs, ksp.shape[0]), \
        f"kx_full shape {kx_full.shape} doesn't match ksp (n_trs={n_trs}, n_adc={ksp.shape[0]})"
    return _adjoint_nufft_core(ksp, kx_full, ky_full, nx, fov)

def estimate_burnin(nav_mag, tol=0.02, min_burnin=5, max_burnin=None, plot=False):
    """Fit an exponential approach-to-steady-state curve to the mean
    (across-coil) navigator magnitude and return the number of TRs needed to
    settle within `tol` fractional amplitude of the asymptote.

    Model: y(n) = C + A*exp(-n/tau), fit by nonlinear least squares. Solving
    |A|*exp(-n/tau) = tol*|A| for n gives n_burnin = tau * ln(1/tol).

    Falls back to (None, None) on a bad fit (non-physical tau, non-convergence)
    so the caller can substitute a fixed default rather than silently using a
    garbage cutoff.
    """
    from scipy.optimize import curve_fit
    y = nav_mag.mean(axis=1)
    n = np.arange(len(y))

    def model(n, A, tau, C):
        return C + A * np.exp(-n / tau)

    p0 = [y[0] - y[-1], max(len(y) / 10, 5), y[-1]]
    try:
        popt, _ = curve_fit(model, n, y, p0=p0, maxfev=5000)
        A_fit, tau_fit, C_fit = popt
        if not np.isfinite(tau_fit) or tau_fit <= 0:
            raise RuntimeError(f"non-physical tau from fit: {tau_fit}")
    except Exception as e:
        print(f"exponential burn-in fit failed ({e}); falling back to a fixed default")
        return None, None

    n_est = int(np.ceil(tau_fit * np.log(1 / tol)))
    n_est = int(np.clip(n_est, min_burnin, max_burnin or len(y) // 2))
    print(f"exponential burn-in fit: A={A_fit:.4g}, tau={tau_fit:.2f} TRs, "
          f"C={C_fit:.4g} -> n_burnin={n_est} (settles within {tol*100:.0f}% "
          f"of asymptote)")

    if plot:
        plt.figure(figsize=(8, 3))
        plt.plot(n, y, label='mean nav magnitude (all coils)')
        plt.plot(n, model(n, *popt), label='exponential fit', lw=1.5)
        plt.axvline(n_est, color='r', ls='--', label=f'n_burnin = {n_est}')
        plt.legend()
        plt.title('steady-state approach: exponential fit')

    return n_est, popt

def apply_axis_delay_correction(t, k_logical, R, tau_vec):
    """Shift a logical-axis k-space trajectory by per-physical-axis gradient
    delays. k(t)=gamma*cumtrapz(G(t)) is linear in G, so a pure gradient
    time-shift gives the identical shift in k-space: k_actual(t) =
    k_nominal(t-tau). Shift-and-interpolate per physical axis, then rotate
    back to logical -- no re-differentiation needed.

    t: [n_t] real per-ADC-sample timepoints (not a synthetic dwell*index base
       -- must come from calculate_kspace's returned t_adc to correctly
       account for gaps between TRs in a continuously-acquired sequence).
    k_logical: [3, n_t] trajectory in logical (read, phase, slice) coords.
    R: logical -> physical (DCS) rotation, from twixtools' Geometry class.
    """
    k_phys = R @ k_logical
    k_phys_corr = np.zeros_like(k_phys)
    for ax in range(3):
        k_phys_corr[ax] = np.interp(t - tau_vec[ax], t, k_phys[ax],
                                     left=0.0, right=k_phys[ax, -1])
    return R.T @ k_phys_corr

def estimate_coil_sensitivities_espirit(kx_all, ky_all, all_data, im_size, pixel_size, n_coils,
                                         k_cutoff_frac=0.25, calib_width=24, thresh=0.02,
                                         kernel_width=6, crop=0.95, device=-1, plot=False):
    """Coil sensitivity maps via sigpy's ESPIRiT calibration (Uecker et al.
    2014). Same low-k selection/gridding as estimate_coil_sensitivities
    (adjoint NUFFT over the k_cutoff_frac-limited region), but instead of
    smoothing the gridded images and RSS-normalizing, FFTs them to a
    synthetic Cartesian k-space and lets EspiritCalib do the calibration --
    generally much better behaved than RSS-normalized low-pass images near
    coil nulls / low-SNR regions.

    device: sigpy Device index, -1 for CPU, 0 (or other) for GPU via cupy.
    """
    kx_flat = kx_all.ravel()
    ky_flat = ky_all.ravel()
    kmax = max(np.max(np.abs(kx_flat)), np.max(np.abs(ky_flat)))
    k_cutoff = k_cutoff_frac * kmax
    mask = (kx_flat**2 + ky_flat**2) <= k_cutoff**2
    print(f"coil sensitivity calibration: {mask.sum()}/{mask.size} samples "
          f"within k_cutoff={k_cutoff:.1f} (frac={k_cutoff_frac})")

    data_flat = all_data.transpose(1, 0, 2).reshape(n_coils, -1)
    omega = np.stack([kx_flat[mask], ky_flat[mask]]) * pixel_size * 2 * np.pi
    omega_t = torch.from_numpy(omega).to(torch.float32)
    sig_t = torch.from_numpy(data_flat[:, mask]).unsqueeze(0).to(torch.complex64)

    dcomp = tkbn.calc_density_compensation_function(omega_t, im_size)
    nufft_adj = tkbn.KbNufftAdjoint(im_size=im_size)
    coil_imgs = nufft_adj(sig_t * dcomp, omega_t)[0].cpu().numpy()  # [n_coils, nx, nx]

    # Re-grid to a synthetic Cartesian calibration k-space -- EspiritCalib
    # expects k-space input, not images.
    ksp_cart = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(coil_imgs, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1)
    ).astype(np.complex64)

    espirit = mr.app.EspiritCalib(
        ksp_cart, calib_width=calib_width, thresh=thresh,
        kernel_width=kernel_width, crop=crop, device=sp.Device(device),
        show_pbar=False
    )
    smaps = espirit.run()  # same shape as ksp_cart: [n_coils, nx, nx]
    if plot:
        n_coils = smaps.shape[0]
        ncols = int(np.ceil(np.sqrt(n_coils)))
        nrows = int(np.ceil(n_coils / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
        axes = np.atleast_2d(axes)

        for c in range(n_coils):
            r, col = divmod(c, ncols)
            ax = axes[r, col]
            ax.imshow(np.abs(smaps[c]), cmap='gray')
            ax.set_title(f"coil {c}", fontsize=8)

        for ax in axes.flatten():
            ax.axis('off')

        plt.suptitle("Coil sensitivity maps")
        plt.tight_layout()
        plt.show()
    return smaps.astype(np.complex64)

def build_binned_indices(valid_phase, n_samples, n_bins, bin_centers):
    """Strict per-readout cardiac binning: each whole readout is assigned to
    its single nearest bin by cardiac phase.
    Returns dict of bin -> list of (readout_idx, sample_mask), sample_mask
    always all-True, so downstream gather_bin_samples/recon code is unchanged.
    """
    binned = {b: [] for b in range(n_bins)}
    full_mask = np.ones(n_samples, dtype=bool)
    for i, phase in enumerate(valid_phase):
        angular_dist = np.angle(np.exp(1j * (phase - bin_centers)))
        b = int(np.argmin(np.abs(angular_dist)))
        binned[b].append((i, full_mask))
    return binned

# ---------------------------------------------------------------------------
# XD-GRASP-style joint reconstruction: per-bin data consistency, coupled
# across cardiac phases by spatial + temporal total-variation regularization
# instead of independent per-bin solves. Data consistency stays exactly the
# per-bin problem CG-SENSE solves (same A, AH, dcomp, smaps); the only
# addition is a joint FISTA loop with two proximal steps.
# ---------------------------------------------------------------------------

def _grad1d_wrap(u):
    return torch.roll(u, -1, dims=0) - u


def _div1d_wrap(p):
    return p - torch.roll(p, 1, dims=0)


def _grad1d_nowrap(u):
    return u[1:] - u[:-1]


def _div1d_nowrap(p):
    d = torch.zeros((p.shape[0] + 1,) + p.shape[1:], dtype=p.dtype, device=p.device)
    d[:-1] -= p
    d[1:] += p
    return d


def chambolle_tv_prox_1d(y, weight, n_iter=10, tau=0.24, wrap=True):
    """Exact (in the n_iter -> inf limit) proximal operator of
    weight * sum_i |y_{i+1} - y_i| along dim 0, via Chambolle's dual
    projected-gradient algorithm. y: real tensor [n_bins, ...]; TV couples
    only along dim 0, independently per remaining (pixel) dimensions.
    wrap=True treats dim 0 as cyclic (cardiac phase 8 borders phase 1).
    """
    if weight <= 0:
        return y
    grad_fn = _grad1d_wrap if wrap else _grad1d_nowrap
    div_fn = _div1d_wrap if wrap else _div1d_nowrap
    p_shape = y.shape if wrap else (y.shape[0] - 1,) + y.shape[1:]
    p = torch.zeros(p_shape, dtype=y.dtype, device=y.device)
    for _ in range(n_iter):
        div_p = div_fn(p)
        p = p + tau * grad_fn(div_p - y / weight)
        p = p / p.abs().clamp(min=1.0)
    return y - weight * div_fn(p)


def _grad2d(u):
    # u: [..., H, W] -> [..., H, W, 2], forward difference, zero at the last row/col
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gx[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    gy[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    return torch.stack([gx, gy], dim=-1)


def _div2d(p):
    # p: [..., H, W, 2] -> [..., H, W]
    px, py = p[..., 0], p[..., 1]
    dx = torch.zeros_like(px)
    dx[..., 1:, :] = px[..., 1:, :] - px[..., :-1, :]
    dx[..., 0, :] = px[..., 0, :]
    dy = torch.zeros_like(py)
    dy[..., :, 1:] = py[..., :, 1:] - py[..., :, :-1]
    dy[..., :, 0] = py[..., :, 0]
    return dx + dy


def chambolle_tv_prox_2d(y, weight, n_iter=10, tau=0.125):
    """Proximal operator of weight * isotropic-TV(y) over the trailing two
    (spatial) dimensions, via Chambolle's 2004 dual projected-gradient
    algorithm. y: real tensor [..., H, W], batched over any leading dims
    (e.g. cardiac bin) -- each leading-dim slice is denoised independently.
    """
    if weight <= 0:
        return y
    p = torch.zeros(y.shape + (2,), dtype=y.dtype, device=y.device)
    for _ in range(n_iter):
        div_p = _div2d(p)
        p = p + tau * _grad2d(div_p - y / weight)
        norm_p = torch.sqrt((p**2).sum(-1, keepdim=True)).clamp(min=1.0)
        p = p / norm_p
    return y - weight * _div2d(p)


def prox_tv_spatial(x_complex, weight, n_iter=10):
    """Spatial TV prox on a complex [n_bins, H, W] image stack. TV isn't
    naturally defined for complex values; this applies the real-valued prox
    to the real and imaginary channels independently, which is the standard
    cheap approximation used in most compressed-sensing MRI implementations
    (as opposed to a magnitude/phase-coupled TV, which is more involved and
    not implemented here).
    """
    xr = chambolle_tv_prox_2d(x_complex.real, weight, n_iter=n_iter)
    xi = chambolle_tv_prox_2d(x_complex.imag, weight, n_iter=n_iter)
    return torch.complex(xr, xi)


def prox_tv_temporal(x_complex, weight, n_iter=10, wrap=True):
    """Temporal TV prox along dim 0 (cardiac bin) of a complex
    [n_bins, H, W] image stack, real/imaginary channels independently."""
    xr = chambolle_tv_prox_1d(x_complex.real, weight, n_iter=n_iter, wrap=wrap)
    xi = chambolle_tv_prox_1d(x_complex.imag, weight, n_iter=n_iter, wrap=wrap)
    return torch.complex(xr, xi)

def gather_bin_samples(valid_data, kx_all, ky_all, entries, n_coils):
    """Concatenate the (possibly partial) readouts contributing to one
    cardiac bin. `entries` is the (readout_idx, sample_mask) list produced by
    `build_keyhole_binned_indices`.
    """
    sig_parts, kx_parts, ky_parts = [], [], []
    for i, mask in entries:
        sig_parts.append(valid_data[i][:, mask])   # [n_coils, n_masked]
        kx_parts.append(kx_all[i][mask])
        ky_parts.append(ky_all[i][mask])
    sig_flat = np.concatenate(sig_parts, axis=1)    # [n_coils, n_pts_total]
    kx_bin = np.concatenate(kx_parts)
    ky_bin = np.concatenate(ky_parts)
    return sig_flat, kx_bin, ky_bin



def _estimate_bin_lipschitz(nufft_fwd, nufft_adj, smaps_t, omega_t, dcomp, im_size,
                             n_power_iter=10):
    """Power iteration for the largest eigenvalue of A^H W A for one bin's
    operator (A = coil-sensitivity-encoded NUFFT, W = density compensation).
    Used to set a safe FISTA step size (1/L). The joint operator across bins
    is block-diagonal (each bin only touches its own image slice), so the
    global Lipschitz constant is the max over per-bin estimates, computed by
    the caller.
    """
    v = torch.randn(1, 1, *im_size, dtype=torch.complex64)
    v = v / v.abs().max()
    for _ in range(n_power_iter):
        Av = nufft_fwd(v, omega_t, smaps=smaps_t)
        AHAv = nufft_adj(dcomp * Av, omega_t, smaps=smaps_t)
        v = AHAv / (AHAv.abs().max() + 1e-12)
    Av = nufft_fwd(v, omega_t, smaps=smaps_t)
    AHAv = nufft_adj(dcomp * Av, omega_t, smaps=smaps_t)
    L = (torch.sum((v.conj() * AHAv).real) / torch.sum((v.conj() * v).real)).item()
    return L


def xdgrasp_reconstruct_cine(kx_all, ky_all, valid_data, binned_indices, n_bins,
                              n_coils, im_size, pixel_size, nx, smaps,
                              n_iter=30, lambda_spatial=5e-6, lambda_temporal=1e-3,
                              tv_inner_iter=10, wrap_temporal=True, lipschitz_iter=10):
    """XD-GRASP-style joint reconstruction of all n_bins cardiac phases:
    per-bin data consistency (identical forward model to cg_sense_reconstruct_cine)
    coupled across bins by spatial + temporal TV regularization, solved by
    FISTA. Unlike CG-SENSE, no bin is reconstructed in isolation -- the
    temporal TV term lets each bin borrow structure from its neighbors
    without merging their raw k-space the way wide keyhole/window sharing
    does, so it doesn't carry the same motion-averaging cost.

    The spatial and temporal proximal steps are applied sequentially each
    iteration (prox_temporal(prox_spatial(x))) rather than jointly -- this
    is the standard operator-splitting heuristic used throughout the
    compressed-sensing MRI literature for combined regularizers; it is not
    the exact proximal operator of the sum, but converges well in practice.
    """
    nufft_fwd = tkbn.KbNufft(im_size=im_size)
    nufft_adj = tkbn.KbNufftAdjoint(im_size=im_size)
    smaps_t = torch.from_numpy(smaps).unsqueeze(0).to(torch.complex64)

    # Precompute per-bin operators/data once -- reused every FISTA iteration.
    omegas, dcomps, ys = [], [], []
    for b in range(n_bins):
        entries = binned_indices[b]
        if len(entries) == 0:
            omegas.append(None); dcomps.append(None); ys.append(None)
            continue
        sig_flat, kx_bin, ky_bin = gather_bin_samples(valid_data, kx_all, ky_all, entries, n_coils)
        y = torch.from_numpy(sig_flat).unsqueeze(0).to(torch.complex64)
        omega = torch.from_numpy(np.stack([kx_bin, ky_bin]) * pixel_size * 2 * np.pi).to(torch.float32)
        dcomp = tkbn.calc_density_compensation_function(omega, im_size)
        omegas.append(omega); dcomps.append(dcomp); ys.append(y)
        print(f"  Bin {b}: {len(entries)} readouts, {kx_bin.size} samples")

    # Step size: 1/L, L = max over bins of the per-bin A^H W A spectral norm.
    L_bins = [
        _estimate_bin_lipschitz(nufft_fwd, nufft_adj, smaps_t, omegas[b], dcomps[b],
                                 im_size, n_power_iter=lipschitz_iter)
        for b in range(n_bins) if omegas[b] is not None
    ]
    L = max(L_bins) * 1.05  # small safety margin
    step = 1.0 / L
    print(f"XD-GRASP: estimated Lipschitz constant L={L:.4g}, step size={step:.4g}")

    x = torch.zeros(n_bins, nx, nx, dtype=torch.complex64)
    z = x.clone()
    t_k = 1.0

    for it in range(n_iter):
        grad = torch.zeros_like(x)
        for b in range(n_bins):
            if omegas[b] is None:
                continue
            zb = z[b:b+1].unsqueeze(0)  # [1, 1, nx, nx]
            Az = nufft_fwd(zb, omegas[b], smaps=smaps_t)
            AHWAz_minus_y = nufft_adj(dcomps[b] * (Az - ys[b]), omegas[b], smaps=smaps_t)
            grad[b] = AHWAz_minus_y[0, 0]

        x_new = z - step * grad
        x_new = prox_tv_spatial(x_new, weight=step * lambda_spatial, n_iter=tv_inner_iter)
        x_new = prox_tv_temporal(x_new, weight=step * lambda_temporal,
                                  n_iter=tv_inner_iter, wrap=wrap_temporal)

        t_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        z = x_new + ((t_k - 1) / t_new) * (x_new - x)
        x, t_k = x_new, t_new

        if (it + 1) % 10 == 0 or it == n_iter - 1:
            data_res = sum(
                torch.sum((dcomps[b] * (nufft_fwd(x[b:b+1].unsqueeze(0), omegas[b], smaps=smaps_t) - ys[b])).abs()**2).item()
                for b in range(n_bins) if omegas[b] is not None
            )
            print(f"  iter {it+1:>3}/{n_iter}: data residual {data_res:.4e}")

    stack = np.zeros((n_bins, nx, nx), dtype=np.float32)
    for b in range(n_bins):
        stack[b, :, :] = np.rot90(x[b].abs().cpu().numpy(), k=3)
    return stack
