#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:43:27 2026

@author: jonah
"""
import math
import re
import pypulseq as pp
import numpy as np
import matplotlib.pyplot as plt
import twixtools
import torch
import torchkbnufft as tkbn
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from roipoly import RoiPoly

# --- Helper functions ---
def read_image_data(filename):
    """Reads raw Siemens .dat file and returns image data as a 3D array."""
    out = list()
    twix_data = twixtools.read_twix(filename, parse_pmu=False)
    for mdb in twix_data[-1]['mdb']:
        if mdb.is_image_scan():
            out.append(mdb.data)
    return np.asarray(out)  # Shape: [acquisitions, n_channel, n_column]

def read_seq_defs(filename):
    """Loads the .seq file and returns the sequence object and its definitions."""
    seq = pp.Sequence()
    seq.read(filename)
    return seq, seq.definitions

def reconstruct_spiral(twix_filename, seq_filename):
    """Reconstructs spiral data from TWIX and .seq files using TorchKbNufft."""
    # 1. Load Sequence and Data
    seq, defs = read_seq_defs(seq_filename)
    ksp_raw = read_image_data(twix_filename)

    # 2. Extract & Infer Parameters
    n_interleaves = int(defs['N_Interleaves'])
    n_samples     = int(defs['MaxAdcSegmentLength'])
    n_coils       = ksp_raw.shape[1]
    nx            = int(defs['Nx'])
    fov           = defs['FOV'][0]  # [m]

    n_offsets = ksp_raw.shape[0] // n_interleaves

    # Reshape k-space: [n_offsets, n_interleaves, n_coils, n_samples]
    ksp = ksp_raw.reshape((n_offsets, n_interleaves, n_coils, n_samples))
    print(f"\n--- Reconstructing {twix_filename.split('/')[-1]} ---")
    print(f"Data Reshaped: {n_offsets} offsets, {n_interleaves} interleaves, {n_coils} coils.")

    # 3. Trajectory preparation
    traj_delay_sec = 5.0e-6
    k_traj_adc, _, _, _, _ = seq.calculate_kspace(trajectory_delay=traj_delay_sec)

    traj_len   = n_interleaves * n_samples
    kx         = k_traj_adc[0, :traj_len]
    ky         = k_traj_adc[1, :traj_len]
    pixel_size = fov / nx
    omega      = np.stack([kx, ky]) * pixel_size * 2 * np.pi
    omega      = torch.from_numpy(omega).to(torch.float32)

    # 4. Reconstruction setup
    im_size  = (nx, nx)
    nufft_adj = tkbn.KbNufftAdjoint(im_size=im_size)
    dcomp    = tkbn.calc_density_compensation_function(omega, im_size)

    recon_stack = np.zeros((n_offsets, nx, nx), dtype=np.float32)
    print("Starting Loop Reconstruction...")

    # 5. Reconstruction loop
    for i in range(n_offsets):
        ksp_offset = ksp[i, :, :, :]
        sig_flat   = ksp_offset.transpose(1, 0, 2).reshape(n_coils, -1)
        sig_tensor = torch.from_numpy(sig_flat).unsqueeze(0).to(torch.complex64)

        reco       = nufft_adj(sig_tensor * dcomp, omega)
        reco_final = torch.sqrt(torch.sum(reco.abs()**2, dim=1))[0]
        
        reco_image = reco_final.cpu().numpy()
        reco_image = np.rot90(reco_image, k=3) # JWW added 5/15/26 for parity with scanner
        recon_stack[i, :, :] = reco_image

        if (i + 1) % 5 == 0 or (i + 1) == n_offsets:
            print(f"Processed {i+1}/{n_offsets} offsets.")

    return recon_stack, defs


def extract_ref_from_stack(recon_stack, offsets_ppm, ref_offset_threshold=50.0):
    """
    Checks whether a far off-resonance reference offset is embedded in the stack.
    If found, removes it from the stack/offsets and returns it separately.
    """
    ref_mask = np.abs(offsets_ppm) >= ref_offset_threshold
    if not np.any(ref_mask):
        return recon_stack, offsets_ppm, None, False

    ref_image         = recon_stack[ref_mask].mean(axis=0)
    recon_stack_clean = recon_stack[~ref_mask]
    offsets_clean     = offsets_ppm[~ref_mask]

    n_ref = ref_mask.sum()
    print(f"  Found {n_ref} embedded reference frame(s) "
          f"at offset(s): {offsets_ppm[ref_mask]} ppm  →  extracted as S0.")
    return recon_stack_clean, offsets_clean, ref_image, True


def correct_b0_from_map(normalized_stack, offsets_ppm, b0_map_ppm, smooth=False):
    """
    Voxelwise B0 correction using an externally measured B0 map (e.g. from WASABI),
    rather than estimating the shift from the Z-spectrum minimum.
    """
    sort_idx       = np.argsort(offsets_ppm)
    offsets_sorted = offsets_ppm[sort_idx]
    stack_sorted   = normalized_stack[sort_idx]

    if smooth:
        b0_map_ppm = gaussian_filter(b0_map_ppm, sigma=1.0)

    _, nx, ny   = stack_sorted.shape
    corrected_stack = np.zeros_like(stack_sorted)

    print("Applying WASABI B0 correction...")
    for x in range(nx):
        for y in range(ny):
            z_curve   = stack_sorted[:, x, y]
            b0        = b0_map_ppm[x, y]
            interp_fn = interp1d(offsets_sorted - b0, z_curve,
                                 kind='linear', bounds_error=False,
                                 fill_value=(z_curve[0], z_curve[-1]))
            corrected_stack[:, x, y] = interp_fn(offsets_sorted)

    # Restore original offset ordering
    restore_idx = np.argsort(sort_idx)
    corrected_stack = corrected_stack[restore_idx]

    print(f"  B0 correction applied. Map range: "
          f"{b0_map_ppm.min():.3f} to {b0_map_ppm.max():.3f} ppm")
    return corrected_stack, b0_map_ppm


def compute_mtr_asym_from_zspectrum(normalized_stack, offsets_ppm, ppm_tol=0.01):
    """
    Computes MTRasym from a single normalized Z-spectrum stack by pairing
    +offset and -offset frames.

        MTRasym(+ω) = Z(-ω) - Z(+ω)
    """
    positive_offsets = np.unique(offsets_ppm[offsets_ppm > 0])
    paired_offsets   = []
    mtr_asym_frames  = []

    for pos in sorted(positive_offsets):
        idx_pos = np.where(np.abs(offsets_ppm - pos) < ppm_tol)[0]
        idx_neg = np.where(np.abs(offsets_ppm + pos) < ppm_tol)[0]

        if idx_pos.size == 0 or idx_neg.size == 0:
            print(f"  Warning: no matching pair for ±{pos:.3f} ppm — skipping.")
            continue

        z_pos = normalized_stack[idx_pos[0]]
        z_neg = normalized_stack[idx_neg[0]]

        mtr_asym_frames.append(z_neg - z_pos)
        paired_offsets.append(pos)

    if not paired_offsets:
        raise RuntimeError("No ±offset pairs found. Check your offset list.")

    print(f"  MTRasym computed for {len(paired_offsets)} offset pair(s): "
          f"{[f'{p:.2f}' for p in paired_offsets]} ppm")

    return np.stack(mtr_asym_frames, axis=0), np.array(paired_offsets)


def plot_image_grid(stack, offsets, title, cmap='gray', vmin=None, vmax=None,
                    colorbar_label=None, cols=5, symmetric_clim=False):
    """
    Generic helper: plots a stack of 2D images on a grid with offset labels.
    """
    n    = len(offsets)
    rows = math.ceil(n / cols)

    if vmax is None:
        vmax = np.percentile(np.abs(stack[np.isfinite(stack)]), 99)
    if symmetric_clim:
        vmin = -vmax
    if vmin is None:
        vmin = 0

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=14)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes_flat):
        if i < n:
            im = ax.imshow(stack[i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{offsets[i]:.2f} ppm")
            if colorbar_label and i == n - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label=colorbar_label)
        ax.axis('off')

    plt.tight_layout()
    return fig


# ==========================================
# Main Execution Block
# ==========================================

# Acquisition mode:
#   'single'      – single offset image (reconstruction only; skips B0 interpolation & MTRasym)
#   'separate'    – positive offsets, negative offsets, and reference are separate acquisitions
#   'zspectrum'   – full Z-spectrum (positive + negative offsets) in one acquisition;
ACQUISITION_MODE = 'single'

# --- File locations ---
twix_filename_cest = '/Users/jonah/Documents/MRI_Data/BIC/Sam_Jun82026/raw/meas_MID00083_FID45671_pulseq_spiral_cest_3_5_ppm_spsp.dat'
seq_filename_cest  = '/Users/jonah/Documents/MRI_Data/BIC/Sam_Jun82026/sequences/spiral_cest_gauss_40_bpm_75.0_ppm.seq'

# Only used in 'separate' mode
twix_filename_asym = '/Volumes/JWW-BIC/PureTemp_Cr_PCr/meas_MID00235_FID44711_pulseq_spsp_zspec.dat'
seq_filename_asym  = '/Volumes/JWW-BIC/PureTemp_Cr_PCr/spiral_cest_gauss_zspec_60_bpm.seq'

# Reference: set SEPARATE_REF=False when embedded in the CEST acquisition
twix_filename_ref = None
seq_filename_ref  = None
SEPARATE_REF      = False

# WASABI maps (.npy, in ppm)
b0_map_path = '../data/recon/MID00231_FID44707_b0.npy'

# Extract ID for saving/logging
match        = re.search(r'(MID\d+_FID\d+)', twix_filename_cest)
extracted_id = match.group(1) if match else "UnknownID"

# ── 1. Reconstruct ────────────────────────────────────────────────────────────
cest_stack, cest_defs = reconstruct_spiral(twix_filename_cest, seq_filename_cest)
offsets_ppm = np.atleast_1d(np.array(cest_defs['Offsets_ppm'], dtype=float))

# Auto-fallback to single offset mode to prevent interpolation/pairing crashes
if ACQUISITION_MODE != 'separate' and len(offsets_ppm) == 1:
    print("\nOnly 1 offset detected. Automatically adapting to 'single' acquisition mode.")
    ACQUISITION_MODE = 'single'

if ACQUISITION_MODE == 'separate':
    asym_stack, asym_defs = reconstruct_spiral(twix_filename_asym, seq_filename_asym)

# ── 2. Obtain S0 ──────────────────────────────────────────────────────────────
if SEPARATE_REF:
    ref_stack, _ = reconstruct_spiral(twix_filename_ref, seq_filename_ref)
    ref_image = ref_stack[0]
    print("\nUsing separate reference acquisition as S0.")
elif ACQUISITION_MODE == 'single':
    print("\nSingle offset mode without separate reference. Images will remain unnormalized.")
    ref_image = None
    found = False
else:
    cest_stack, offsets_ppm, ref_image, found = extract_ref_from_stack(
        cest_stack, offsets_ppm
    )
    if not found and ACQUISITION_MODE == 'separate':
        asym_offsets = np.atleast_1d(np.array(asym_defs['Offsets_ppm'], dtype=float))
        asym_stack, asym_offsets, ref_image, found = extract_ref_from_stack(
            asym_stack, asym_offsets
        )
    if not found:
        raise RuntimeError(
            "SEPARATE_REF=False but no far off-resonance frame (|ppm| >= 50) "
            "was found. Check your sequence definitions or set SEPARATE_REF=True."
        )

# ── 3. Plot unnormalized images ───────────────────────────────────────────────
print("\nPlotting unnormalized reconstructed images...")

sort_idx       = np.argsort(offsets_ppm)
offsets_sorted = offsets_ppm[sort_idx]
cest_sorted    = cest_stack[sort_idx]

plot_image_grid(
    cest_sorted, offsets_sorted,
    title=f'Unnormalized Reconstructed Images — {extracted_id}',
    cmap='gray',
    cols=5
)
plt.show()

# ── 4. Normalize ──────────────────────────────────────────────────────────────
if ref_image is not None:
    print("\nNormalizing stack...")
    epsilon = 1e-12
    normalized_cest_stack = cest_stack / (ref_image[np.newaxis] + epsilon)
else:
    normalized_cest_stack = cest_stack

# ── 5. B0 Correction ──────────────────────────────────────────────────────────
if ACQUISITION_MODE == 'single' or len(offsets_ppm) < 2:
    print("\nSkipping B0 correction: Single offset mode or insufficient offsets for interpolation.")
    b0_map_ppm = None
else:
    b0_map_ppm = np.load(b0_map_path)
    
    assert b0_map_ppm.shape == (cest_stack.shape[1], cest_stack.shape[2]), (
        f"B0 map shape {b0_map_ppm.shape} does not match image shape "
        f"{cest_stack.shape[1:]}. Check that WASABI and CEST were acquired "
        f"with the same matrix size and FOV."
    )
    
    print(f"B0 map loaded. Range: {b0_map_ppm.min():.3f} to {b0_map_ppm.max():.3f} ppm")
    
    normalized_cest_stack, b0_map_ppm = correct_b0_from_map(
        normalized_cest_stack, offsets_ppm, b0_map_ppm
    )

# ── 6. Compute MTRasym ────────────────────────────────────────────────────────
print("\nCalculating MTRasym...")

if ACQUISITION_MODE == 'single':
    print(" -> Skipping MTRasym calculation for single offset.")
    mtr_asym = None
    paired_offsets = offsets_ppm
elif ACQUISITION_MODE == 'separate':
    normalized_asym_stack = asym_stack / (ref_image[np.newaxis] + epsilon)
    mtr_asym       = normalized_asym_stack - normalized_cest_stack
    paired_offsets  = offsets_ppm
elif ACQUISITION_MODE == 'zspectrum':
    mtr_asym, paired_offsets = compute_mtr_asym_from_zspectrum(
        normalized_cest_stack, offsets_ppm
    )

print(f"Reconstruction complete for ID: {extracted_id}")

# ── 7. Plotting ───────────────────────────────────────────────────────────────
n_pairs = len(paired_offsets)

# Re-sort after B0 correction for display
sort_idx_norm  = np.argsort(offsets_ppm)
offsets_sorted = offsets_ppm[sort_idx_norm]
norm_sorted    = normalized_cest_stack[sort_idx_norm]

if ACQUISITION_MODE == 'single':
    # --- Single Offset Plotting (Bypasses Pair logic) ---
    print("\nPlotting Single Offset Image...")
    ncols = 2 if ref_image is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 5))
    axes = np.atleast_1d(axes)

    col = 0
    if ref_image is not None:
        vmax_ref = np.percentile(ref_image, 99)
        axes[col].imshow(ref_image, cmap='gray', vmin=0, vmax=vmax_ref)
        axes[col].set_title("Reference ($S_0$)")
        col += 1

    display_stack = norm_sorted if ref_image is not None else cest_stack
    vmax_val = np.percentile(display_stack[0], 99)
    axes[col].imshow(display_stack[0], cmap='gray', vmin=0, vmax=vmax_val)
    
    title_suffix = " (Normalized)" if ref_image is not None else " (Unnormalized)"
    axes[col].set_title(f"Offset: {offsets_sorted[0]:.2f} ppm{title_suffix}")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

elif n_pairs == 1:
    # --- Single Pair MTRasym Comparison ---
    print("\nPlotting Single Offset Comparison...")
    ncols = 3 if ACQUISITION_MODE == 'zspectrum' else 4
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 5))
    fig.suptitle(f'Single Offset Pair: |{paired_offsets[0]:.2f}| ppm\n{extracted_id}',
                 fontsize=16)

    col = 0
    vmax_ref = np.percentile(ref_image, 99)
    axes[col].imshow(ref_image, cmap='gray', vmin=0, vmax=vmax_ref)
    axes[col].set_title("Reference ($S_0$)")
    col += 1

    if ACQUISITION_MODE == 'separate':
        vmax_val = np.percentile(cest_stack[0], 99)
        axes[col].imshow(cest_stack[0], cmap='gray', vmin=0, vmax=vmax_val)
        axes[col].set_title(f"Saturate (+{paired_offsets[0]:.2f} ppm)")
        col += 1
        axes[col].imshow(asym_stack[0], cmap='gray', vmin=0, vmax=vmax_val)
        axes[col].set_title(f"Saturate (-{paired_offsets[0]:.2f} ppm)")
        col += 1

    vmax_mtr = np.max(np.abs(np.percentile(mtr_asym[0], [1, 99])))
    if vmax_mtr == 0: vmax_mtr = 0.05
    im = axes[col].imshow(mtr_asym[0], cmap='coolwarm',
                          vmin=-vmax_mtr, vmax=vmax_mtr)
    axes[col].set_title("MTRasym")
    fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04,
                 label='$\Delta M_z/M_0$')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

else:
    # --- Grid Plots ---
    print(f"\nPlotting grids for {n_pairs} offset pair(s)...")
    cols = 5

    # 1. Unnormalized (sorted by offset)
    plot_image_grid(
        cest_sorted, offsets_sorted,
        title=f'Unnormalized Images — {extracted_id}',
        cmap='gray',
        cols=cols
    )

    # 2. Z-spectrum / normalized (sorted by offset) — zspectrum mode only
    if ACQUISITION_MODE == 'zspectrum':
        plot_image_grid(
            norm_sorted, offsets_sorted,
            title=f'Z-Spectrum (S/S$_0$, B0-corrected) — {extracted_id}',
            cmap='gray',
            vmin=0, vmax=1,
            cols=cols
        )

    # 3. MTRasym
    vmax_mtr_grid = np.max(np.abs(np.percentile(mtr_asym, [1, 99])))
    if vmax_mtr_grid == 0: vmax_mtr_grid = 0.05
    plot_image_grid(
        mtr_asym, paired_offsets,
        title=f'MTRasym — {extracted_id}',
        cmap='coolwarm',
        vmax=vmax_mtr_grid,
        symmetric_clim=True,
        colorbar_label='$\Delta M_z/M_0$',
        cols=cols
    )

    plt.show()
    
#%%
from roipoly import MultiRoi
fig, ax = plt.subplots()
ax.imshow(ref_image, cmap='gray')
ax.axis('off')

multiroi_named = MultiRoi(roi_names=['Cr', 'PCr', 'Cr+PCr'])

#%%
# ── 8. Plot average Z-spectra per ROI ─────────────────────────────────────────
print("\nPlotting average Z-spectra per ROI...")

# norm_sorted and offsets_sorted are already sorted by offset (from step 7)
# Shape of norm_sorted: [n_offsets, nx, ny]

fig, ax = plt.subplots(figsize=(8, 5))

z_spectra_gauss = {}

for roi_name, roi in multiroi_named.rois.items():
    mask = roi.get_mask(ref_image)          # boolean mask, shape [nx, ny]
    # Average over all masked voxels for each offset
    z_mean = np.array([
        norm_sorted[i][mask].mean() for i in range(len(offsets_sorted))
    ])
    ax.plot(offsets_sorted, z_mean, marker='o', markersize=3, label=roi_name)
    z_spectra_gauss[roi_name] = z_mean

ax.set_xlabel('Offset (ppm)')
ax.set_ylabel('Z = S/S$_0$')
ax.set_title(f'Average Z-Spectra by ROI — {extracted_id}')
ax.invert_xaxis()          # convention: downfield (positive) on the left
ax.set_ylim(0, 1.05)
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
