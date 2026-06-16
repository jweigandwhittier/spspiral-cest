#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:07:19 2026
@author: jonah
"""
import re
import pypulseq as pp
import numpy as np
import matplotlib.pyplot as plt
import twixtools
import torch
import torchkbnufft as tkbn
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

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

# --- File locations & loading ---
twix_filename = '/Users/jonah/Documents/Vandsburger_Lab/open-cardiac-cest/data/raw/PureTemp_Cr_PCr/meas_MID00231_FID44707_pulseq_wasabi.dat'
seq_filename = '/Users/jonah/Documents/Vandsburger_Lab/open-cardiac-cest/data/raw/Compare_FieldMap_Zspec_2/spiral_wasabi_60_bpm.seq'
seq, defs = read_seq_defs(seq_filename)
ksp_raw = read_image_data(twix_filename)

# Get ID
match = re.search(r'(MID\d+_FID\d+)', twix_filename)
if match:
    extracted_id = match.group(1)

# --- Parameter extraction & reshaping ---
n_interleaves = int(defs['N_Interleaves'])
n_samples = int(defs['MaxAdcSegmentLength'])
n_coils = ksp_raw.shape[1]
offsets_ppm = defs['Offsets_ppm'] # [ppm]
n_offsets = len(offsets_ppm)
nx = int(defs['Nx'])
fov = defs['FOV'][0] # [m]
nominal_b1_T = defs['B1peak']*1e-6 # [uT]
tp = defs['tp'] # [s]
b0_T = defs['B0']
# Also need gamma
gamma = 2.675221e8  # [rad/s/T]
# Reshape k-space: [n_offsets, n_interleaves, n_coils, n_samples]
ksp = ksp_raw.reshape((n_offsets, n_interleaves, n_coils, n_samples))
print(f"Data Loaded: {n_offsets} offsets, {n_interleaves} interleaves, {n_coils} coils.")

# --- Trajectory preparation ---
# Compensation for system gradient delay
# I think this comes from ADC mismatch in the sequence (0.5 * gradient raster)
traj_delay_sec = 5.0e-6 
k_traj_adc, _, _, _, _ = seq.calculate_kspace(trajectory_delay=traj_delay_sec)
# Calculate the number of points in one image
traj_len = n_interleaves * n_samples
kx = k_traj_adc[0, :traj_len]
ky = k_traj_adc[1, :traj_len]
# Scale to radians [-pi, pi] for torchkbnufft
pixel_size = fov / nx
omega = np.stack([kx, ky]) * pixel_size * 2 * np.pi
omega = torch.from_numpy(omega).to(torch.float32)

# --- Reconstruction setup ---
im_size = (nx, nx)
nufft_adj = tkbn.KbNufftAdjoint(im_size=im_size)
dcomp = tkbn.calc_density_compensation_function(omega, im_size)
# Array to store the final images
recon_stack = np.zeros((n_offsets, nx, nx), dtype=np.float32)
print("Starting Loop Reconstruction...")

# --- Reconstruction loop ---
for i in range(n_offsets):
    # Select data for current offset: [n_interleaves, n_coils, n_samples]
    ksp_offset = ksp[i, :, :, :]
    # Reorganize to [1, n_coils, total_samples_per_image]
    # Transpose to (coils, interleaves, samples) then flatten interleaves/samples
    sig_flat = ksp_offset.transpose(1, 0, 2).reshape(n_coils, -1)
    sig_tensor = torch.from_numpy(sig_flat).unsqueeze(0).to(torch.complex64)
    # Execute Adjoint NUFFT
    reco = nufft_adj(sig_tensor * dcomp, omega)
    # Channel Combination (RSS)
    reco_final = torch.sqrt(torch.sum(reco.abs()**2, dim=1))[0]
    # Save to stack
    reco_image = reco_final.cpu().numpy()
    reco_image = np.rot90(reco_image, k=3) # JWW added 5/15/26 for parity with scanner
    recon_stack[i, :, :] = reco_image
    if (i + 1) % 5 == 0 or (i + 1) == n_offsets:
        print(f"Processed {i+1}/{n_offsets} offsets.")

# --- Generate field maps ---
# Normalize by S0 image
s0_image = recon_stack[0,:,:]
normalized_stack = recon_stack / s0_image
normalized_stack = np.delete(normalized_stack, 0, axis=0)
offsets_ppm_clean = np.delete(offsets_ppm, 0)

# WASABI fits require the offset and gamma*B1 to be in the same units (e.g., rad/s).
larmor_freq_mhz = gamma * b0_T / (2 * np.pi) / 1e6
# Convert ppm to rad/s
offsets_rad_s = offsets_ppm_clean * larmor_freq_mhz * 2 * np.pi 

# --- Define the fit function ---
# Note: curve_fit requires the independent variable (offset) as the first argument
def wasabi_fit(offset, omega, c, d, b1):
    # offset and omega are in rad/s; gamma*b1 is in rad/s
    term1 = np.sin(np.arctan2(gamma * b1, offset - omega))**2
    term2 = np.sin(np.sqrt((gamma * b1)**2 + (offset - omega)**2) * tp/2)**2
    return np.abs(c - d * term1 * term2)

def fit_single_pixel(y, x, pixel_data, offsets, guess, fit_bounds):
    # Skip if the pixel data is completely flat or contains NaNs
    if np.isnan(pixel_data).any() or np.all(pixel_data == pixel_data[0]):
        return y, x, 0.0, 0.0
    
    try:
        popt, _ = curve_fit(wasabi_fit, offsets, pixel_data, p0=guess, bounds=fit_bounds, maxfev=10000)
        return y, x, popt[0], popt[3]
    except RuntimeError:
        # If curve_fit fails to converge, return 0 for the parameters
        return y, x, 0.0, 0.0

# Initialize maps
omega_map = np.zeros((nx, nx))
b1_map = np.zeros((nx, nx))

# Initial guesses: [omega (rad/s), c (baseline), d (dip depth), b1 (Tesla)]
p0 = [0.0, 1.0, 2.0, 3.7e-6] 

# Set bounds to prevent the optimizer from drifting into unphysical values
bounds = ([-np.inf, 0.5, 0.0, 0.0], [np.inf, 1.5, 5.0, 10e-6])

total_pixels = nx ** 2

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(fit_single_pixel)(
        y, x, normalized_stack[:, y, x], offsets_rad_s, p0, bounds
    )
    for y in range(nx) for x in range(nx)
)

print("Fitting complete. Reconstructing maps...")

# Reconstruct the 2D parameter maps from the 1D list of results
for y, x, omega_val, b1_val in results:
    omega_map[y, x] = omega_val
    b1_map[y, x] = b1_val

# --- Process and plot parameter maps ---
# Convert the maps to standard units
omega_map_ppm = omega_map / (larmor_freq_mhz * 2 * np.pi)

# Calculate relative B1 (actual / nominal)
b1_map_rel = b1_map / nominal_b1_T

# Create two figures, one for each map, overlaid on S0
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # --- Subplot 1: B0 Shift Map Overlaid ---
axes[0].set_title(r'$\Delta B_0$ Shift ($\omega_0$) Map [ppm]', fontsize=14)

im0 = axes[0].imshow(omega_map_ppm, cmap='viridis')
axes[0].axis('off')

cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.set_label('ppm', fontsize=12)

# # --- Subplot 2: Relative B1 Map Overlaid ---
axes[1].set_title(r'Relative $B_1$ Map (Actual / Nominal)', fontsize=14)

im1 = axes[1].imshow(b1_map_rel, cmap='viridis')
axes[1].axis('off')

cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label('Fraction of Nominal $B_1$', fontsize=12)

plt.tight_layout()
plt.show()

# --- Save reconstructed maps ---
np.save(f'../data/recon/{extracted_id}_b1.npy', b1_map_rel)
np.save(f'../data/recon/{extracted_id}_b0.npy', omega_map_ppm)