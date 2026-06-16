#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:05:00 2026

@author: jonah
"""
import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp
import GERecon
import torch
import torchkbnufft as tkbn

def read_seq_defs(filename):
    """Loads the .seq file and returns the sequence object and its definitions."""
    seq = pp.Sequence()
    seq.read(filename)
    return seq, seq.definitions

# --- 1. Load Sequence and GE Data ---
seq_filename = '../sequences/cest/spiral_cest_gauss_100_bpm_75p0_ppm_ge.seq'
h5_filename = '/Users/jonah/Documents/MRI_Data/UCSF/Exam3312/Series14/ScanArchive_415MB3T_20260613_031845974.h5'

print(f"Loading sequence and data...")
seq, defs = read_seq_defs(seq_filename)
archive = GERecon.Archive(h5_filename)

# Extract basic definitions
n_interleaves = int(defs['N_Interleaves'])
n_samples = int(defs['MaxAdcSegmentLength'])
nx = int(defs.get('Nx', 128))       # Default to 128 if missing
fov = defs.get('FOV', [0.2])[0]     # Default to 0.2m (20cm) if missing

# Extract the 12 interleaves from the GE archive
frame = archive.NextFrame()
n_coils = np.shape(frame)[1]

# Set up ksp array: [n_samples, n_coils, n_interleaves]
ksp = np.zeros((n_samples, n_coils, n_interleaves), dtype=np.complex64)
ksp[:, :, 0] = frame

for i in range(1, n_interleaves):
    ksp[:, :, i] = archive.NextFrame()

coil_energy = np.mean(np.abs(ksp), axis=(0, 2))  # [n_coils]
plt.bar(range(n_coils), coil_energy)

print(f"Data Extracted: {n_interleaves} interleaves, {n_coils} coils, {n_samples} samples.")

# --- 2. Trajectory Preparation ---
print("Calculating k-space trajectory...")
traj_delay_sec = 2e-6
k_traj_adc, _, _, _, _ = seq.calculate_kspace(trajectory_delay=traj_delay_sec)

traj_len = n_interleaves * n_samples
kx = k_traj_adc[0, :traj_len]
ky = k_traj_adc[1, :traj_len]
pixel_size = fov / nx

# Scale to TorchKbNufft required bounds [-pi, pi]
omega = np.stack([kx, ky]) * pixel_size * 2 * np.pi
omega = torch.from_numpy(omega).to(torch.float32)

# --- 3. Reconstruction Setup ---
im_size = (nx, nx)
nufft_adj = tkbn.KbNufftAdjoint(im_size=im_size)
dcomp = tkbn.calc_density_compensation_function(omega, im_size)

# Reshape k-space to match trajectory
# Current shape: [n_samples, n_coils, n_interleaves]
# Required flat shape for NuFFT: [n_coils, n_interleaves * n_samples]
ksp_reordered = ksp.transpose(1, 2, 0)  # Now [n_coils, n_interleaves, n_samples]
sig_flat = ksp_reordered.reshape(n_coils, -1)
sig_tensor = torch.from_numpy(sig_flat).unsqueeze(0).to(torch.complex64)

# --- 4. Reconstruction Execution ---
print("Running TorchKbNufft...")
reco = nufft_adj(sig_tensor * dcomp, omega)

# Root Sum of Squares (RSS) coil combination
reco_final = torch.sqrt(torch.sum(reco.abs()**2, dim=1))[0]
reco_image = reco_final.cpu().numpy()

# Rotate for parity with the scanner (as implemented in your original script)
reco_image = np.rot90(reco_image, k=0) 

# --- 5. Display ---
fig, ax = plt.subplots(figsize=(6, 6))
vmax_val = np.percentile(reco_image, 99)

ax.imshow(reco_image, cmap='gray', vmin=0, vmax=vmax_val)
ax.set_title(f"TorchKbNufft GE Recon\n{seq_filename.split('/')[-1]}", fontsize=12)
ax.axis('off')

plt.tight_layout()
plt.show()