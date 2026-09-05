#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:12:47 2026
@author: jonah
"""

# %% Step 0: Choose scanner
scanner = 'ge'  # 'ge' or 'siemens'
assert scanner in ('ge', 'siemens'), "scanner must be 'ge' or 'siemens'"

paths = {
    'ge': {
        'data_dir': 'data/raw/ge/example/Exam785',  # Parent directory for GE
        'seq_fn': 'sequences/ge/continuous_spiral_zspec_gauss_75p0_ppm_ge.seq',  # Sequence for GE (with calibration)
    },
    'siemens': {
        'data_dir': 'data/raw/siemens/example/meas_MID00190_FID50697_pulseq.dat',  # Data file for Siemens
        'seq_fn': 'sequences/siemens/continuous_spiral_gauss_75p0_ppm.seq',  # Sequence for Siemens
    },
}
data_dir = paths[scanner]['data_dir']
seq_fn = paths[scanner]['seq_fn']

# %% Step 1: Extract raw data
import os
import numpy as np
from utils import extract_raw

if scanner == 'ge':
    # Sort so offsets stack in a deterministic, reproducible order
    subfolders = sorted(
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    )
    ksp_list = []
    # Extract k-space
    for folder in subfolders:
        folder_path = os.path.join(data_dir, folder)
        h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
        if len(h5_files) != 1:
            raise ValueError(f"Expected exactly one .h5 file in {folder_path}, found {len(h5_files)}")
        h5_path = os.path.join(folder_path, h5_files[0])
        _, ksp, _, vendor = extract_raw.extract_kspace(h5_path)
        ksp_list.append(ksp)

    ksp_all = np.array(ksp_list)  # [offsets, samples, interleaves, coils]

elif scanner == 'siemens':
    _, ksp_all, _ = extract_raw.extract_kspace(data_dir)

# %% Step 2: Get trajectory and recon info from Pulseq
from utils import recon

seq = extract_raw.load_seq(seq_fn)
k_traj_adc, _, _, _, t_adc = seq.calculate_kspace(trajectory_delay=0)

nx = seq.definitions['Nx']
fov = seq.definitions['FOV'][0]
n_adc = int(seq.definitions['MaxAdcSegmentLength'])
n_trs = int(seq.definitions['N_TRs'])
pislquant = int(seq.definitions.get('pislquant', 0))  # 0 for Siemens, set explicitly for GE
n_extra = 1 if scanner == 'ge' else 0

kx_full, ky_full = recon.get_rotated_trajectory(seq, k_traj_adc, n_adc, n_trs, pislquant)

# Discard calibration + dead-TR scans from raw ksp
main_start = n_extra + pislquant
ksp_real = ksp_all[:, :, main_start:main_start + n_trs, :]

# %% Step 3: Reconstruct images
n_offsets = ksp_real.shape[0]
image_stack = []
for i in range(n_offsets):
    ksp_offset = ksp_real[i, :, :, :]
    image = recon.adjoint_nufft_from_traj(ksp_offset, kx_full, ky_full, nx, fov)
    image_stack.append(image)

# %% Step 4: Draw ROI on reference image and construct Z-spectrum
import matplotlib.pyplot as plt
from roipoly import RoiPoly
ref_image = image_stack[0]
plt.figure()
plt.imshow(ref_image, cmap='gray')
plt.title('Click to draw ROI, double-click to close')
roi = RoiPoly(color='r')
mask = roi.get_mask(ref_image)
plt.figure()
plt.imshow(ref_image, cmap='gray')
plt.imshow(mask, cmap='Reds', alpha=0.3)
plt.title('ROI overlay')
plt.show()
signal = np.array([img[mask].mean() for img in image_stack])
s0 = signal[0]
z_spectrum = signal / s0
_zspec_part1 = np.arange(-10, -5, 1)
_zspec_part2 = np.arange(-5, 5.01, 0.2)
_zspec_part3 = np.arange(6, 10.01, 1)
offsets_ppm = np.round(np.concatenate((_zspec_part1, _zspec_part2, _zspec_part3)), 2)

# %% Step 5: Fit CEST peaks (in Z, physically incorrect)
from utils import cest_fitting as cf
fit = cf.two_step(z_spectrum[1:], offsets_ppm[1:])
print("B0 shift (ppm):", fit['B0_Shift'])
print(fit['Contrasts'])
cf.plot_fit(fit, title="Z-Spectrum Fit")