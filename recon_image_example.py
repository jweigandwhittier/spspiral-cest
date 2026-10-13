#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:12:26 2026
@author: jonah
"""
import numpy as np
from utils import extract_raw

#%% Step 1: Reconstruct image
from utils import recon

data_fn = '/Volumes/JWW-BIC/lipids/test_te/meas_MID00068_FID54555_pulseq_v1p5.dat'
seq_fn = '/Volumes/JWW-BIC/lipids/test_te/continuous_spiral_gauss_75p0_ppm.seq'

_, ksp, _, vendor = extract_raw.extract_kspace(data_fn)

seq = extract_raw.load_seq(seq_fn)
k_traj_adc, _, _, _, t_adc = seq.calculate_kspace(trajectory_delay=10e-6)

# Get nominal matrix size
nx = seq.definitions['Nx']
fov = seq.definitions['FOV'][0]

# TR bookkeeping
n_trs = int(seq.definitions['N_TRs'])
n_adc = int(seq.definitions['MaxAdcSegmentLength'])
n_extra = 1 if vendor == 'ge' else 0  
pislquant = int(seq.definitions.get('pislquant', 0))
n_burnin = 0

main_start = n_extra + pislquant
ksp_main = ksp[:, main_start : main_start + n_trs, :]
ksp_burnin = ksp_main[:, n_burnin:, :]

# Rotated trajectory for ALL n_trs main TRs (handles calibration offset
# internally; does NOT know about n_burnin -- sliced separately below)
kx_full_all, ky_full_all = recon.get_rotated_trajectory(seq, k_traj_adc, n_adc, n_trs, pislquant)
kx_full = kx_full_all[n_burnin:]
ky_full = ky_full_all[n_burnin:]

image = recon.adjoint_nufft_from_traj(ksp_burnin, kx_full, ky_full, nx, fov)

#%% Step 2: Show image
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.axis('off')

