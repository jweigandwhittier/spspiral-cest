#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:05:48 2026

@author: jonah
"""
import torch
import twixtools
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
import torchkbnufft as tkbn
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift

def read_image_data(filename):
    """Reads raw Siemens .dat file and returns image data as a 3D array."""
    out = list()
    twix_data = twixtools.read_twix(filename, parse_pmu=False)
    yaps = twix_data[-1]['hdr']['MeasYaps']
    for mdb in twix_data[-1]['mdb']:
        if mdb.is_image_scan():
            out.append(mdb.data)
    return yaps, np.asarray(out)  # Shape: [acquisitions, n_channel, n_column]

def read_seq_defs(filename):
    """Loads the .seq file and returns the sequence object and its definitions."""
    seq = pp.Sequence()
    seq.read(filename)
    return seq, seq.definitions

def build_rotation_matrix(normal_vector, theta_radians):
    """
    Constructs the Logical-to-Physical 3x3 rotation matrix.
    R maps [Logical RO, Logical PE, Logical SS] -> [Physical X, Physical Y, Physical Z]
    """
    # 1. Normalize the slice-select (normal) vector
    n = np.array(normal_vector, dtype=float)
    if np.linalg.norm(n) == 0: 
        n = np.array([0.0, 0.0, 1.0])  # Default transverse
    n = n / np.linalg.norm(n)
    
    # 2. Guess the base Phase Encode (PE) vector based on Siemens' dominant axis rule
    abs_n = np.abs(n)
    if abs_n[0] > abs_n[1] and abs_n[0] > abs_n[2]:    # Sagittal dominant
        p_guess = np.array([0.0, 1.0, 0.0])  
    elif abs_n[1] > abs_n[0] and abs_n[1] > abs_n[2]:  # Coronal dominant
        p_guess = np.array([0.0, 0.0, 1.0])  
    else:                                              # Transversal dominant
        p_guess = np.array([0.0, 1.0, 0.0])  

    # 3. Gram-Schmidt: Force PE to be perfectly orthogonal to the Normal
    p = p_guess - np.dot(p_guess, n) * n
    p = p / np.linalg.norm(p)
    
    # 4. Readout vector is the cross product of PE and Normal
    r = np.cross(p, n)
    r = r / np.linalg.norm(r)
    
    # 5. Apply the In-Plane Rotation (theta) to RO and PE
    RO = r * np.cos(theta_radians) - p * np.sin(theta_radians)
    PE = r * np.sin(theta_radians) + p * np.cos(theta_radians)
    SS = n
    
    # Assemble the 3x3 matrix: Columns are RO, PE, SS
    R = np.column_stack((RO, PE, SS))
    return R

twix_filename = '/Users/jonah/Documents/MRI_Data/BIC/Sam_Jun82026/raw/meas_MID00083_FID45671_pulseq_spiral_cest_3_5_ppm_spsp.dat'
seq_filename = '/Users/jonah/Documents/MRI_Data/BIC/Sam_Jun82026/sequences/spiral_cest_gauss_40_bpm_75.0_ppm.seq'
girf_filename = '../girf/bic_girf_data.npz'

# Load image data and sequence
yaps, twix = read_image_data(twix_filename)
seq, defs = read_seq_defs(seq_filename)

# Load GIRF data
girf = np.load(girf_filename)
girf_complex = girf['girf_complex']
girf_freqs = girf['freqs']

# Get rotation matrix
slice_info = yaps['sSliceArray']['asSlice'][0]
print("--- Slice Orientation ---")
if 'sNormal' in slice_info:
    dSag = slice_info['sNormal'].get('dSag', 0.0)
    dCor = slice_info['sNormal'].get('dCor', 0.0)
    dTra = slice_info['sNormal'].get('dTra', 0.0)
    normal_vector = [dSag, dCor, dTra]
    print(f"Normal Vector (Sag, Cor, Tra): ({dSag}, {dCor}, {dTra})")
else:
    print("Normal Vector is purely Transversal (default)")
in_plane_rot = slice_info.get('dInPlaneRot', 0.0)
print(f"In-Plane Rotation (radians): {in_plane_rot}")

R = build_rotation_matrix(normal_vector, in_plane_rot)
R_inv = np.linalg.inv(R)

n_interleaves = int(defs['N_Interleaves'])
n_samples = int(defs['MaxAdcSegmentLength'])
n_coils = twix.shape[1]
nx = int(defs['Nx'])
fov = defs['FOV'][0]  # [m]
dt = seq.system.grad_raster_time

n_offsets = twix.shape[0] // n_interleaves

k_traj_adc, _, _, _, _ = seq.calculate_kspace(trajectory_delay=0) # Trajectory with no delay
k_traj_adc = np.reshape(k_traj_adc, (3, n_interleaves, n_samples))

# Do GIRF operations first
t_interleaf = np.arange(n_samples)*dt
n_fft_interleaf = 1 << (n_samples + 8192).bit_length()
grad_freqs_interleaf = fftshift(fftfreq(n_fft_interleaf, d=dt))
girf_interp = np.zeros((3, n_fft_interleaf), dtype=complex)
for ax in range(3):
    real_interp = np.interp(grad_freqs_interleaf, girf_freqs, np.real(girf_complex[ax, :]), 
                            left=girf_complex[ax, 0].real, right=girf_complex[ax, -1].real)
    imag_interp = np.interp(grad_freqs_interleaf, girf_freqs, np.imag(girf_complex[ax, :]), 
                            left=girf_complex[ax, 0].imag, right=girf_complex[ax, -1].imag)
    girf_interp[ax, :] = real_interp + 1j * imag_interp

# Pre-allocate full trajectory arrays for the complete profile
kx_corrected = np.zeros(n_interleaves * n_samples)
ky_corrected = np.zeros(n_interleaves * n_samples)
kx_nominal = np.zeros(n_interleaves * n_samples)
ky_nominal = np.zeros(n_interleaves * n_samples)

for i in range(n_interleaves):
    k_interleaf = k_traj_adc[:,i,:]
    
    # Store the clean nominal profile sequentially
    idx_start = i * n_samples
    idx_end = idx_start + n_samples
    kx_nominal[idx_start:idx_end] = k_interleaf[0, :]
    ky_nominal[idx_start:idx_end] = k_interleaf[1, :]
    
    # Calculate gradients
    g_log = np.diff(k_interleaf, axis=1, prepend=k_interleaf[:, 0:1]) / (seq.system.gamma * dt)
    g_phys = R @ g_log 
    g_phys_corr = np.zeros_like(g_phys)
    for ax in range(3):
        G_f = fftshift(fft(g_phys[ax, :], n=n_fft_interleaf))
        G_f_corr = G_f * girf_interp[ax, :]
        g_phys_corr[ax, :] = np.real(ifft(ifftshift(G_f_corr)))[:n_samples]
    
    # Integrate back
    k_phys_corr = np.cumsum(g_phys_corr, axis=1) * seq.system.gamma * dt
    k_log_corr = R_inv @ k_phys_corr
    
    # Reset baseline offsets to the raw trajectory positions
    k_log_corr[0, :] += (k_interleaf[0, 0] - k_log_corr[0, 0])
    k_log_corr[1, :] += (k_interleaf[1, 0] - k_log_corr[1, 0])
    
    # Store corrected profile paths
    kx_corrected[idx_start:idx_end] = k_log_corr[0, :]
    ky_corrected[idx_start:idx_end] = k_log_corr[1, :]
    
    fig, ax = plt.subplots()
    ax.plot(kx_nominal[idx_start:idx_end], ky_nominal[idx_start:idx_end], 'k-', label='Nominal')
    ax.plot(kx_corrected[idx_start:idx_end], ky_corrected[idx_start:idx_end], 'r--', label='Corrected')
    ax.legend()