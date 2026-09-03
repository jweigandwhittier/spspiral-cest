#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:34:46 2026

@author: jonah
"""
from utils import extract_raw

# ========================================== #
# Recon variables
# ========================================== #
# Filenames
DATA_FN = "/Volumes/JWW-BIC/continuous_spiral/7302027_1615/meas_MID00047_FID139555_spsp__2ppm.dat"
SEQ_FN = "/Volumes/JWW-BIC/continuous_spiral/continuous_spiral_gauss_75p0_ppm.seq"

# Plotting flags
PLOT_BURNIN_FIT = True
PLOT_COIL_MAPS = True
PLOT_KSPACE_COVERAGE = True
SAVE_GIF = True

# Retrospective gating parameters
CARD_BAND = (0.8, 2.0) # [Hz], frequency band for PC scoring
N_BINS = 8 # Number of bins for retrospective cardiac gating (i.e, cine frames)
K_NAV = 8 # Readout samples for retrospective gating (from spiral center)
N_HARM = 6 # Fourier harmonics for golden-angle-lock removal
N_BURNIN_FALLBACK = 40  # Used only if the automatic exponential fit below fails
burnin_tol = 0.02 # Fit settles to within this fractional amplitude of asymptote before being called "steady state"

# For coil maps 
SMAP_KCUTOFF_FRAC = 0.25 # Fraction of kmax used for the low-res calibration region

# Gradient delays per PHYSICAL axis from GIRF measurement
# Convention: G_actual(t) = G_nominal(t - tau).
tau_phys = {'X': 2.01e-6, 'Y': 4.55e-6, 'Z': 7.03e-6}  # seconds; UCSF China Basin MR8

#%% Step 1: Extract raw data and sequence parameters 
# Extract k-space per vendor
meas, ksp, timestamps, vendor = extract_raw.extract_kspace(DATA_FN)
n_total_acq, n_coils, n_samples = ksp.shape

# Extract sequence (parameters + trajectory)
seq = extract_raw.load_seq(SEQ_FN)
defs = seq.definitions
k_traj_adc, _, _, _, t_adc = seq.calculate_kspace(trajectory_delay=0)

# Get TR and frequency
tr = defs['TR']
fs = 1 / tr

# Get nominal matrix size and FOV
nx = int(defs['Nx'])
fov = defs['FOV'][0]
pixel_size = fov / nx
im_size = (nx, nx)

#%% Step 2: Vendor specific physio extraction
import numpy as np

if meas is not None: # Siemens
    try:
        pmu = meas['pmu']
        puls_sig = pmu.signal['PULS']
        puls_ts = pmu.get_time('PULS')
        puls_trig_ts = pmu.timestamp_trigger['PULS'][pmu.trigger['PULS']] * 2.5e-3
        puls_at_acq = np.array([pmu.get_signal('PULS', ts / 2.5e-3) for ts in timestamps])
        nan_mask = np.isnan(puls_at_acq)
        if nan_mask.any():
            valid_idx = ~nan_mask
            puls_at_acq[nan_mask] = np.interp(np.flatnonzero(nan_mask),
                                               np.flatnonzero(valid_idx),
                                               puls_at_acq[valid_idx])
        puls_period = np.diff(puls_trig_ts)
        HAVE_PULS = True
        print(f"PULS trigger period: median {np.median(puls_period):.3f} s "
              f"({60/np.median(puls_period):.0f} bpm), n_triggers={len(puls_trig_ts)}")
    except (KeyError, AttributeError) as e:
        HAVE_PULS = False
        puls_at_acq = None
        puls_trig_ts = None
        print(f"no PULS/PMU trace available ({e}); proceeding fully self-gated")

# else: # GE

#%% Step 3: Coil-covariance navigator, golden-angle-lock removal, PCA
from scipy.signal import detrend
from utils import recon

nav_data = ksp[:, :, :K_NAV]
nav_mag = np.mean(np.abs(nav_data), axis=2)
nav_detrend = detrend(nav_mag, axis=0, type='linear')

n_burnin, _ = recon.estimate_burnin(nav_mag, tol=burnin_tol, plot=PLOT_BURNIN_FIT)
if n_burnin is None:
    n_burnin = N_BURNIN_FALLBACK
    print(f"using fallback n_burnin={n_burnin}")
    
golden_angle = np.pi * (3 - np.sqrt(5))
theta = np.mod(np.arange(nav_detrend.shape[0]) * golden_angle, 2 * np.pi)
design = np.column_stack(
    [np.ones_like(theta)] +
    [f(k * theta) for k in range(1, N_HARM + 1) for f in (np.cos, np.sin)]
)
coef, *_ = np.linalg.lstsq(design, nav_detrend, rcond=None)
nav_detrend = nav_detrend - design @ coef  # remove golden-angle-locked component

U, S, Vt = np.linalg.svd(nav_detrend - nav_detrend.mean(axis=0), full_matrices=False)
pcs = U * S
print(f"variance explained by first 5 PCs: {(S[:5]**2 / np.sum(S**2))}")

#%% Step 4: Pick the cardiac gating PC and compare with pulse signal if it exists
from scipy.signal import welch
from scipy.stats import rankdata

n_pc_check = min(8, pcs.shape[1])
card_frac_arr, ac_peak_arr, ac_lag_arr = (np.zeros(n_pc_check) for _ in range(3))
for i in range(n_pc_check):
    x = pcs[:, i]
    f, Pxx = welch(x, fs=fs, nperseg=min(128, len(x)))
    total_power = np.trapz(Pxx, f)
    band_mask = (f >= CARD_BAND[0]) & (f <= CARD_BAND[1])
    card_frac_arr[i] = np.trapz(Pxx[band_mask], f[band_mask]) / total_power
    ac = np.correlate(x - x.mean(), x - x.mean(), mode='full')
    ac = ac[len(ac)//2:] / ac[len(ac)//2]
    search = ac[3:min(len(ac), int(fs * 3))]
    ac_peak_arr[i] = np.max(search)
    ac_lag_arr[i] = 3 + np.argmax(search)   # lag in TRs -> a within-PC period estimate

if HAVE_PULS:
    puls_corr_arr = np.array([np.corrcoef(detrend(pcs[:, i]), detrend(puls_at_acq))[0, 1]
                               for i in range(n_pc_check)])
    composite = (rankdata(card_frac_arr) + rankdata(ac_peak_arr) + rankdata(np.abs(puls_corr_arr))) / 3
else:
    puls_corr_arr = np.full(n_pc_check, np.nan)
    composite = (rankdata(card_frac_arr) + rankdata(ac_peak_arr)) / 2

best_pc = int(np.argmax(composite))

# %% Step 5: Extract cardiac phase / self-gate triggers from the selected PC
from scipy.signal import butter, filtfilt, find_peaks, hilbert

x = pcs[:, best_pc]

if HAVE_PULS:
    f_card = 1.0 / np.median(puls_period)
else:
    f_psd, Pxx_best = welch(x, fs=fs, nperseg=min(256, len(x)))
    band_mask = (f_psd >= CARD_BAND[0]) & (f_psd <= CARD_BAND[1])
    f_card_psd = f_psd[band_mask][np.argmax(Pxx_best[band_mask])]
    f_card_ac = fs / ac_lag_arr[best_pc]
    print(f"self-gated f_card estimates: PSD-peak={f_card_psd:.3f} Hz "
          f"({60*f_card_psd:.0f} bpm), autocorr-lag={f_card_ac:.3f} Hz "
          f"({60*f_card_ac:.0f} bpm)")
    f_card = 0.5 * (f_card_psd + f_card_ac)
    print(f"using f_card={f_card:.3f} Hz ({60*f_card:.0f} bpm) for bandpass center")

bw = 0.5
min_distance = int(0.5 * fs)

b, a = butter(4, [max(f_card - bw, 0.1), f_card + bw], btype='band', fs=fs)
x_filt = filtfilt(b, a, x - x.mean())
best_phase = np.angle(hilbert(x_filt))
peak_idx, _ = find_peaks(x_filt, distance=min_distance)
trig_ts = timestamps[peak_idx]

# Sanity check
ihr_period = np.diff(trig_ts)
f_card_refined = 1.0 / np.median(ihr_period)
print(f"self-gate: n_triggers={len(trig_ts)}, IHR median={60*f_card_refined:.0f} bpm "
      f"(cv={np.std(ihr_period)/np.median(ihr_period):.2f})")

if not HAVE_PULS and abs(f_card_refined - f_card) / f_card > 0.15:
    print("refining bandpass center from detected triggers and re-filtering")
    f_card = f_card_refined
    b, a = butter(4, [max(f_card - bw, 0.1), f_card + bw], btype='band', fs=fs)
    x_filt = filtfilt(b, a, x - x.mean())
    best_phase = np.angle(hilbert(x_filt))
    peak_idx, _ = find_peaks(x_filt, distance=min_distance)
    trig_ts = timestamps[peak_idx]
    ihr_period = np.diff(trig_ts)
    print(f"after refine: n_triggers={len(trig_ts)}, IHR median="
          f"{60/np.median(ihr_period):.0f} bpm")

if HAVE_PULS:
    matched_err = np.array([t - puls_trig_ts[np.argmin(np.abs(puls_trig_ts - t))] for t in trig_ts])
    print(f"PC{best_pc+1}: n_selfgate={len(trig_ts)}, n_puls={len(puls_trig_ts)}, "
          f"err mean={matched_err.mean()*1e3:.1f} ms, std={matched_err.std()*1e3:.1f} ms")
else:
    print(f"PC{best_pc+1}: n_selfgate={len(trig_ts)} (no PULS reference -- "
          f"IHR cv above and the burn-in/PSD plots are your only sanity checks)")

valid_phase = best_phase[n_burnin:]
valid_data = ksp[n_burnin:]

#%% Step 6: Gradient delay correction and steady-state cutoff
total_samples = n_total_acq * n_samples
kx_full_raw = k_traj_adc[0, :total_samples].reshape(n_total_acq, n_samples)
ky_full_raw = k_traj_adc[1, :total_samples].reshape(n_total_acq, n_samples)
kz_full_raw = (k_traj_adc[2, :total_samples] if k_traj_adc.shape[0] >= 3
               else np.zeros(total_samples)).reshape(n_total_acq, n_samples)
t_adc_full = t_adc[:total_samples]

if meas is not None: # Get geometry from TWIX if available
    geom = meas['geometry'][0]
    R_log2phys = np.column_stack([geom.rps_to_xyz() @ e for e in np.eye(3)])
    print("R (logical->physical, DCS):\n", R_log2phys)
    
    tau_vec = np.array([tau_phys['X'], tau_phys['Y'], tau_phys['Z']])
    k_logical_full = np.stack([kx_full_raw.ravel(), ky_full_raw.ravel(), kz_full_raw.ravel()])
    k_corrected_full = recon.apply_axis_delay_correction(t_adc_full, k_logical_full, R_log2phys, tau_vec)

    kx_full = k_corrected_full[0].reshape(n_total_acq, n_samples)
    ky_full = k_corrected_full[1].reshape(n_total_acq, n_samples)
else: # GE -- no geometry source wired up yet, skip axis-delay correction
    print("GE data: R_log2phys not available, skipping axis-delay correction "
          "(using nominal trajectory)")
    kx_full = kx_full_raw
    ky_full = ky_full_raw

kx_valid = kx_full[n_burnin:]
ky_valid = ky_full[n_burnin:]

#%% Step 6: Get coil sensitivity maps
smaps = recon.estimate_coil_sensitivities_espirit(kx_valid, ky_valid, valid_data, im_size,
                                             pixel_size, n_coils,
                                             k_cutoff_frac=SMAP_KCUTOFF_FRAC,
                                             calib_width=24, thresh=0.02,
                                             plot=PLOT_COIL_MAPS)

#%% Step 7: Reconstruct CINE
bin_centers = np.linspace(-np.pi, np.pi, N_BINS, endpoint=False)
binned_indices = recon.build_binned_indices(valid_phase, n_samples, N_BINS, bin_centers)
recon_stack = recon.xdgrasp_reconstruct_cine(kx_valid, ky_valid, valid_data, binned_indices, N_BINS, n_coils, im_size, pixel_size, nx, smaps)

#%% Step 8: Plot images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes_flat = axes.flatten()
vmax = np.percentile(recon_stack, 99.5)
for b in range(N_BINS):
    axes_flat[b].imshow(recon_stack[b], cmap='gray', vmin=0, vmax=vmax)
    axes_flat[b].set_title(f"Phase {b+1}")
    axes_flat[b].axis('off')
plt.tight_layout()
plt.show()

#%% Step 9: Save raw data
import re

# Extract filename
match = re.search(r'(MID\d+_FID\d+_[^/\\]+?)(?:\.\w+)?$', DATA_FN)
extracted_id = match.group(1) if match else "Unknown_ID"
print(f"Extracted Scan ID: {extracted_id}")

np.save(f"data/recon/{vendor}/{extracted_id}", recon_stack)


#%% Step 10: Export CINE to GIF
if SAVE_GIF:
    from PIL import Image
    
    vmax = np.percentile(recon_stack, 99.5)
    norm_stack = np.clip(recon_stack / vmax * 255, 0, 255).astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in norm_stack]
    
    gif_filename = f"{extracted_id}_cardiac_cine.gif"
    output_path = f"gifs/{gif_filename}"
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=150, loop=0)
    print(f"GIF saved successfully to: {output_path}")

