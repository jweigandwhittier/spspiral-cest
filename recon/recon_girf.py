#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:04:21 2026

@author: jonah
"""
import twixtools
import numpy as np

data = "../data/raw/meas_MID00252_FID44728_pulseq_girf.dat"
twix = twixtools.read_twix(data, verbose=True, parse_pmu=False)

N_adcpoints = 20000
N_adcsegs = 4

out = []
for mdb in twix[-1]['mdb']:
    if mdb.is_image_scan():
        out.append(mdb.data)
out = np.array(out)

# ADC is segmented because of Siemens ADC limits, join back up
out = out.transpose([1, 0, 2])
out = out.reshape([out.shape[0], -1, out.shape[2]*N_adcsegs])
out = out.transpose([1, 0, 2])
print(f"out.shape = {out.shape}")

#%%
# In the future these values will be better encoded in the seq file, but now need to be known
N_average = 4
N_axis = 3
N_waves = 12
N_polarity = 3
N_slices = 5
do_phase_subtraction = True

raw_data = out.reshape([N_average, N_axis, N_waves, N_polarity, N_slices, out.shape[1], out.shape[2]])
print(f"raw_data.shape = {raw_data.shape}")

# Subtract off the "waveform-off" acquisitions
# Not strictly necessary, but makes processing easier
raw_data_c = raw_data.copy()
if do_phase_subtraction:
    raw_data_c[:, :, :, 0, :, :, :] *= np.exp(-1j * np.angle(raw_data_c[:, :, :, 1, :, :, :]))
    raw_data_c[:, :, :, 2, :, :, :] *= np.exp(-1j * np.angle(raw_data_c[:, :, :, 1, :, :, :]))

# Get the accrued phase by subtracting positive and negative acquisitions
kdata_c = raw_data_c[:, :, :, (0,2), :, :, :].mean(0)
kdata_c = np.unwrap(np.angle(kdata_c[:, :, 1, :, :, :])) - np.unwrap(np.angle(kdata_c[:, :, 0, :, :, :]))
kdata_c /= 2
print(f"kdata_c.shape = {kdata_c.shape}")

# Combine coil data just by averaging, weighted by coil magnitude
# (This needs to be investigated further)
coil_mag = np.abs(raw_data).mean(axis=(0,2,3,6))[:,np.newaxis,...,np.newaxis]
kdata = (kdata_c * coil_mag).mean(3)
kdata /= coil_mag.mean(3)
print(f"kdata.shape = {kdata.shape}")

# The first couple of samples are noisy, copy the next good sample
kdata[...,:4] = kdata[...,4:5]
kdata[...,-4:] = kdata[...,-4:-3]

#%%
from scipy import signal

# ADC sampling rate
dt = 4e-6

# Slice location parameters
FOVz = 80e-3
slice_shift = 1e-3

# Filter the data (OPTIONAL)
sos = signal.bessel(3, 10000, 'lp', fs=1/dt, output='sos')
kdata_f = kdata.copy()
kdata_f = signal.sosfiltfilt(sos, kdata_f)


all_offsets = np.linspace(-FOVz/2, FOVz/2, N_slices) + slice_shift
P = np.vstack([np.ones(N_slices), all_offsets]).T

measured = np.linalg.pinv(P) @ kdata_f.transpose([2,0,1,3]).reshape([kdata_f.shape[2], -1])
measured = measured.reshape([2, kdata_f.shape[0], kdata_f.shape[1], kdata_f.shape[3]])
measured = np.pad(np.diff(measured, axis=-1) / dt, ((0, 0), (0, 0), (0, 0), (1, 0)))
print(f"measured.shape = {measured.shape}")

#%%
np.save('../girf/girf_data.npy')
