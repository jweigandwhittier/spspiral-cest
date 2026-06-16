#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:06:15 2026

@author: jonah
"""
import pickle
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the measured waveforms
measured = np.load("girf_measured.npy")
N_samples = measured.shape[-1]

# Now, load the nominal waveforms
seqfile = "../sequences/girf/phantom_girf_v1_4av.seq"
seq = pp.Sequence()
seq.read(seqfile)

gamma = seq.system.gamma
dt_grad = seq.system.grad_raster_time
dt_adc = 4e-6

pfile = "../sequences/girf/phantom_girf_v1_4av.pickle"
with open(pfile, 'rb') as file:
    info = pickle.load(file)
    
N_average = 4
N_axis = 3
N_waves = 12
N_polarity = 3
N_slices = 5

adc_time = np.arange(N_samples) * dt_adc + 1e-3

nominal = np.zeros([N_waves, N_samples])
 
block_offset = 4
WaveJump = 90  

for i in range(N_waves):
    block = seq.get_block(block_offset + WaveJump * i)
    waveform = block.gx.waveform
    grad_time = np.linspace(0, waveform.shape[0], waveform.shape[0]) * seq.system.grad_raster_time + block.gx.delay
    waveform_interp = np.interp(adc_time, grad_time, -2 * np.pi * waveform, 0, 0)
    nominal[i] = waveform_interp
    plt.figure()
    plt.plot(adc_time, nominal[i], 'k-', lw=1, label='Nominal')   
    plt.plot(adc_time, measured[1,2,i],'r--', lw=0.4, label='Measured')
    plt.xlim([0,0.05])
    plt.legend()

# Now compute the GIRF
nom_spect = np.zeros([N_waves, N_samples], dtype=complex)
meas_spect = np.zeros([3, N_waves, N_samples], dtype=complex)
freqs_hz = np.fft.fftshift(np.fft.fftfreq(N_samples, d=seq.gradient_raster_time))

for i in range(N_waves):
    nom_spect[i] = np.fft.fftshift(np.fft.fft(nominal[i]))
    for ax in range(3):
        meas_spect[ax,i] = np.fft.fftshift(np.fft.fft(measured[1,ax,i]))

girf = np.zeros([3, N_samples], dtype=complex)
inspect_meanmag = np.abs(nom_spect).sum(axis=(0))

for ax in range(3):
    for wave in range(N_waves):
        girf[ax] = girf[ax] + meas_spect[ax, wave]/nom_spect[wave] * np.abs(nom_spect[wave])
    girf[ax] /= inspect_meanmag

# Plot GIRFs (per axis)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 150,
})

colors = ['#378ADD', '#1D9E75', '#D85A30']
labels = ['x', 'y', 'z']

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for i in range(3):
    axs[0].plot(freqs_hz, np.abs(girf[i]), color=colors[i], lw=1.5, label=f'{labels[i]}-axis')
    axs[1].plot(freqs_hz, np.angle(girf[i]), color=colors[i], lw=1.5, label=f'{labels[i]}-axis')

for ax, ylabel in zip(axs, ['Magnitude', 'Phase (rad)']):
    ax.set_xlim([-5e3, 5e3])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}k' if abs(x) >= 1e3 else f'{x:.0f}'))

fig.suptitle('GIRF\nSiemens Prisma, UC Berkeley', fontsize=12, fontweight='normal', y=1.02)
fig.tight_layout()
plt.show()
        
#%%
np.savez('girf_data.npz', girf_complex=girf, freqs=freqs_hz)
    

    