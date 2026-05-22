#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 02:45:11 2026

@author: jonah
"""
import numpy as np
import pypulseq as pp
from utils import sim_cest_rf

# Load example B1 map and corresponding dummy .seq file
b1_map = np.load('data/recon/cindy_example_b1.npy')
seq_filename = 'sequences/example/cindy_b1.seq'

# Pulse length
tp = 36e-3

# System limits
sys = pp.opts.Opts(
    max_grad = 38,
    grad_unit = 'mT/m',
    max_slew = 170.068, 
    slew_unit = 'mT/m/ms',
    rf_ringdown_time = 20e-6,
    rf_dead_time = 100e-6,
    adc_dead_time = 10e-6,
    B0 = 3.0,
    )

# Creat spsp pulse and corresponding gradients
spsp_objects = sim_cest_rf.calc_spsp(b1_map, seq_filename, tp, sys, False)
spsp_rf_shape = spsp_objects['full_rf']
sat_pulse = pp.make_arbitrary_rf(spsp_rf_shape, flip_angle = np.pi, dwell = 1e-5, delay = sys.rf_dead_time)

# Initialize sequence
seq = pp.Sequence(system=sys)

# Add and plot
seq.add_block(sat_pulse, spsp_objects['full_gx'], spsp_objects['full_gy'])
seq.plot(grad_disp='mT/m')