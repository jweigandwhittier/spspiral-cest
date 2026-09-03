#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:08:17 2026

@author: jonah
"""
import os
import sys
from pathlib import Path
import numpy as np
import pypulseq as pp
from seqeyes import seqeyes
from packaging.version import Version
from utils import vds, spsp, prep_pge2

# ========================================== #
# Hardware limits
# This should be adjusted based on real hardware limits AND based on simulated PNS
# Both Siemens (through Seqeyes) and GE (as part of file conversion) run PNS simulations
# Set the max slew rate such that PNS limits are not violated 
# ========================================== #
DESIGN_MAX_GRAD = 30   # mT/m
DESIGN_MAX_SLEW = 78   # mT/m/ms

# Siemens 3T
system = pp.opts.Opts(
    max_grad = DESIGN_MAX_GRAD,
    grad_unit = 'mT/m',
    max_slew = DESIGN_MAX_SLEW,
    slew_unit = 'mT/m/ms',
    rf_ringdown_time = 60e-6, 
    rf_dead_time = 100e-6,
    adc_dead_time = 20e-6, 
    adc_raster_time = 10e-6,
    B0 = 3.00
    )

# GE 3T 
system_ge = pp.opts.Opts(
    max_grad = DESIGN_MAX_GRAD,
    grad_unit = 'mT/m',
    max_slew = DESIGN_MAX_SLEW,
    slew_unit = 'mT/m/ms',
    rf_ringdown_time = 60e-6, 
    rf_dead_time = 100e-6,
    adc_dead_time = 20e-6,
    grad_raster_time = 4e-6, # This is important
    adc_raster_time = 2e-6, # This is also important
    rf_raster_time = 2e-6, # Finally, need this
    block_duration_raster = 4e-6,
    B0 = 3.00
    )

# ========================================== #
# Sequence flags
# ========================================== #
# Flags for sequence writing
FLAG_GE = True # Write sequence for GE?
FLAG_SIM = False # Also write a sequence for simulation with BMCTool?

# ZSPEC: when True, ONLY the ZSPEC offsets are written (the offsets_ppm
# loop below is skipped entirely). On Siemens this is ONE combined
# sequence sweeping zspec_offsets_ppm with a recovery delay dropped
# between offsets. On GE this is a SEPARATE .seq file per zspec_offsets_ppm.
FLAG_ZSPEC = False

# Site selection - drives conversion wait params + deployment locations, see prep_pge2.py
SITE = "berkeley" # "site1" or "site2", edit this for your own sites (see prep_pge2_example.py)
FLAG_DEPLOY = True # Also generate/run the pge2 deploy script?

# SPSP pulse flags
FLAG_SPSP = False # Use SPSP pulses?
FLAG_GENERIC = True # If using SPSP pulses, use generic pulse?

# Plotting and visualization flags 
FLAG_SEQEYES = False
FLAG_PLOT = False # Also for plotting with GE

# Other flags
FLAG_TEST_REPORT = False

# ==========================================
# Run PyPulseq version check
# ==========================================
# Get Pulseq version
pypulseq_ver = Version(pp.__version__)
pypulseq_base_ver = pypulseq_ver.base_version

# Rotation support is available only in the rotext-enabled 1.5.1 build
USE_ROTATION = False

if FLAG_GE:
    # GE is intentionally strict
    if Version(pypulseq_base_ver) < Version("1.5.0"):
        sys.exit(
            f"GE target requires PyPulseq 1.5.0 with rotext "
            f"(detected: {pp.__version__})."
        )
    if not hasattr(pp, 'make_rotation'):
        sys.exit(
            f"GE target requires the rotext-enabled PyPulseq 1.5.1 fork "
            f"(pp.make_rotation not found; detected pp.__version__="
            f"{pp.__version__}). "
            f"Install: pip install git+https://github.com/mcencini/pypulseq"
        )
    USE_ROTATION = True

    # GE uses the GE-specific hardware limits.
    system = system_ge
    # Import once
    import matlab.engine
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab("-java")
    # Force light mode
    eng.eval("set(groot, 'DefaultFigureColor', [1, 1, 1]);", nargout=0)
    eng.eval("set(groot, 'DefaultAxesColor', [1, 1, 1]);", nargout=0)
    eng.eval("set(groot, 'DefaultAxesXColor', [0, 0, 0]);", nargout=0)
    eng.eval("set(groot, 'DefaultAxesYColor', [0, 0, 0]);", nargout=0)
    eng.eval("set(groot, 'DefaultTextColor', [0, 0, 0]);", nargout=0)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Adding {script_dir} to MATLAB path")
    eng.addpath(script_dir, nargout=0)

else:
    # Siemens supports either:
    #   1.4.2 -> traditional/manual rotation
    #   >=1.5.0 -> rotext rotation
    if pypulseq_base_ver != "1.4.2" and Version(pypulseq_base_ver) < Version("1.5.0"):
        sys.exit(
            f"Siemens target supports PyPulseq 1.4.2 or >=1.5.0 "
            f"(detected: {pp.__version__})."
        )
    if Version(pypulseq_base_ver) >= Version("1.5.0"):
        if not hasattr(pp, 'make_rotation'):
            sys.exit(
                f"Siemens PyPulseq >=1.5.0 requires the rotext-enabled build "
                f"(pp.make_rotation not found; detected pp.__version__="
                f"{pp.__version__}). "
                f"Install: pip install git+https://github.com/mcencini/pypulseq"
            )
        USE_ROTATION = True
    # Siemens always uses Siemens hardware limits.
    system = system
    print(
        f"ALERT: Siemens sequence is being written with "
        f"PyPulseq {pp.__version__}."
    )
    if USE_ROTATION:
        print("         Using rotext gradient rotation.")
    else:
        print("         Using traditional/manual gradient rotation.")
        
# ========================================== #
# Sequence parameters
# ========================================== #
# General parameters
seq_dur = 12 # Total desired duration of the sequence [s]

# CEST parameters
b1p = 1.20 # Saturation pulse peak B1 amplitude [uT]
tp = 36e-3 # Saturation pulse duration [s]
spoil_rise_time = 1e-3 # CEST spoiler gradient rise time [s]
spoil_dur = 6.5e-3 # CEST spoiler gradient duration [s]
spoil_amp = 0.8 * system.max_grad # CEST spoiler gradient amplitude
offsets_ppm = [75.00, 2.00, -2.00]
tissue_t1 = 1.2 # Assumed tissue/phantom T1 [s]; used for ZSPEC inter-offset recovery delay

# Readout parameters
fov = 150e-3 # Field of view [m] (try 300 mm for patients, 150 mm for phantom)
nx = 128 # Matrix size
n_interleaves = 12 # Number of spiral interleaves (for "fully" sampled k-space minus VD)
slice_thickness = 8e-3 # Slice thickness [m]
adc_dwell = 10e-6 # Hardcode this to preserve readout bandwidth (BWPP --> SNR)
rf_spoiling_inc = 117 # [deg]

# GE specific parameters
pislquant = 4 if FLAG_GE else 0

# Constants 
gamma_hz = system.gamma * 1e-6
freq = gamma_hz * system.B0

# ========================================== #
# Load maps for tailored SPSP pulses
# ========================================== #
b1_map = np.load('generic_spsp/generic_b1.npy') # Or paste DICOM file path directly
wasabi_seq_filename = 'generic_spsp/generic_b1.seq' # Put 'dicom' here if the map is from a DICOM
mask = None # Keep this
if FLAG_GENERIC: # Don't touch this
    b1_map = np.load('generic_spsp/generic_b1.npy')
    wasabi_seq_filename = 'generic_spsp/generic_b1.seq'
    mask = np.load('generic_spsp/mask.npy')
    
# ========================================== #
# Prepare sequence elements
# ========================================== #
# Calculate raster aligned timing
align_to_raster = lambda t: np.ceil(t / system.grad_raster_time) * system.grad_raster_time
tp = align_to_raster(tp)

# Z-spectrum (ZSPEC) offsets + recovery delay (used only if FLAG_ZSPEC).
_zspec_part0 = np.array([75.00])
_zspec_part1 = np.arange(-10, -5, 1)
_zspec_part2 = np.arange(-5, 5.01, 0.2)
_zspec_part3 = np.arange(6, 10.01, 1)
zspec_offsets_ppm = np.round(np.concatenate((_zspec_part0, _zspec_part1, _zspec_part2, _zspec_part3)), 2)
zspec_recovery_delay_s = align_to_raster(5 * tissue_t1)

# Write dummy excitation
rf_exc, gz_exc, gzr = pp.make_sinc_pulse(
    flip_angle = np.deg2rad(12.0), # Based on sim for T1 = 1.3 s
    duration = 1e-3, # [s]
    slice_thickness = slice_thickness,
    apodization = 0.5,
    time_bw_product = 8.0,
    system = system,
    return_gz = True,
    delay = system.rf_dead_time,
    use = 'excitation')

# Spiral parameters
resolution = fov / nx
max_kspace_radius = 0.5 / (resolution)
sampling_period = system.grad_raster_time
fov_coefficients = [fov, -1/4 * fov]

# ========================================== #
# Rotation design margin (readout AND rewinder)
# ========================================== #
SPIRAL_DESIGN_MARGIN_DIVISOR = np.sqrt(2)
import copy
system_rotation_design = copy.deepcopy(system)
system_rotation_design.max_slew = system.max_slew / SPIRAL_DESIGN_MARGIN_DIVISOR
print(f"Rotation design slew target: {1/SPIRAL_DESIGN_MARGIN_DIVISOR*100:.1f}% of hardware max_slew "
      f"(divisor={SPIRAL_DESIGN_MARGIN_DIVISOR}, applied to readout AND rewinder)")

# Write spiral
(k, g, s, timing, r, theta) = vds.variable_density_spiral_trajectory(
        system=system_rotation_design,
        sampling_period=sampling_period,
        n_interleaves=n_interleaves,
        fov_coefficients=fov_coefficients,
        max_kspace_radius=max_kspace_radius
    )

# Number of samples in the gradient waveform
num_grad_samples = np.shape(g)[0]

# Calculate ADC
active_grad_time = num_grad_samples * system.grad_raster_time
sampling_time = active_grad_time - 2 * system.adc_dead_time
max_samples = int(np.floor(sampling_time / adc_dwell))
adc_total_samples = max_samples - (max_samples % 4)
assert adc_total_samples <= 8192, 'ADC samples exceed maximum value of 8192.'
adc = pp.make_adc(num_samples=adc_total_samples, dwell=adc_dwell, delay=system.adc_dead_time, system=system)
print(f'ADC Samples: {adc_total_samples}')
# Special "dummy" ADC for simulation
if FLAG_SIM:
    adc_sim = pp.make_adc(num_samples=1, dwell=system.adc_raster_time, delay=system.adc_dead_time, system=system)
    
# Unrotated reference gradients
gx_ro_ref = pp.make_arbitrary_grad(channel='x', waveform=np.real(g), delay=adc.delay, system=system)
gy_ro_ref = pp.make_arbitrary_grad(channel='y', waveform=np.imag(g), delay=adc.delay, system=system)

gx_rew_ref, *_ = pp.make_extended_trapezoid_area(
    area=-gx_ro_ref.area, channel='x', grad_start=gx_ro_ref.last, grad_end=0, system=system_rotation_design)
gy_rew_ref, *_ = pp.make_extended_trapezoid_area(
    area=-gy_ro_ref.area, channel='y', grad_start=gy_ro_ref.last, grad_end=0, system=system_rotation_design)

# Write spoiler
n_cycles = 4
spoil_area = n_cycles / slice_thickness - gz_exc.area / 2

gz_spoil = pp.make_trapezoid(channel='z', area=spoil_area, system=system_rotation_design)
    
# Placeholder saturation pulses
if FLAG_SPSP:
    spsp_objects = spsp.calc_spsp(b1_map, wasabi_seq_filename, tp, system, mask)
    spsp_grad_x = spsp_objects['full_gx']
    spsp_grad_y = spsp_objects['full_gy']
    spsp_rf_shape = spsp_objects['full_rf']
    
placeholder_offset = offsets_ppm[0] * freq
flip_calib_duration = spsp_objects['achieved_duration'] if FLAG_SPSP else tp
sat_pulse = pp.make_gauss_pulse(flip_angle=np.pi, 
                                duration=flip_calib_duration, 
                                time_bw_product=0.2,
                                apodization=0.5, 
                                delay=100e-6, 
                                freq_offset=placeholder_offset,
                                system=system,
                                use='preparation')
target_peak_hz = b1p * gamma_hz
current_peak_hz = np.max(np.abs(sat_pulse.signal))
sat_pulse.signal *= (target_peak_hz / current_peak_hz)
dt = system.rf_raster_time
total_flip_angle = np.abs(np.sum(sat_pulse.signal)) * dt * 2 * np.pi
if FLAG_SPSP:
    spsp_pulse = pp.make_arbitrary_rf(spsp_rf_shape, 
                                      flip_angle=total_flip_angle, 
                                      dwell=system.rf_raster_time,
                                      delay=system.rf_dead_time,
                                      freq_offset=placeholder_offset,
                                      system=system,
                                      use='preparation')
    
# CEST spoilers
gx_spoil_cest, gy_spoil_cest, gz_spoil_cest = [
    pp.make_trapezoid(channel=c, system=system, amplitude=spoil_amp,
                      duration=spoil_dur, rise_time=spoil_rise_time)
    for c in ["x", "y", "z"]
]
    
# ========================================== #
# Calculate number of TRs
# ========================================== #
# CEST prep
prep_duration = pp.calc_duration(sat_pulse) + pp.calc_duration(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)
print(f"CEST Prep Duration: {prep_duration * 1000:.2f} ms")
# Excitation
exc_duration = pp.calc_duration(rf_exc, gz_exc) + pp.calc_duration(gzr)
print(f"Excitation Duration: {exc_duration * 1000:.2f} ms")
# Readout (+ rewinder/spoiler)
readout_duration = pp.calc_duration(gx_ro_ref, gy_ro_ref)
print(f"Readout Duration: {readout_duration * 1000:.2f} ms")
max_rewinder_duration = max(pp.calc_duration(gx_rew_ref), pp.calc_duration(gy_rew_ref),
                             pp.calc_duration(gz_spoil))
print(f"Rewinder Duration: {max_rewinder_duration * 1000:.2f} ms")
# Total TR
tr = prep_duration + exc_duration + readout_duration + max_rewinder_duration
tr = align_to_raster(tr)
n_trs = int(np.floor(seq_dur / tr))
print(f"Total TR: {tr * 1000:.2f} ms")
print(f"Number of TRs: {n_trs}")
golden_angle = np.pi * (3 - np.sqrt(5))
angles = np.mod(np.arange(n_trs) * golden_angle, 2 * np.pi)

# ========================================== #
# Build sequence
# ========================================== #
# Define labels for GE once (it's easy here!)
calib_label = pp.make_label('TRID', 'SET', 1)
prep_label = pp.make_label('TRID', 'SET', 2)
exc_label = pp.make_label('TRID', 'SET', 3)
ro_label = pp.make_label('TRID', 'SET', 4)

output_dir = 'sequences/ge' if FLAG_GE else 'sequences/siemens'
sim_output_dir = 'sequences/sim'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(sim_output_dir, exist_ok=True)

deploy_batch_tasks = []  


def write_offset_tr_train(seqs, seq_scan, offset_ppm, pislquant):
    """
    Appends one continuous golden-angle TR train (n_trs TRs, using the
    module-level `angles`) at `offset_ppm` to every sequence in `seqs`.

    `seq_scan` must be the entry of `seqs` that gets the real spiral
    readout; any other entries in `seqs` are treated as the BMCSim
    simulation sequence and get a dummy ADC instead. Reuses the sat pulse,
    spiral gradients, spoilers, and labels set up above at module scope.

    Called once per discrete offset (offsets_ppm) to build a whole
    standalone sequence, once per offset inside the combined Siemens ZSPEC
    sequence (with a labeled recovery delay dropped in between calls), and
    once per offset to build each standalone GE ZSPEC sequence.
    """
    offsets_hz = offset_ppm * freq
    sat_pulse.freq_offset = offsets_hz
    if FLAG_SPSP:
        spsp_pulse.freq_offset = offsets_hz

    rf_phase = 0
    rf_inc = 0
    
    # Receive gain calibration for GE systems only
    for _ in range(pislquant):
        rf_exc.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi

        for seq in seqs:
            seq.add_block(rf_exc, gz_exc, calib_label)
            seq.add_block(gzr)
            if seq is seq_scan:
                seq.add_block(gx_ro_ref, gy_ro_ref, adc)
                seq.add_block(gx_rew_ref, gy_rew_ref, gz_spoil)
            else:
                seq.add_block(adc_sim)
                dummy_delay = readout_duration - pp.calc_duration(adc_sim)
                seq.add_block(pp.make_delay(dummy_delay))
                seq.add_block(pp.make_delay(max_rewinder_duration))

        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

    # Real TRs
    for n in range(n_trs):
        theta = angles[n]
        rf_exc.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi

        if not USE_ROTATION:
            c, s = np.cos(theta), np.sin(theta)
            gxw = np.real(g) * c - np.imag(g) * s
            gyw = np.real(g) * s + np.imag(g) * c

            gx_ro = pp.make_arbitrary_grad(channel='x', waveform=gxw, delay=adc.delay, system=system)
            gy_ro = pp.make_arbitrary_grad(channel='y', waveform=gyw, delay=adc.delay, system=system)

            gx_rew, *_ = pp.make_extended_trapezoid_area(
                area=-gx_ro.area, channel='x', grad_start=gx_ro.last, grad_end=0, system=system)
            gy_rew, *_ = pp.make_extended_trapezoid_area(
                area=-gy_ro.area, channel='y', grad_start=gy_ro.last, grad_end=0, system=system)

            rewinder_duration_n = max(pp.calc_duration(gx_rew), pp.calc_duration(gy_rew),
                                       pp.calc_duration(gz_spoil))
            pad = max_rewinder_duration - rewinder_duration_n
            pad = np.ceil(pad / system.grad_raster_time) * system.grad_raster_time if pad > 0 else 0

        for seq in seqs:
            # Sat pulse + CEST spoiler
            if FLAG_SPSP:
                seq.add_block(spsp_pulse, prep_label)
            else:
                seq.add_block(sat_pulse, prep_label)
            seq.add_block(gx_spoil_cest, gy_spoil_cest, gz_spoil_cest)

            # Excitation
            seq.add_block(rf_exc, gz_exc, exc_label)
            seq.add_block(gzr)
            if seq is seq_scan:
                # Real spiral readout
                # 1.5.1 with rotation extension
                if USE_ROTATION:
                    rot = pp.make_rotation(theta)
                    seq.add_block(rot, gx_ro_ref, gy_ro_ref, adc, ro_label)
                    seq.add_block(rot, gx_rew_ref, gy_rew_ref, gz_spoil)
                # 1.4.2 without rotation extension
                else:
                    seq.add_block(gx_ro, gy_ro, adc, ro_label)
                    seq.add_block(gx_rew, gy_rew, gz_spoil)
                    if pad > 0:
                        seq.add_block(pp.make_delay(pad))
            else:
                seq.add_block(adc_sim)
                dummy_delay = readout_duration - pp.calc_duration(adc_sim)
                seq.add_block(pp.make_delay(dummy_delay))
                seq.add_block(pp.make_delay(max_rewinder_duration))

        # Increment RF spoiling phase
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]


def convert_and_queue_for_ge(seq_filename, description):
    """
    Runs the .seq -> .pge2 conversion (if FLAG_GE) and, if FLAG_DEPLOY,
    queues the sequence into the shared `deploy_batch_tasks` list so
    offsets_ppm sequences AND ZSPEC sequence(s) all land in the same
    deploy_batch.sh / pulseq_scans.list at the end of the script.
    """
    if not FLAG_GE:
        return

    system_dict = {
        "maxGrad": float(system.max_grad),
        "maxSlew": float(system.max_slew),
        "gamma": float(system.gamma),

        # Timing
        "gradRasterTime": float(system.grad_raster_time),
        "rfRasterTime": float(system.rf_raster_time),
        "adcRasterTime": float(system.adc_raster_time),

        # Dead/ringdown times
        "rfDeadTime": float(system.rf_dead_time),
        "rfRingdownTime": float(system.rf_ringdown_time),
        "adcDeadTime": float(system.adc_dead_time),

        "blockDurationRaster": float(system.block_duration_raster),
    }

    wait_params = prep_pge2.get_conversion_wait_params(SITE)
    psd_rf_wait = wait_params['psdRfWait'] if wait_params['psdRfWait'] is not None else []
    psd_grd_wait = wait_params['psdGrdWait'] if wait_params['psdGrdWait'] is not None else []
    coil = wait_params['coil'] if wait_params['coil'] is not None else []

    eng.convert_pge2(
        str(seq_filename), system_dict, pislquant, False,
        psd_rf_wait, psd_grd_wait, coil,
        nargout=0,
    )
    # Python handles the input safely in your 
    if FLAG_PLOT:
        input("Press Enter in this Python console to close the MATLAB plot and continue...")
    # Force the engine to close the open figure
    eng.eval("close(gcf);", nargout=0)

    if FLAG_DEPLOY:
        deploy_batch_tasks.append({
            "seq_path": Path(seq_filename),
            "description": description,
        })


def build_and_save_single_offset_seq(seq_name, offset_ppm):
    """
    Shared helper: given a seq_name and a single offset_ppm, builds a fresh
    seq_scan (+ optional seq_sim), writes one full TR train at that offset,
    checks timing, sets definitions, writes the .seq file(s), plots/seqeyes
    if requested, and converts+queues for GE deploy. Returns seq_filename.

    Used by both the regular offsets_ppm loop and the GE ZSPEC-per-offset
    branch, since they now do exactly the same thing at the sequence level.
    """
    seq_scan = pp.Sequence(system=system)
    seqs = [seq_scan]
    if FLAG_SIM:
        seq_sim = pp.Sequence(system=system)
        seqs.append(seq_sim)

    write_offset_tr_train(seqs, seq_scan, offset_ppm, pislquant)

    for seq in seqs:
        ok, error_report = seq.check_timing()
        tag = 'seq_scan' if seq is seq_scan else 'seq_sim'
        if ok:
            print(f'[{seq_name} / {tag}] Timing check passed successfully!')
            if FLAG_TEST_REPORT:
                print(seq.test_report())
        else:
            print(f'[{seq_name} / {tag}] Timing check FAILED! Error listing follows:')
            [print(e) for e in error_report]

    seq_scan.set_definition('Name', seq_name)
    seq_scan.set_definition('FOV', [fov, fov, slice_thickness])
    seq_scan.set_definition('Offset_ppm', float(offset_ppm))
    seq_scan.set_definition('N_TRs', n_trs)
    seq_scan.set_definition('TR', tr)
    seq_scan.set_definition('Nx', nx)
    seq_scan.set_definition('MaxAdcSegmentLength', adc_total_samples)
    seq_scan.set_definition('pislquant', pislquant)
    seq_scan.set_definition('RotationAngles_rad', [float(a) for a in angles])

    seq_filename = f'{output_dir}/{seq_name}.seq'
    seq_scan.write(seq_filename)

    if FLAG_SIM:
        seq_sim.set_definition('Name', seq_name)
        offsets_expanded = np.full(n_trs, float(offset_ppm))
        seq_sim.set_definition('offsets_ppm', offsets_expanded)
        seq_sim.set_definition('num_meas', n_trs)
        seq_sim_filename = f'{sim_output_dir}/{seq_name}.seq'
        seq_sim.write(seq_sim_filename)

    if FLAG_SEQEYES:
        seqeyes(seq_filename)

    if FLAG_PLOT:
        seq_scan.plot()

    return seq_filename


pulse_type = 'gauss' if not FLAG_SPSP else 'spsp'
scanner = '' if not FLAG_GE else '_ge'

if not FLAG_ZSPEC:
    # ========================================== #
    # Regular mode: one standalone sequence per offsets_ppm entry
    # ========================================== #
    for offset_idx, offset_ppm in enumerate(offsets_ppm):
        offset_str = str(offset_ppm).replace('.', 'p')
        seq_name = f'continuous_spiral_{pulse_type}_{offset_str}_ppm{scanner}'

        seq_filename = build_and_save_single_offset_seq(seq_name, offset_ppm)

        # ========================================== #
        # Conversion for GE (.seq to .pge2) + batch deploy queueing
        # ========================================== #
        convert_and_queue_for_ge(seq_filename, f"{offset_ppm} ppm")

elif FLAG_GE:
    # ========================================== #
    # ZSPEC on GE: ignore offsets_ppm entirely. Write ONE standalone
    # sequence PER zspec_offsets_ppm entry (same structure as the regular
    # loop above, just driven by zspec_offsets_ppm), each converted and
    # queued for deploy individually.
    # ========================================== #
    for zidx, offset_ppm in enumerate(zspec_offsets_ppm):
        offset_str = str(offset_ppm).replace('.', 'p')
        seq_name = f'continuous_spiral_zspec_{pulse_type}_{offset_str}_ppm{scanner}'

        seq_filename = build_and_save_single_offset_seq(seq_name, offset_ppm)

        convert_and_queue_for_ge(seq_filename, f"zspec {offset_ppm} ppm")

else:
    # ========================================== #
    # ZSPEC on Siemens: ignore offsets_ppm entirely. ONE combined sequence
    # sweeping zspec_offsets_ppm, with a labeled recovery delay dropped
    # between offset segments.
    # ========================================== #
    seq_scan = pp.Sequence(system=system)
    seqs = [seq_scan]
    if FLAG_SIM:
        seq_sim = pp.Sequence(system=system)
        seqs.append(seq_sim)

    for zidx, offset_ppm in enumerate(zspec_offsets_ppm):
        write_offset_tr_train(seqs, seq_scan, offset_ppm, pislquant)

        is_last_offset = (zidx == len(zspec_offsets_ppm) - 1)
        if not is_last_offset:
            for seq in seqs:
                seq.add_block(pp.make_delay(zspec_recovery_delay_s))

    seq_name = f'continuous_spiral_zspec_{pulse_type}{scanner}'

    for seq in seqs:
        ok, error_report = seq.check_timing()
        tag = 'seq_scan' if seq is seq_scan else 'seq_sim'
        if ok:
            print(f'[{seq_name} / {tag}] Timing check passed successfully!')
            if FLAG_TEST_REPORT:
                print(seq.test_report())
        else:
            print(f'[{seq_name} / {tag}] Timing check FAILED! Error listing follows:')
            [print(e) for e in error_report]

    seq_scan.set_definition('Name', seq_name)
    seq_scan.set_definition('FOV', [fov, fov, slice_thickness])
    seq_scan.set_definition('Offsets_ppm', [float(o) for o in zspec_offsets_ppm])
    seq_scan.set_definition('N_TRs_per_offset', n_trs)
    seq_scan.set_definition('TR', tr)
    seq_scan.set_definition('RecoveryDelay', float(zspec_recovery_delay_s))
    seq_scan.set_definition('MaxAdcSegmentLength', adc_total_samples)
    seq_scan.set_definition('RotationAngles_rad', [float(a) for a in angles])

    seq_filename = f'{output_dir}/{seq_name}.seq'
    seq_scan.write(seq_filename)

    if FLAG_SIM:
        seq_sim.set_definition('Name', seq_name)
        offsets_expanded = np.repeat(zspec_offsets_ppm.astype(float), n_trs)
        seq_sim.set_definition('offsets_ppm', offsets_expanded)
        seq_sim.set_definition('num_meas', n_trs * len(zspec_offsets_ppm))
        seq_sim_filename = f'{sim_output_dir}/{seq_name}.seq'
        seq_sim.write(seq_sim_filename)

    if FLAG_SEQEYES:
        seqeyes(seq_filename)

    if FLAG_PLOT:
        seq_scan.plot()

    convert_and_queue_for_ge(seq_filename, "zspec")

# ========================================== #
# Write ONE deploy script for all sequences generated above
# ========================================== #
if FLAG_DEPLOY and deploy_batch_tasks:
    deploy_script_path = Path(f'{output_dir}/deploy_batch.sh')
    prep_pge2.prep_pge2_batch(
        deploy_batch_tasks,
        site=SITE,
        output_script_path=deploy_script_path,
    )
    print(f'[deploy] Generated deploy script for {len(deploy_batch_tasks)} sequence(s): {deploy_script_path}')
    print(f'[deploy] Run it manually to push to the {SITE} scanner: bash {deploy_script_path}')