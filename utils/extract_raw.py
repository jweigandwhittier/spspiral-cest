#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:36:47 2026

@author: jonah
"""

import twixtools
import GERecon # This must be installed separately! 
import numpy as np
import pypulseq as pp
from pathlib import Path

_DEFAULT_RASTER = {
    'GradientRasterTime': 10e-6,
    'RadiofrequencyRasterTime': 1e-6,
    'AdcRasterTime': 10e-6,
    'BlockDurationRaster': 10e-6,
}

def _peek_seq_raster_times(seq_fn):
    found = {}
    with open(seq_fn, 'r') as f:
        in_defs = False
        for line in f:
            line = line.strip()
            if line == '[DEFINITIONS]':
                in_defs = True
                continue
            if in_defs:
                if line.startswith('['):
                    break
                if not line:
                    continue
                key, *rest = line.split()
                if key in _DEFAULT_RASTER and rest:
                    found[key] = float(rest[0])
    return {**_DEFAULT_RASTER, **found}

def load_seq(seq_fn):
    raster = _peek_seq_raster_times(seq_fn)
    system = pp.opts.Opts(
        grad_raster_time=raster['GradientRasterTime'],
        rf_raster_time=raster['RadiofrequencyRasterTime'],
        adc_raster_time=raster['AdcRasterTime'],
        block_duration_raster=raster['BlockDurationRaster'],
    )
    seq = pp.Sequence(system=system)
    seq.read(seq_fn)
    return seq

def _detect_vendor(data_fn):
    suffix = Path(data_fn).suffix
    if suffix == '.dat':
        print("Detected vendor: Siemens")
        return 'siemens'
    if suffix == '.h5':
        print("Detected vendor: GE")
        return 'ge'
    raise ValueError(f"Unrecognized file type: {suffix}")

def _extract_siemens(data_fn):
    twix_data = twixtools.read_twix(data_fn, parse_pmu=True, parse_geometry=True)
    meas = twix_data[-1]
    data_list = []
    timestamp_list = []
    skip_flags = ('ACQEND', 'RTFEEDBACK', 'HPFEEDBACK', 'SYNCDATA', 'REFPHASESTABSCAN',
                  'PHASESTABSCAN', 'PHASCOR', 'NOISEADJSCAN', 'PATREFSCAN',
                  'PATREFANDIMASCAN')
    n_skipped = {}
    for mdb in meas['mdb']:
        if mdb.is_image_scan():
            data_list.append(mdb.data)
            timestamp_list.append(mdb.mdh.TimeStamp)
        else:
            matched = [f for f in skip_flags if mdb.is_flag_set(f)]
            key = ','.join(matched) if matched else 'other'
            n_skipped[key] = n_skipped.get(key, 0) + 1
    ksp = np.asarray(data_list)             
    timestamps = np.asarray(timestamp_list, dtype=float) * 2.5e-3 
    return meas, ksp, timestamps

def _extract_ge(data_fn):
    # Load archive
    archive = GERecon.Archive(data_fn)
    metadata = archive.Metadata()
    num_control = metadata.get("controlCount", 100000)
    y_res = metadata["acquiredYRes"]
    num_channels = metadata["numChannels"]
    # Allocate k-space
    ksp = None
    for i in range(num_control):
        try: 
            control = archive.NextControl()
        except RuntimeError:
            break
        if control.get("packetType") != "ProgrammableControlPacket":
            continue
        if control["opcode"] != 1:
            continue
        view_num = control.get("viewNum")
        slice_num = control.get("sliceNum")
        if view_num is None or slice_num is None:
            continue
        if not (0 <= view_num <= y_res):
            continue
        frame = np.squeeze(archive.NextFrame())
        if ksp is None:
            n_samples = frame.shape[0]
            ksp = np.zeros([n_samples, y_res + 1, num_channels], dtype=np.complex64)
        if frame.shape[0] != ksp.shape[0]:
            continue
        ksp[:, view_num, :] = frame
    return ksp
    
def extract_kspace(data_fn):
    vendor = _detect_vendor(data_fn)
    if vendor == 'siemens':
        meas, ksp, timestamps = _extract_siemens(data_fn)
        return meas, ksp, timestamps, vendor
    elif vendor == 'ge':
        ksp = _extract_ge(data_fn)
        return None, ksp, None, vendor # TODO: Look into extracting physio data from GE

    
    