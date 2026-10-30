#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:46:36 2026

@author: jonah

Load a B1 (flip-angle) map from DICOM and convert it to relative B1.
Handles classic and enhanced (multi-frame, Siemens XA) DICOM.

Pixel spacing priority:
  1. PixelSpacing (top level or inside the functional group sequences)
  2. Siemens ASCCONV FOV / reconstructed matrix
  3. ReconstructionDiameter / reconstructed matrix (GE)
"""
import re
import warnings

import numpy as np
import pydicom


def _find_attr(ds, keyword):
    """Return the first value for `keyword`: top level, then shared, then per-frame groups."""
    if keyword in ds:
        return ds[keyword].value
    for seq_name in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        seq = getattr(ds, seq_name, None)
        if seq:
            for elem in seq[0].iterall():
                if elem.keyword == keyword:
                    return elem.value
    # last resort: anywhere in the dataset
    return next((e.value for e in ds.iterall() if e.keyword == keyword), None)


def _siemens_fov(filename):
    """Read readout/phase FOV [mm] from the embedded Siemens ASCCONV protocol text."""
    with open(filename, "rb") as f:
        raw = f.read()
    ro = re.search(rb"sSliceArray\.asSlice\[0\]\.dReadoutFOV\s*=\s*([\d.]+)", raw)
    pe = re.search(rb"sSliceArray\.asSlice\[0\]\.dPhaseFOV\s*=\s*([\d.]+)", raw)
    if ro is None or pe is None:
        return None
    return float(ro.group(1)), float(pe.group(1))


def _spacing_from_fov(filename, ds):
    """Pixel spacing [mm] as (dx, dy) from FOV / reconstructed matrix, or None."""
    ny, nx = ds.Rows, ds.Columns  # reconstructed matrix (includes interpolation)

    fov = _siemens_fov(filename)
    if fov is not None:
        fov_ro, fov_pe = fov
        pe_dir = _find_attr(ds, "InPlanePhaseEncodingDirection")  # 'ROW' or 'COL'
        if pe_dir == "ROW":  # phase encoding horizontal -> readout vertical
            return fov_pe / nx, fov_ro / ny
        return fov_ro / nx, fov_pe / ny  # 'COL' (default): readout horizontal

    recon_diam = _find_attr(ds, "ReconstructionDiameter")  # GE: square FOV [mm]
    if recon_diam is not None:
        return float(recon_diam) / nx, float(recon_diam) / ny

    return None


def _get_spacing(filename, ds):
    """Pixel spacing (dx, dy) in [m]."""
    pixel_spacing = _find_attr(ds, "PixelSpacing")
    if pixel_spacing is not None:
        dy = float(pixel_spacing[0])  # row spacing    -> y [mm]
        dx = float(pixel_spacing[1])  # column spacing -> x [mm]
    else:
        spacing = _spacing_from_fov(filename, ds)
        if spacing is None:
            raise ValueError(
                f"No PixelSpacing, Siemens FOV, or ReconstructionDiameter in {filename}."
            )
        dx, dy = spacing
    return dx * 1e-3, dy * 1e-3


def _select_fa_frames(ds, img):
    """For multi-frame files, keep only frames labelled as a flip-angle map, if identifiable."""
    if img.ndim < 3 or "PerFrameFunctionalGroupsSequence" not in ds:
        return img

    keep = []
    for i, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
        frame_type = next(
            (e.value for e in frame.iterall() if e.keyword == "FrameType"), None
        )
        if frame_type is not None and any("FLIP" in str(t).upper() for t in frame_type):
            keep.append(i)

    if keep and len(keep) < img.shape[0]:
        img = img[keep]
    elif not keep:
        warnings.warn(
            "Multi-frame file: could not identify the flip-angle-map frame(s); "
            "returning all frames. Check which frame is the FA map."
        )
    return np.squeeze(img)


def dicom_b1(filename):
    """
    Returns
    -------
    b1_map : ndarray   relative B1 (actual FA / nominal FA)
    nx, ny : int       reconstructed matrix size (columns, rows)
    dx, dy : float     pixel spacing [m]
    """
    ds = pydicom.dcmread(filename)
    img = ds.pixel_array.astype(np.float64)
    img = _select_fa_frames(ds, img)

    ny, nx = ds.Rows, ds.Columns
    dx, dy = _get_spacing(filename, ds)

    flip_angle = _find_attr(ds, "FlipAngle")
    if flip_angle is None:
        raise ValueError(f"Flip Angle missing in {filename}. Cannot calculate B1 map.")

    # Siemens tfl_b1map stores actual FA x 10 [deg]
    b1_map = img / (10.0 * float(flip_angle))

    return b1_map, nx, dx, dy


if __name__ == "__main__":
    import sys

    b1, nx, ny, dx, dy = dicom_b1(sys.argv[1])
    print(f"B1 map shape: {b1.shape}, matrix: {nx} x {ny}, "
          f"spacing: {dx*1e3:.3f} x {dy*1e3:.3f} mm")
    print(f"B1 range: {np.nanmin(b1):.3f} - {np.nanmax(b1):.3f}, "
          f"median: {np.nanmedian(b1):.3f}")