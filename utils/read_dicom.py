#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:46:36 2026

@author: jonah
"""
import re
import pydicom

def dicom_b1(filename):
    ds = pydicom.dcmread(filename)
    img = ds.pixel_array
    nx = ds.Rows
    pixel_spacing = ds.PixelSpacing
    dx = pixel_spacing[0] * 1e-3 # Covert to [m]
    dy = pixel_spacing[1] * 1e-3
    
    flip_angle = next((elem.value for elem in ds.iterall() if elem.keyword == "FlipAngle"), None)
    
    if flip_angle is not None:
        b1_map = img / (10 * float(flip_angle))
    else:
        raise ValueError(f"Flip Angle missing in {filename}. Cannot calculate B1 map.")
        
    return b1_map, nx, dx, dy
