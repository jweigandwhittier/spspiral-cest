#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:46:36 2026

@author: jonah
"""
import re
import pydicom

def dicom_b1_siemens(filename):
    ds = pydicom.dcmread(filename)
    img = ds.pixel_array
    nx = ds.Rows
    
    flip_angle = next((elem.value for elem in ds.iterall() if elem.keyword == "FlipAngle"), None)
    fov_string = next((elem.value for elem in ds.iterall() if elem.tag == (0x0021, 0x105E)), "")
    
    fov = [float(x) * 1e-3 for x in re.findall(r'\d+', str(fov_string))]
    
    if flip_angle is not None:
        b1_map = img / (10 * float(flip_angle))
    else:
        raise ValueError(f"Flip Angle missing in {filename}. Cannot calculate B1 map.")
        
    return b1_map, nx, fov

def dicom_b1_ge(filename):
    return 