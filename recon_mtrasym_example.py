#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:40:29 2026

@author: jonah
"""

# %% Step 0: Load reconstructed images
import numpy as np

im_ref = np.load('data/recon/siemens/MID00045_FID139553_spsp_ref.npy')
im_cest = np.load('data/recon/siemens/MID00046_FID139554_spsp_2ppm.npy')
im_conj = np.load('data/recon/siemens/MID00047_FID139555_spsp__2ppm.npy')

mtr_asym = (im_conj - im_cest) / im_ref

#%% Step 0.5: Draw ROIs and crop around the heart
# Needs interactive backend, e.g.: %matplotlib qt
from roipoly import RoiPoly
import matplotlib.pyplot as plt

# Use a representative frame for contrast when drawing (mean across bins)
ref_img = im_ref.mean(axis=0)
vmax = np.percentile(ref_img, 99.5)

fig1 = plt.figure()
plt.imshow(ref_img, cmap='gray', vmin=0, vmax=vmax)
plt.title('Draw rough MYOCARDIUM outline, then close the polygon (Enter)')
myo_roi = RoiPoly(color='r', fig=fig1)

myo_mask = myo_roi.get_mask(ref_img)

# Bounding box of the ROI, with a small margin
pad = 10
rows = np.any(myo_mask, axis=1)
cols = np.any(myo_mask, axis=0)
rmin, rmax = np.where(rows)[0][[0, -1]]
cmin, cmax = np.where(cols)[0][[0, -1]]
rmin, cmin = max(rmin - pad, 0), max(cmin - pad, 0)
rmax = min(rmax + pad, ref_img.shape[0] - 1)
cmax = min(cmax + pad, ref_img.shape[1] - 1)

def crop(arr):
    return arr[..., rmin:rmax+1, cmin:cmax+1]

im_ref   = crop(im_ref)
im_cest  = crop(im_cest)
im_conj  = crop(im_conj)
mtr_asym = crop(mtr_asym)
myo_mask = myo_mask[rmin:rmax+1, cmin:cmax+1]

#%% Step 1: Draw epi/endo ROIs on the reference image
# Needs interactive backend, e.g.: %matplotlib qt
from roipoly import RoiPoly
from scipy.ndimage import binary_erosion

ref_disp = im_ref.mean(axis=0)
vmax_ref = np.percentile(ref_disp, 99.5)

fig1 = plt.figure()
plt.imshow(ref_disp, cmap='gray', vmin=0, vmax=vmax_ref)
plt.title('Draw EPICARDIAL boundary, then close the polygon (Enter)')
epi_roi = RoiPoly(color='c', fig=fig1)

fig2 = plt.figure()
plt.imshow(ref_disp, cmap='gray', vmin=0, vmax=vmax_ref)
epi_roi.display_roi()
plt.title('Draw ENDOCARDIAL boundary, then close the polygon (Enter)')
endo_roi = RoiPoly(color='y', fig=fig2)

epi_mask = epi_roi.get_mask(ref_disp)
endo_mask = endo_roi.get_mask(ref_disp)

# Shrink each boundary inward by 1 pixel before combining, so the ring
# excludes edge/partial-volume pixels on both the epi and endo sides
erode_px = 1
epi_eroded = binary_erosion(epi_mask, iterations=erode_px)      # shrink epi boundary inward
endo_dilated = ~binary_erosion(~endo_mask, iterations=erode_px)  # grow endo boundary outward (into myocardium)
myo_ring_mask = epi_eroded & ~endo_dilated

n_bins = np.shape(mtr_asym)[0]

#%% Step 2: Mark RV insertion points and run AHA segmentation
import math

def distance(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def centroid(mask):
    ys, xs = np.nonzero(mask)
    return xs.mean(), ys.mean()

def aha_segmentation(mask, ip_mask):
    """
    Performs AHA segmentation on the myocardium using LV and RV insertion point masks.
    """
    mask_coords = np.argwhere(mask)
    ip_coords = np.argwhere(ip_mask)
    ip_coords = np.array([ip_coords[0], ip_coords[-1]])
    # Get points in myocardium with closest proximity to defined insertion points 
    insertion_points = []
    for coord in ip_coords:
        closest = mask_coords[0]
        for c in mask_coords:
            if distance(c, coord) < distance(closest, coord):
                closest = c
        insertion_points.append(closest)
    arv = insertion_points[0]
    irv = insertion_points[1]
    cx, cy = centroid(mask)
    [y, x] = np.nonzero(mask)
    inds = np.nonzero(mask)
    inds = list(zip(inds[0], inds[1]))
    # Offset all points by centroid
    x = x - cx
    y = y - cy
    arvx = arv[1] - cx
    arvy = arv[0] - cy
    irvx = irv[1] - cx
    irvy = irv[0] - cy
    # Find angular segment cutoffs
    pi = math.pi
    angle = lambda a, b: (math.atan2(a, b)) % (2 * pi)
    arv_ang = angle(arvy, arvx)
    irv_ang = angle(irvy, irvx)
    ang = [angle(yc, xc) for yc, xc in zip(y, x)]
    sept_cutoffs = np.linspace(0, arv_ang - irv_ang, num=3)  # two septal segments
    wall_cutoffs = np.linspace(arv_ang - irv_ang, 2 * pi, num=5)  # four wall segments
    cutoffs = []
    cutoffs.extend(sept_cutoffs)
    cutoffs.extend(wall_cutoffs[1:])
    ang = [(a - irv_ang) % (2 * pi) for a in ang]
    # Create arrays of each pixel/index in each segment
    segment_image = lambda a, b: [j for (i, j) in enumerate(inds) if ang[i] >= a and ang[i] < b]
    segmented_indices = [segment_image(a, b) for a, b in zip(cutoffs[:6], cutoffs[1:])]
    # List of labeled segments
    labeled_segments = {}
    labeled_segments['Inferoseptal'] = segmented_indices[0]
    labeled_segments['Anteroseptal'] = segmented_indices[1]
    labeled_segments['Anterior'] = segmented_indices[2]
    labeled_segments['Anterolateral'] = segmented_indices[3]
    labeled_segments['Inferolateral'] = segmented_indices[4]
    labeled_segments['Inferior'] = segmented_indices[5]
    return labeled_segments

fig3 = plt.figure()
plt.imshow(ref_disp, cmap='gray', vmin=0, vmax=vmax_ref)
plt.imshow(np.ma.masked_where(~myo_ring_mask, myo_ring_mask), cmap='cool', alpha=0.4)
plt.title('Click RV ANTEROSEPTAL insertion point, then RV INFEROSEPTAL insertion point')
pts = plt.ginput(2, timeout=0)
plt.close(fig3)

ip_mask = np.zeros_like(myo_ring_mask, dtype=bool)
for (px, py) in pts:
    ip_mask[int(round(py)), int(round(px))] = True

segments = aha_segmentation(myo_ring_mask, ip_mask)

#%% Step 3: Boxplot of MTRasym — septum (Infero+Antero pooled)
segment_names = ['Inferoseptal', 'Anteroseptal']
n_bins = np.shape(mtr_asym)[0]

# Combine pixel indices from both septal segments
sept_idx = segments['Inferoseptal'] + segments['Anteroseptal']
rows, cols = np.array(sept_idx).T

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True)
axes_flat = axes.flatten()
for b in range(n_bins):
    data = [mtr_asym[b][rows, cols]]
    try:
        axes_flat[b].boxplot(data, tick_labels=['Septum'])
    except TypeError:
        axes_flat[b].boxplot(data, labels=['Septum'])
    axes_flat[b].set_title(f"Phase {b+1}")
for b in range(n_bins, len(axes_flat)):
    axes_flat[b].axis('off')
plt.tight_layout()
plt.show()

