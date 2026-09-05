#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:42:20 2026

@author: jonah
"""

import numpy as np
import yaml
import pypulseq as pp
import matplotlib.pyplot as plt

from bmctool.parameters.Parameters import Parameters
from bmctool.parameters.WaterPool import WaterPool
from bmctool.parameters.CESTPool import CESTPool
from bmctool.parameters.System import System
from bmctool.parameters.Options import Options
from bmctool.simulation.BMCSim import BMCSim

# ========================================== #
# Config
# ========================================== #
CONFIG_YAML = 'config_7pool.yaml'

# Combined Siemens ZSPEC sim .seq file written by continuous_spiral.py
# (FLAG_ZSPEC=True, FLAG_GE=False, FLAG_SIM=True).
SIM_SEQ_FN = '../sequences/sim/continuous_spiral_zspec_gauss.seq'

N_BURNIN = 80           # TRs discarded at the start of every same-offset run
REF_OFFSET_PPM = 75.0  # far-offset pseudo-M0 reference for normalization

# MTRasym(delta_omega) = Z(-delta_omega) - Z(+delta_omega), computed over
# positive offsets in this range (inclusive). Requires a matching -delta_omega
# sample for every +delta_omega sample in the range -- true here because
# zspec_offsets_ppm in continuous_spiral.py is built symmetrically (0.2 ppm
# steps -5..5, 1 ppm steps out to +/-10).
MTR_ASYM_MIN_PPM = 0.2
MTR_ASYM_MAX_PPM = 10.0

OUT_PNG = 'continuous_spiral_zspec_7pool_sim.png'
OUT_NPZ = 'continuous_spiral_zspec_7pool_sim.npz'


# ========================================== #
# Config -> Parameters
# ========================================== #
def load_config(fn):
    with open(fn) as f:
        return yaml.safe_load(f)


def build_params(cfg):
    """
    Builds a 5-pool Parameters object straight from config_5pool.yaml's
    own (flat, standard pulseq-cest) layout -- NOT nested under
    'system'/'options' like the old single-pool config used.
    """
    wp = cfg['water_pool']
    water_pool = WaterPool(f=wp['f'], t1=wp['t1'], t2=wp['t2'])

    cest_pools = []
    for cp in cfg['cest_pool'].values():
        cest_pools.append(CESTPool(f=cp['f'], t1=cp['t1'], t2=cp['t2'],
                                    k=cp['k'], dw=cp['dw']))
    print(f"[sim] Loaded {len(cest_pools)} CEST/NOE/MT pools: "
          f"{list(cfg['cest_pool'].keys())}")

    system = System(b0=cfg['b0'], gamma=cfg['gamma'],
                     b0_inhom=cfg['b0_inhom'], rel_b1=cfg['rel_b1'])

    options = Options(verbose=cfg['verbose'], reset_init_mag=cfg['reset_init_mag'],
                       scale=cfg['scale'], max_pulse_samples=cfg['max_pulse_samples'])

    return Parameters(water_pool=water_pool, cest_pools=cest_pools,
                       mt_pool=None, system=system, options=options)


# ========================================== #
# Seq loading helpers
# ========================================== #
def _to_float_array(val):
    if isinstance(val, str):
        return np.array([float(x) for x in val.split()])
    return np.array(val, dtype=float)


def load_seq_offsets(seq_path):
    seq = pp.Sequence()
    seq.read(seq_path)
    offsets_ppm = seq.definitions.get('offsets_ppm', None)
    if offsets_ppm is None:
        raise ValueError(f"{seq_path} does not define 'offsets_ppm' -- "
                          f"regenerate with continuous_spiral.py (FLAG_SIM=True).")
    return seq, _to_float_array(offsets_ppm)


def contiguous_runs(offsets_ppm):
    """
    Groups a per-TR offset array into (offset, start, stop) runs of
    contiguous identical offsets, in the order they occur. Handles both
    a combined ZSPEC file (many runs, one per offset) and a
    single-offset file (one run spanning the whole array) the same way.
    """
    runs = []
    start = 0
    for i in range(1, len(offsets_ppm) + 1):
        if i == len(offsets_ppm) or offsets_ppm[i] != offsets_ppm[start]:
            runs.append((offsets_ppm[start], start, i))
            start = i
    return runs


def compute_mtr_asym(unique_offsets, z_spectrum, min_ppm, max_ppm):
    """
    MTRasym(delta_omega) = Z(-delta_omega) - Z(+delta_omega) for every
    positive offset in [min_ppm, max_ppm] that has a matching negative
    counterpart in unique_offsets. Offsets without a match (shouldn't
    happen for the symmetric zspec_offsets_ppm grid, but would for a
    mismatched/edited one) are silently skipped -- check the returned
    length against your expected offset count if that matters to you.
    """
    z_by_offset = {round(float(o), 2): float(z) for o, z in zip(unique_offsets, z_spectrum)}

    mtr_offsets, mtr_asym = [], []
    for o in sorted(z_by_offset):
        if min_ppm - 1e-6 <= o <= max_ppm + 1e-6:
            neg = round(-o, 2)
            if neg in z_by_offset:
                mtr_offsets.append(o)
                mtr_asym.append(z_by_offset[neg] - z_by_offset[o])

    return np.array(mtr_offsets), np.array(mtr_asym)


# ========================================== #
# Main
# ========================================== #
def main():
    cfg = load_config(CONFIG_YAML)
    params = build_params(cfg)

    seq, offsets_ppm = load_seq_offsets(SIM_SEQ_FN)
    print(f"[sim] {SIM_SEQ_FN}: {len(offsets_ppm)} TRs, "
          f"{len(np.unique(offsets_ppm))} unique offset(s)")

    sim = BMCSim(params=params, seq=seq, verbose=False)
    sim.run()

    # State vector is grouped by component, not by pool: indices
    # [0 .. n_pools) = Mx for every pool, [n_pools .. 2*n_pools) = My,
    # [2*n_pools .. 3*n_pools) = Mz, each block ordered water-first.
    # Confirmed empirically: params.m_vec.shape == (15,) for 5 pools
    # (water + 4 CEST/NOE/MT) and params.mz_loc == 10 == 2*5 + 0, i.e.
    # water's Mz. That pins water's Mx/My the same way -- derived here
    # instead of hardcoded so it stays correct if the pool count changes.
    n_pools = params.m_vec.shape[0] // 3
    mx_loc = params.mz_loc - 2 * n_pools
    my_loc = params.mz_loc - n_pools
    mx = sim.m_out[mx_loc, :]
    my = sim.m_out[my_loc, :]
    mxy = np.abs(mx + 1j * my)

    if len(mxy) != len(offsets_ppm):
        raise RuntimeError(
            f"{SIM_SEQ_FN}: BMCSim returned {len(mxy)} points, expected "
            f"{len(offsets_ppm)} (one per TR) -- check the .seq file "
            f"is up to date with continuous_spiral.py.")

    # offset -> list of post-burn-in Mxy arrays, one entry per
    # contiguous same-offset run in the combined ZSPEC sequence.
    signal_by_offset = {}
    for offset, start, stop in contiguous_runs(offsets_ppm):
        run_mxy = mxy[start:stop]
        if N_BURNIN >= len(run_mxy):
            raise ValueError(
                f"N_BURNIN={N_BURNIN} >= run length {len(run_mxy)} for "
                f"offset {offset} ppm -- lower N_BURNIN or check the seq file.")
        signal_by_offset.setdefault(float(offset), []).append(run_mxy[N_BURNIN:])

    # Average all pooled post-burn-in samples per offset, then sort by
    # offset (ascending) so the plot is well-ordered.
    unique_offsets = np.array(sorted(signal_by_offset.keys()))
    avg_signal = np.array([
        np.concatenate(signal_by_offset[o]).mean() for o in unique_offsets
    ])

    ref_matches = np.isclose(unique_offsets, REF_OFFSET_PPM)
    if not ref_matches.any():
        raise ValueError(f"No offset matching REF_OFFSET_PPM={REF_OFFSET_PPM} found in "
                          f"the sim sequence(s) -- can't normalize the Z-spectrum.")
    s_ref = avg_signal[ref_matches][0]
    z_spectrum = avg_signal / s_ref

    mtr_offsets, mtr_asym = compute_mtr_asym(unique_offsets, z_spectrum,
                                              MTR_ASYM_MIN_PPM, MTR_ASYM_MAX_PPM)
    n_expected_positive = np.sum((unique_offsets >= MTR_ASYM_MIN_PPM - 1e-6) &
                                  (unique_offsets <= MTR_ASYM_MAX_PPM + 1e-6) & ~ref_matches)
    if len(mtr_offsets) < n_expected_positive:
        print(f"[sim] WARNING: only found {len(mtr_offsets)}/{n_expected_positive} "
              f"positive offsets in [{MTR_ASYM_MIN_PPM}, {MTR_ASYM_MAX_PPM}] ppm with a "
              f"matching negative counterpart -- offset grid may not be symmetric there.")

    # -- Plot -------------------------------------------------------------
    # Drop the far-offset reference point from the PLOTTED Z-spectrum
    # curve only -- it's a pseudo-M0 normalization anchor, not part of
    # the actual -10..10 ppm sweep, and leaving it in forces the x-axis
    # to span all the way out to REF_OFFSET_PPM, squishing the real
    # Z-spectrum into a sliver. z_spectrum/avg_signal/unique_offsets
    # (and the saved .npz) still include it -- only the plot excludes it.
    plot_mask = ~ref_matches
    plot_offsets = unique_offsets[plot_mask]
    plot_z = z_spectrum[plot_mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    ax1.plot(plot_offsets, plot_z, marker='o', markersize=3, color='#440154')
    ax1.invert_xaxis()  # conventional CEST Z-spectrum orientation
    ax1.set_xlabel('Offset (ppm)')
    ax1.set_ylabel(f'Z (normalized to {REF_OFFSET_PPM:.0f} ppm reference)')
    ax1.set_title('Simulated Z-spectrum, continuous_spiral ZSPEC (5-pool)')

    ax2.plot(mtr_offsets, mtr_asym, marker='o', markersize=3, color='#31688e')
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax2.invert_xaxis()  # same high-to-low ppm convention as ax1
    ax2.set_xlabel('Offset (ppm)')
    ax2.set_ylabel('MTRasym = Z(-off) - Z(+off)')
    ax2.set_title(f'MTRasym, {MTR_ASYM_MIN_PPM}-{MTR_ASYM_MAX_PPM} ppm')

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f'Saved {OUT_PNG}')

    np.savez_compressed(
        OUT_NPZ,
        offsets_ppm=unique_offsets,
        avg_signal=avg_signal,
        z_spectrum=z_spectrum,
        mtr_asym_offsets_ppm=mtr_offsets,
        mtr_asym=mtr_asym,
        pools=list(cfg['cest_pool'].keys()),
    )
    print(f'Saved {OUT_NPZ}')


if __name__ == '__main__':
    main()