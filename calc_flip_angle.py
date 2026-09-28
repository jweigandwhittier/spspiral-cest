#!/usr/bin/env python3
"""
Get the flip angle of the excitation pulses in a Pulseq .seq file.

This parses the [RF], [SHAPES] and [DEFINITIONS] sections of the .seq file
directly, instead of going through pypulseq's Sequence.read(). That call
throws a KeyError on some pypulseq versions when reading files written in
the Pulseq 1.4.x RF format (id, amp, mag_id, phase_id, time_shape_id, delay,
freq, phase) -- the extra time_shape_id column gets misparsed as a shape id
that doesn't exist, and the shape-id-0 lookup then throws
shape_library.data[mag_shape] -> KeyError: 0. Parsing the sections
directly sidesteps that bug and only needs numpy.

Flip angle for an RF block is:

    360 deg * amp (Hz) * dt (s) * | sum( mag_shape(t) * cos(2*pi*phase_shape(t)) ) |

NOT 360 * amp * dt * sum(mag_shape). Pulses with side lobes (sinc, apodized
pulses, etc.) have a phase shape that flips between 0 and 0.5 cycles (0 and
pi radians) to encode the sign of each negative lobe, since the magnitude
shape itself is stored as a non-negative envelope. A negative lobe SUBTRACTS
from the net flip angle -- summing |mag_shape| alone (ignoring that sign)
over-counts every side lobe and inflates the result. This matches exactly
how pypulseq computes the nominal flip angle at design time (see
make_sinc_pulse.py: flip = np.sum(signal) * dwell * 2 * pi, using the
signed, real-valued waveform, not its absolute value).
"""

import sys
from collections import defaultdict

import numpy as np

SEQ_FILE = "/Users/jonah/Documents/Vandsburger_Lab/spspiral-cest/sequences/siemens/continuous_spiral_gauss_2p0_ppm.seq"


def decompress_shape(num_samples, data_pack):
    """Pulseq run-length-on-derivative decompression (same algorithm as pypulseq)."""
    data_pack = np.asarray(data_pack, dtype=float)
    data_pack_len = len(data_pack)
    num_samples = int(num_samples)

    if num_samples == data_pack_len:
        return data_pack

    decompressed = np.zeros(num_samples)
    diff = data_pack[1:] - data_pack[:-1]
    markers = np.where(diff == 0.0)[0]

    count_pack = 0
    count_unpack = 0
    for idx in markers:
        next_pack = idx
        n = next_pack - count_pack
        if n < 0:
            continue
        elif n > 0:
            decompressed[count_unpack:count_unpack + n] = data_pack[count_pack:next_pack]
            count_pack += n
            count_unpack += n
        rep = int(data_pack[count_pack + 2] + 2)
        decompressed[count_unpack:count_unpack + rep] = data_pack[count_pack]
        count_pack += 3
        count_unpack += rep

    if count_pack <= data_pack_len - 1:
        decompressed[count_unpack:] = data_pack[count_pack:]

    return np.cumsum(decompressed)


def get_section(lines, name):
    """Return the raw lines of a [NAME] section."""
    out = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == f"[{name}]":
            in_section = True
            continue
        if in_section:
            if stripped.startswith("[") and stripped.endswith("]"):
                break
            out.append(line)
    return out


def parse_definitions(lines):
    defs = {}
    for line in get_section(lines, "DEFINITIONS"):
        parts = line.split()
        if not parts:
            continue
        defs[parts[0]] = parts[1:]
    return defs


def parse_rf(lines):
    """id -> dict(amp, mag_id, phase_id, time_id, delay, freq, phase)."""
    rf = {}
    for line in get_section(lines, "RF"):
        parts = line.split()
        if not parts or not parts[0].lstrip("-").isdigit():
            continue
        rid, amp, mag_id, phase_id, time_id, delay, freq, phase = parts[:8]
        rf[int(rid)] = dict(
            amp=float(amp), mag_id=int(mag_id), phase_id=int(phase_id),
            time_id=int(time_id), delay=float(delay), freq=float(freq),
            phase=float(phase),
        )
    return rf


def parse_shapes(lines):
    """shape_id -> decompressed numpy array."""
    shapes = {}
    cur_id, cur_n, cur_data = None, None, []
    for raw in get_section(lines, "SHAPES"):
        line = raw.strip()
        if not line:
            if cur_id is not None:
                shapes[cur_id] = decompress_shape(cur_n, cur_data)
                cur_id, cur_n, cur_data = None, None, []
            continue
        if line.startswith("shape_id"):
            cur_id = int(line.split()[1])
            cur_data = []
        elif line.startswith("num_samples"):
            cur_n = int(line.split()[1])
        else:
            cur_data.append(float(line))
    if cur_id is not None:
        shapes[cur_id] = decompress_shape(cur_n, cur_data)
    return shapes


def parse_blocks_rf_ids(lines):
    """RF ids actually used in [BLOCKS], in block order."""
    ids = []
    for line in get_section(lines, "BLOCKS"):
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        # NUM DUR RF GX GY GZ ADC EXT
        if len(parts) >= 3:
            rf_id = int(parts[2])
            if rf_id != 0:
                ids.append(rf_id)
    return ids


def main(path):
    with open(path, "r") as f:
        lines = f.readlines()

    defs = parse_definitions(lines)
    rf_raster = float(defs.get("RadiofrequencyRasterTime", ["1e-6"])[0])

    rf_defs = parse_rf(lines)
    shapes = parse_shapes(lines)
    block_rf_ids = parse_blocks_rf_ids(lines)

    if not rf_defs:
        print("No [RF] section found -- is this a valid .seq file?")
        return

    # Flip angle per RF definition id, using the phase shape to recover the
    # sign of each sample (see module docstring -- this is NOT just sum(mag)).
    flip = {}
    for rid, d in rf_defs.items():
        mag = shapes.get(d["mag_id"])
        if mag is None:
            flip[rid] = float("nan")
            continue
        phase_id = d["phase_id"]
        if phase_id == 0 or phase_id not in shapes:
            # No phase shape -> purely real, non-negative envelope (e.g. a
            # plain Gaussian saturation pulse with no side lobes).
            signed = mag
        else:
            phase = shapes[phase_id]
            signed = mag * np.cos(2.0 * np.pi * phase)
        flip[rid] = 360.0 * abs(d["amp"]) * rf_raster * abs(float(np.sum(signed)))

    print(f"File: {path}")
    print(f"RF raster time: {rf_raster * 1e6:.3f} us")
    print(f"RF pulse definitions found: {len(rf_defs)}")

    # Distinguish pulses by frequency offset: 0 Hz offset pulses are the on-resonance
    # excitation pulses; a large offset (e.g. a CEST saturation pulse) is not excitation.
    by_freq = defaultdict(list)
    for rid, d in rf_defs.items():
        by_freq[round(d["freq"], 1)].append(rid)

    print("\nRF definitions grouped by frequency offset:")
    for freq, rids in sorted(by_freq.items()):
        angles = np.array([flip[r] for r in rids])
        usage = sum(1 for u in block_rf_ids if u in rids)
        label = "excitation (on-resonance)" if freq == 0.0 else f"off-resonance ({freq:.1f} Hz)"
        print(f"  {label}: {len(rids)} definition(s), used in {usage} block(s)")
        print(f"    flip angle min/mean/max: {angles.min():.3f} / {angles.mean():.3f} / {angles.max():.3f} deg")
        uniq = np.unique(np.round(angles, 3))
        if uniq.size <= 6:
            print(f"    unique flip angle(s): {', '.join(f'{a:.3f}' for a in uniq)} deg")

    print("\n(Excitation pulses are the 0 Hz group above; a nonzero-offset group,")
    print(" if present, is a saturation/preparation pulse rather than excitation.)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SEQ_FILE)