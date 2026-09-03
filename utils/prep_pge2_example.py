#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:42:09 2026

List of GE coils from pge2 (https://github.com/HarmonizedMRI/pge2/blob/main/matlab/%2Bpge2/opts.m)

% coil       Scanner   Gradient   chronaxie rheobase alpha  gmax  smax
% 'xrmw'     MR750w    XRMW       360d-6    20.0     0.324  33    120
% 'xrm'      MR750     XRM        334d-6    23.4     0.333  50    200
% 'whole'    HDx       TRM WHOLE  370d-6    23.7     0.344  23    77
% 'zoom'     HDx       TRM ZOOM   354d-6    29.1     0.309  40    150
% 'hrmbuhp'  UHP       HRMB       359d-6    26.5     0.370  100   200
% 'hrmw'     Premier   HRMW       642.4d-6  17.9     0.310  70    200
% 'magnus'   MAGNUS    MAGNUS     611d-6    52.2     0.324  300   750

@author: jonah
"""

from pathlib import Path
from datetime import datetime
import json
import os
import stat


# ============================================================== #
# Site configuration
# ============================================================== #
# psd_rf_wait / psd_grd_wait are the pge2 conversion-side timing
# compensation params (analogous to what write_vdspiral_cest.py handled
# manually per-scanner). pge_start is the first entry number to use the
# very first time a given site's counter file doesn't exist yet.
SITE_CONFIGS = {
    "site_1": {
        "single_hop": True,
        "ssh_host": "ge",         
        "scanner_sequence_dir": Path("/usr/g/research/jonah"),
        "scanner_entry_dir": Path("/srv/nfs/psd/usr/psd/pulseq/v7"),
        "scanner_list_dir": Path("/export/home/sdc/example"),
        "scan_list_filename": "pulseq_scans.list",
        "pge_start": 10,
        "psd_rf_wait": 58e-6,   # Make sure this is correct, examine CVs directly!!
        "psd_grd_wait": 60e-6,  # Make sure this is correct, examine CVs directly!!
        "coil": "xrmw",         # Make sure this is correct!
    },
    "site_2": {
        "single_hop": False,      # Maybe your setup requires a proxy jump
        "proxy_host": "proxy",
        "scanner_target": "ge",
        "scanner_sequence_dir": Path("/usr/g/research/jonah"),
        "scanner_entry_dir": Path("/srv/nfs/psd/usr/psd/pulseq/v7/"),
        "scanner_list_dir": Path("/export/home/sdc/example"), 
        "scan_list_filename": "pulseq_scans.list",
        "pge_start": 10, 
        "psd_rf_wait": 148e-6,  
        "psd_grd_wait": 152e-6,  
        "coil": "hrmw",         
    },
}

# Where the per-site "next entry number" counter is persisted between runs.
_STATE_PATH = Path(__file__).resolve().parent / ".pge2_state.json"


def get_site_config(site: str) -> dict:
    site_key = site.lower()
    if site_key not in SITE_CONFIGS:
        raise ValueError(
            f"Unknown site '{site}'. Valid options: {list(SITE_CONFIGS.keys())}"
        )
    return SITE_CONFIGS[site_key]


def _load_state() -> dict:
    """
    State schema per site: {"next_n": int, "entries": {"<pge_n>": {"scan": str, "description": str}}}
    Transparently migrates the old {site: next_n} format (no entries) if found.
    """
    if _STATE_PATH.exists():
        try:
            raw = json.loads(_STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        migrated = {}
        for site_key, val in raw.items():
            if isinstance(val, dict):
                val.setdefault("entries", {})
                migrated[site_key] = val
            else:
                migrated[site_key] = {"next_n": val, "entries": {}}
        return migrated
    return {}


def _save_state(state: dict):
    _STATE_PATH.write_text(json.dumps(state, indent=2))


def _get_site_state(state: dict, site_key: str) -> dict:
    config = get_site_config(site_key)
    return state.setdefault(site_key, {"next_n": config["pge_start"], "entries": {}})


def next_pge_n(site: str, override: int = None) -> int:
    """
    Returns the next pge<N> entry number to use for `site`, persisting
    the counter across runs so successive sequences don't collide.

    If `override` is given, that number is used instead, and the persisted
    counter is bumped to override + 1 so subsequent auto-numbered calls
    continue on from there.
    """
    site_key = site.lower()
    state = _load_state()
    site_state = _get_site_state(state, site_key)

    n = override if override is not None else site_state.get("next_n", 0)
    site_state["next_n"] = n + 1
    _save_state(state)
    return n


def peek_next_pge_n(site: str) -> int:
    """Non-mutating look at what next_pge_n() would currently return."""
    site_key = site.lower()
    state = _load_state()
    return _get_site_state(state, site_key).get("next_n", 0)


def record_scan_list_entry(site: str, pge_n: int, seq_name: str, description: str):
    """
    Records/overwrites one entry (keyed by pge_n) in the persisted per-site
    manifest used to regenerate pulseq_scans.list in full each deploy.
    """
    site_key = site.lower()
    state = _load_state()
    site_state = _get_site_state(state, site_key)
    site_state["entries"][str(pge_n)] = {"scan": seq_name, "description": description}
    _save_state(state)


def render_scan_list(site: str) -> str:
    """
    Renders the FULL pulseq_scans.list contents (header + one line per known
    entry, sorted by opuser1/pge_n) from the persisted manifest for `site`.
    This is meant to fully overwrite the remote file each deploy, not append,
    so stale/duplicate/reused-number entries don't accumulate.
    """
    site_key = site.lower()
    state = _load_state()
    entries = _get_site_state(state, site_key)["entries"]
    lines = ["# opuser1\tscan\tdescription"]
    for n in sorted(entries.keys(), key=int):
        e = entries[n]
        lines.append(f"{n}\t{e['scan']}.mat\t{e['description']}")
    return "\n".join(lines) + "\n"


def get_conversion_wait_params(site: str) -> dict:
    """
    psdRfWait / psdGrdWait / coil to merge into the system_dict passed to
    eng.convert_pge2(). Keys are camelCase to match convert_pge2.m's
    isfield(sys, 'psdRfWait') / 'psdGrdWait' / 'coil' checks - if a value is
    None here, convert_pge2.m falls back to its UCSF Premier defaults.
    """
    config = get_site_config(site)
    return {
        "psdRfWait": config["psd_rf_wait"],
        "psdGrdWait": config["psd_grd_wait"],
        "coil": config["coil"],
    }


# ============================================================== #
# Deploy script generation
# ============================================================== #
def generate_batch_deploy_script(
    batch_data: list,
    output_script_path: Path,
    site: str,
):
    """
    Generates a SINGLE .sh script to deploy multiple prepared files (.pge,
    .entry, .mat, and pulseq_scans.list are all treated the same way here -
    plain files pushed to their configured remote_dir, overwriting whatever
    was there).

    Two transfer modes, selected by SITE_CONFIGS[site]["single_hop"]:
      - single_hop=True
        `ssh_host` (e.g. "ge") - no staging needed.
      - single_hop=False
        two-stage transfer (Local -> proxy_host -> scanner_target) to keep SSH keys isolated on the proxy.
    """
    config = get_site_config(site)
    single_hop = config.get("single_hop", False)

    if single_hop:
        script_content = _render_single_hop_script(batch_data, site, config)
    else:
        script_content = _render_two_hop_script(batch_data, site, config)

    try:
        output_script_path.write_text(script_content)
        st = os.stat(output_script_path)
        os.chmod(output_script_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"\n[Success] Generated master deploy script: '{output_script_path}'")
    except Exception as e:
        print(f"Error generating master deploy script: {e}")


# ============================================================== #
# Deploy script renderers
# ============================================================== #
def _render_two_hop_script(batch_data, site, config):
    proxy_host = config["proxy_host"]
    scanner_target = config["scanner_target"]

    # Stage every local file (flattened to its basename) into one local temp
    # dir, then send it to the proxy in a SINGLE tar-over-ssh call, and do the
    # proxy->scanner push in a SINGLE remote script (one more ssh call).
    # Previously this was 2-3 separate ssh/scp invocations PER ITEM, and some
    # 2FA setups (e.g. Duo) challenge on every new exec/session even when the
    # connection itself is multiplexed - collapsing to two total ssh calls to
    # the proxy avoids that.
    stage_copies = "\n".join(
        f'cp "{item["local_path"]}" "$LOCAL_STAGE/{item["local_path"].name}"'
        for item in batch_data
    )

    # Parallel bash arrays consumed by the remote push script: filename -> remote dir on scanner_target.
    files_array = " ".join(f'"{item["local_path"].name}"' for item in batch_data)
    dirs_array = " ".join(f'"{item["remote_dir"]}"' for item in batch_data)

    remote_script = f"""set -e
cd "$PROXY_TMP_DIR"
FILES=({files_array})
DIRS=({dirs_array})
for i in "${{!FILES[@]}}"; do
    echo "Pushing to scanner: ${{FILES[$i]}} -> ${{DIRS[$i]}}"
    scp -r "${{FILES[$i]}}" "{scanner_target}:${{DIRS[$i]}}"
done"""

    return f"""#!/bin/bash
#
# This script was auto-generated by prep_pge2.py
# It deploys a batch of {len(batch_data)} items via a secure TWO-STAGE copy,
# consolidated into two ssh calls total (not per-item) to minimize 2FA prompts.
# Site: {site}
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#

set -e

# --- Configuration ---
PROXY_HOST="{proxy_host}"
SCANNER_TARGET="{scanner_target}"

# Generate unique temp paths using the script's Process ID ($$)
SOCKET="/tmp/pge2_deploy_socket_$$"
PROXY_TMP_DIR="/tmp/pge2_staging_$$"
LOCAL_STAGE=$(mktemp -d)

echo "--- Starting Secure TWO-STAGE pge2 Deployment ({len(batch_data)} items, site={site}) ---"
echo "Route: Home -> $PROXY_HOST (Staging) -> $SCANNER_TARGET"
echo ""
echo "🔑 Opening master connection. Please complete your 2FA for $PROXY_HOST..."

# 1. Establish Master Connection (No -J jump flag, just a direct connection to proxy)
ssh -M -S "$SOCKET" -f -N "$PROXY_HOST"

# 2. Safety Cleanup
# Ensures the staging folders are wiped and the background socket is closed on exit
trap 'echo -e "\\n🧹 Cleaning up staging files..."; rm -rf "$LOCAL_STAGE"; ssh -S "$SOCKET" "$PROXY_HOST" "rm -rf \\"$PROXY_TMP_DIR\\""; echo "🧹 Closing master SSH connection..."; ssh -S "$SOCKET" -O exit "$PROXY_HOST" 2>/dev/null' EXIT

# 3. Configure SCP and SSH to use the established socket
SCP_OPT="-o ControlPath=$SOCKET"
SSH_OPT="-o ControlPath=$SOCKET"

echo "✅ Connection established."

# --- Stage all files locally, then send in ONE tar transfer (1 ssh call) ---
{stage_copies}

echo "1. Sending all {len(batch_data)} files to proxy in a single transfer..."
tar -czf - -C "$LOCAL_STAGE" . | ssh $SSH_OPT "$PROXY_HOST" "mkdir -p \\"$PROXY_TMP_DIR\\" && tar -xzf - -C \\"$PROXY_TMP_DIR\\""

# --- Push everything from proxy to scanner (1 more ssh call) ---
echo "2. Pushing all files from proxy to scanner ($SCANNER_TARGET)..."
ssh $SSH_OPT "$PROXY_HOST" "PROXY_TMP_DIR=\\"$PROXY_TMP_DIR\\" bash -s" <<'REMOTE_EOF'
{remote_script}
REMOTE_EOF

echo
echo "=== All {len(batch_data)} items successfully deployed! ==="
exit 0
"""


def _render_single_hop_script(batch_data, site, config):
    ssh_host = config["ssh_host"]

    deployment_steps = []
    for item in batch_data:
        deployment_steps.append(f"""
echo "--------------------------------------------------"
echo "Deploying: {item['local_path'].name}"
echo "--------------------------------------------------"
ssh $SSH_OPT "$SCANNER_HOST" "mkdir -p \\"{item['remote_dir']}\\""
scp $SCP_OPT -r "{item['local_path']}" "$SCANNER_HOST:{item['remote_dir']}/"
""")
    steps_joined = "\n".join(deployment_steps)

    return f"""#!/bin/bash
#
# This script was auto-generated by prep_pge2.py
# It deploys a batch of {len(batch_data)} items via a DIRECT single-hop copy.
# Site: {site}
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#

set -e

# --- Configuration ---
SCANNER_HOST="{ssh_host}"

# Generate a unique socket path using the script's Process ID ($$)
SOCKET="/tmp/pge2_deploy_socket_$$"

echo "--- Starting Direct pge2 Deployment ({len(batch_data)} items, site={site}) ---"
echo "Route: Home -> $SCANNER_HOST"
echo ""
echo "🔑 Opening master connection to $SCANNER_HOST..."

# 1. Establish Master Connection directly to the scanner
ssh -M -S "$SOCKET" -f -N "$SCANNER_HOST"

# 2. Safety Cleanup: close the background socket on exit
trap 'echo -e "\\n🧹 Closing master SSH connection..."; ssh -S "$SOCKET" -O exit "$SCANNER_HOST" 2>/dev/null' EXIT

# 3. Configure SCP and SSH to use the established socket
SCP_OPT="-o ControlPath=$SOCKET"
SSH_OPT="-o ControlPath=$SOCKET"

echo "✅ Connection established."

# --- Deployment Steps ---
{steps_joined}
echo
echo "=== All {len(batch_data)} items successfully deployed! ==="
exit 0
"""


def prep_pge2_batch(
    batch_tasks: list,
    site: str,
    output_script_path: Path,
    update_scan_list: bool = True,
):
    """
    Processes a batch of sequence files, creates .entry files, and consolidates
    everything into a single deployment manifest.

    batch_tasks format:
      [{"seq_path": Path, "pge_n": int (optional), "description": str (optional)}, ...]
    If "pge_n" is omitted, the next number is auto-assigned per site and the
    persisted counter is advanced (see next_pge_n()). "description" defaults
    to the sequence name if omitted.

    site: "site_1" or "site_2" - selects deployment locations from
    SITE_CONFIGS.
    """
    config = get_site_config(site)
    scanner_sequence_dir = config["scanner_sequence_dir"]
    scanner_entry_dir = config["scanner_entry_dir"]
    scanner_list_dir = config["scanner_list_dir"]
    scan_list_filename = config["scan_list_filename"]

    batch_manifest = []

    for task in batch_tasks:
        seq_filename = task["seq_path"]
        if not isinstance(seq_filename, Path):
            seq_filename = Path(seq_filename)

        pge_n = next_pge_n(site, override=task.get("pge_n"))

        # 1. Target the .pge directory directly
        pge_local_path = seq_filename.with_suffix('.pge').resolve()
        seq_name = seq_filename.stem

        # 2. Create the .entry file locally (write/overwrite each time)
        entry_filename = f"pge{pge_n}.entry"
        # Add .resolve() here to guarantee an absolute path in the bash script
        entry_local_path = (seq_filename.parent / entry_filename).resolve()

        # Build the exact scanner string format
        # Force forward slashes for the Linux scanner environment
        remote_pge_path = f"{scanner_sequence_dir.as_posix()}/{pge_local_path.name}"
        entry_content = f"1\n{remote_pge_path}\n\n"

        entry_local_path.write_text(entry_content)

        # 3. Append the .pge sequence directory to manifest
        batch_manifest.append({
            "local_path": pge_local_path,
            "remote_dir": scanner_sequence_dir,
            "item_id": f"{seq_name}_pge",
        })

        # 4. Append the .entry file to manifest
        batch_manifest.append({
            "local_path": entry_local_path,
            "remote_dir": scanner_entry_dir,
            "item_id": f"{seq_name}_entry",
        })

        # 5. Append the .mat file (psq/params/pislquant, saved by convert_pge2.m
        # alongside the .pge). Lives next to pulseq_scans.list per the
        # PulSeg/pge2 README's directory layout, needed for pulseq_shift_fov.sh.
        mat_local_path = seq_filename.with_suffix('.mat').resolve()
        batch_manifest.append({
            "local_path": mat_local_path,
            "remote_dir": scanner_list_dir,
            "item_id": f"{seq_name}_mat",
        })

        description = task.get("description", seq_name)
        record_scan_list_entry(site, pge_n, seq_name, description)
        print(f"[prep_pge2] {seq_name} -> pge{pge_n}.entry (site={site})")

    # Regenerate the FULL pulseq_scans.list (from the persisted per-site
    # manifest, not just this batch) and push it as a normal file - it
    # overwrites the remote copy rather than appending, so stale/duplicate
    # entries from earlier runs or reused pge numbers don't accumulate.
    if update_scan_list and batch_manifest:
        list_local_path = (output_script_path.parent / scan_list_filename).resolve()
        list_local_path.write_text(render_scan_list(site))
        batch_manifest.append({
            "local_path": list_local_path,
            "remote_dir": scanner_list_dir,
            "item_id": "scan_list",
        })

    # Generate ONE master deployment script for all sequences and entries combined
    if batch_manifest:
        generate_batch_deploy_script(batch_manifest, output_script_path, site=site)


def prep_pge2(
    seq_path,
    site: str,
    output_script_path: Path,
    pge_n: int = None,
    description: str = None,
    update_scan_list: bool = True,
):
    """
    Convenience single-sequence wrapper around prep_pge2_batch, for the
    common case of prepping/deploying one sequence at a time (e.g. straight
    out of continuous_spiral.py).
    """
    task = {"seq_path": seq_path}
    if pge_n is not None:
        task["pge_n"] = pge_n
    if description is not None:
        task["description"] = description
    prep_pge2_batch(
        [task],
        site=site,
        output_script_path=output_script_path,
        update_scan_list=update_scan_list,
    )