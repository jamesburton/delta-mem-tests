"""SSH coordination helper for the Strix Halo training box.

Wraps the common operations we'd do over SSH from this Windows host to
the Strix Halo box during training runs:

  - check        : confirm SSH reachable + GPU is free (not running other jobs)
  - copy-up      : rsync repo state up to Strix (script + data + adapter, no submodule downloads)
  - copy-down    : pull a checkpoint dir back from Strix
  - tail-log     : stream the latest training log
  - kill         : forcefully stop a training PID (use only after confirming with the user)

The Strix host is read from $env:STRIX_SSH_TARGET (default 'strix'); set
this in your shell to your actual SSH alias (e.g. 'jamesb@strix-halo.local').
The remote workdir is $env:STRIX_REPO_DIR (default '/home/jamesb/delta-mem-tests').

Usage from this Windows host:

    python -m tools.strix_ssh check
    python -m tools.strix_ssh copy-up
    python -m tools.strix_ssh tail-log
    python -m tools.strix_ssh copy-down checkpoints/lora_v1_strix
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import Optional


def _strix_target() -> str:
    return os.environ.get("STRIX_SSH_TARGET", "strix")


def _strix_workdir() -> str:
    return os.environ.get("STRIX_REPO_DIR", "/home/jamesb/delta-mem-tests")


def _ssh_run(remote_cmd: str, capture: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command on the Strix host via SSH. Returns CompletedProcess."""
    full = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            _strix_target(), remote_cmd]
    return subprocess.run(
        full, capture_output=capture, text=True, timeout=timeout,
    )


def _parse_nvidia_smi_csv(output: str) -> list[dict]:
    """Parse nvidia-smi --format=csv,noheader output into a list of dicts."""
    rows = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({
                "name": parts[0],
                "util_pct": parts[1],
                "mem_used_mib": parts[2],
                "mem_total_mib": parts[3],
            })
    return rows


def cmd_check() -> int:
    """Confirm SSH works AND check whether GPU is currently busy."""
    target = _strix_target()
    print(f"[strix] target: {target}")
    try:
        # First: SSH reachability
        result = _ssh_run("uname -a", timeout=15)
        if result.returncode != 0:
            print(f"  SSH FAILED: {result.stderr.strip()}")
            print(f"  hint: set $env:STRIX_SSH_TARGET to your actual SSH alias")
            print(f"        and confirm key-based auth works: ssh {target} uname -a")
            return 1
        print(f"  SSH OK: {result.stdout.strip()}")

        # GPU state
        result = _ssh_run(
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits",
            timeout=15,
        )
        if result.returncode != 0:
            print(f"  nvidia-smi FAILED: {result.stderr.strip()}")
            print(f"  hint: AMD ROCm box? try: rocm-smi  (this helper assumes nvidia-smi)")
            return 2
        gpus = _parse_nvidia_smi_csv(result.stdout)
        if not gpus:
            print(f"  no GPUs reported by nvidia-smi — check driver state on Strix")
            return 3

        any_busy = False
        for i, g in enumerate(gpus):
            util = int(g["util_pct"])
            mem_used = int(g["mem_used_mib"])
            mem_total = int(g["mem_total_mib"])
            mem_pct = 100 * mem_used / max(mem_total, 1)
            status = "BUSY" if (util > 5 or mem_pct > 10) else "FREE"
            if status == "BUSY":
                any_busy = True
            print(f"  GPU{i} ({g['name']}): util={util}%  mem={mem_used}/{mem_total} MiB ({mem_pct:.1f}%)  -> {status}")

        # Who's using it?
        if any_busy:
            print("\n  GPU appears in use. Active processes:")
            result = _ssh_run(
                "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                "--format=csv,noheader,nounits",
                timeout=15,
            )
            print(f"  {result.stdout.strip() or '(none reported)'}")
            print("\n  Coordinate before submitting a job. Use 'who' or 'tmux ls' on Strix"
                  " to find who owns the work.")
            return 10  # non-zero but specific to "busy"

        print("\n  -> Strix GPU is FREE. Safe to submit a training job.")
        return 0
    except subprocess.TimeoutExpired:
        print(f"  SSH TIMEOUT to {target}")
        print(f"  hint: check network and that the host responds to ping")
        return 4


def cmd_copy_up(args: argparse.Namespace) -> int:
    """rsync the local repo state up to Strix. Excludes .venv, .git/objects,
    HF caches, large data files unless --include-data."""
    excludes = [
        "--exclude=.venv/", "--exclude=__pycache__/", "--exclude=.git/objects/",
        "--exclude=outputs/", "--exclude=report/raw/", "--exclude=checkpoints/",
    ]
    if not args.include_data:
        excludes.append("--exclude=data/")
    src = "./"
    dst = f"{_strix_target()}:{_strix_workdir()}/"
    cmd = ["rsync", "-az", "--info=progress2", "--delete"] + excludes + [src, dst]
    print(f"[strix] rsync -> {dst}")
    print(f"  excludes: {' '.join(excludes)}")
    return subprocess.run(cmd).returncode


def cmd_copy_down(args: argparse.Namespace) -> int:
    """rsync a checkpoint dir down from Strix to local."""
    remote_path = f"{_strix_target()}:{_strix_workdir()}/{args.remote_path}"
    local_path = args.remote_path  # mirror the path structure
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    cmd = ["rsync", "-az", "--info=progress2", remote_path + "/", local_path + "/"]
    print(f"[strix] rsync from {remote_path}")
    return subprocess.run(cmd).returncode


def cmd_tail_log(args: argparse.Namespace) -> int:
    """Stream the tail of the latest training log on Strix."""
    log_pattern = args.log_pattern or "logs/train_*.log"
    remote_cmd = (
        f"cd {shlex.quote(_strix_workdir())} && "
        f"ls -t {log_pattern} 2>/dev/null | head -1 | xargs tail -F"
    )
    print(f"[strix] tailing: {log_pattern}  (Ctrl-C to stop)")
    return subprocess.run(["ssh", "-t", _strix_target(), remote_cmd]).returncode


def cmd_run(args: argparse.Namespace) -> int:
    """Execute an arbitrary command on Strix in the repo dir."""
    remote_cmd = f"cd {shlex.quote(_strix_workdir())} && {args.cmd}"
    return subprocess.run(["ssh", "-t", _strix_target(), remote_cmd]).returncode


def main() -> int:
    ap = argparse.ArgumentParser(prog="tools.strix_ssh", description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)

    sub.add_parser("check", help="confirm SSH + GPU free")

    p_up = sub.add_parser("copy-up", help="rsync repo to Strix")
    p_up.add_argument("--include-data", action="store_true",
                      help="also rsync data/ (default skipped; large)")
    p_up.set_defaults(func=cmd_copy_up)

    p_down = sub.add_parser("copy-down", help="rsync a checkpoint dir back from Strix")
    p_down.add_argument("remote_path", help="path relative to Strix repo, e.g. checkpoints/lora_v1_strix")
    p_down.set_defaults(func=cmd_copy_down)

    p_tail = sub.add_parser("tail-log", help="tail -F the latest training log on Strix")
    p_tail.add_argument("--log-pattern", default=None,
                        help="glob pattern relative to repo (default logs/train_*.log)")
    p_tail.set_defaults(func=cmd_tail_log)

    p_run = sub.add_parser("run", help="run an arbitrary command in the Strix repo dir")
    p_run.add_argument("cmd", help="the command to run (use quotes for multi-word)")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    if args.action == "check":
        return cmd_check()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
