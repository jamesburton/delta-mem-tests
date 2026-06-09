"""SSH coordination helper for the Strix Halo training box.

Wraps the common operations we'd do over SSH from this Windows host to
the Strix Halo box during training runs:

  - check        : confirm SSH reachable + GPU is free (not running other jobs)
  - copy-up      : send the repo state up to Strix (rsync if available, else scp -r)
  - copy-down    : pull a checkpoint dir back from Strix
  - tail-log     : stream the latest training log
  - run          : execute an arbitrary command on Strix

Environment variables:

  STRIX_SSH_TARGET   SSH alias / user@host (default 'strix')
  STRIX_REPO_DIR     Remote repo path. Default depends on STRIX_SHELL:
                       STRIX_SHELL=cmd | powershell -> C:\\Users\\james\\delta-mem-tests
                       STRIX_SHELL=bash              -> /home/james/delta-mem-tests
  STRIX_SHELL        Remote shell flavour for wrapping commands:
                       'cmd'        (default) - Windows cmd.exe lands on SSH;
                                                 commands passed through raw.
                       'powershell' - wraps in `powershell.exe -NoProfile -Command "..."`.
                       'bash'       - wraps in `bash -lc "..."` (legacy Linux/WSL).

On a Windows local host we use the native OpenSSH
(C:\\Windows\\System32\\OpenSSH\\ssh.exe) to avoid the msys2/Git-Bash
ssh.exe that fails with `libcrypto-3-x64.dll not found`.

Usage from this Windows host:

    python -m tools.strix_ssh check
    python -m tools.strix_ssh copy-up
    python -m tools.strix_ssh tail-log
    python -m tools.strix_ssh copy-down checkpoints/longctx-v1-32k
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


_VALID_SHELLS = {"cmd", "powershell", "bash"}


def _strix_target() -> str:
    return os.environ.get("STRIX_SSH_TARGET", "strix")


def _strix_shell() -> str:
    s = os.environ.get("STRIX_SHELL", "cmd").lower()
    if s not in _VALID_SHELLS:
        print(f"[strix] WARN: STRIX_SHELL={s!r} not in {_VALID_SHELLS}; "
              f"defaulting to 'cmd'", file=sys.stderr)
        s = "cmd"
    return s


def _strix_workdir() -> str:
    default_windows = r"C:\Users\james\delta-mem-tests"
    default_linux = "/home/james/delta-mem-tests"
    fallback = default_linux if _strix_shell() == "bash" else default_windows
    return os.environ.get("STRIX_REPO_DIR", fallback)


def _ssh_exe() -> str:
    if os.name == "nt":
        native = Path(r"C:\Windows\System32\OpenSSH\ssh.exe")
        if native.exists():
            return str(native)
    found = shutil.which("ssh")
    return found if found else "ssh"


def _scp_exe() -> str:
    if os.name == "nt":
        native = Path(r"C:\Windows\System32\OpenSSH\scp.exe")
        if native.exists():
            return str(native)
    return shutil.which("scp") or "scp"


def _wrap_for_remote_shell(cmd: str) -> str:
    """Wrap a command string per STRIX_SHELL.

    cmd        : passed through raw (the SSH server starts cmd.exe).
    powershell : `powershell.exe -NoProfile -Command "<escaped>"`.
    bash       : `bash -lc "<escaped>"`.
    """
    shell = _strix_shell()
    if shell == "bash":
        return f"bash -lc {shlex.quote(cmd)}"
    if shell == "powershell":
        escaped = cmd.replace('"', '""')
        return f'powershell.exe -NoProfile -Command "{escaped}"'
    return cmd


def _ssh_run(remote_cmd: str, capture: bool = True,
             timeout: int = 30) -> subprocess.CompletedProcess:
    wrapped = _wrap_for_remote_shell(remote_cmd)
    full = [_ssh_exe(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            _strix_target(), wrapped]
    return subprocess.run(full, capture_output=capture, text=True,
                          timeout=timeout)


def _parse_nvidia_smi_csv(output: str) -> list[dict]:
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


def _parse_rocm_smi(output: str) -> list[dict]:
    """Best-effort parse of `rocm-smi --showuse --showmemuse` plaintext."""
    util = mem = None
    for line in output.splitlines():
        s = line.strip()
        if "GPU use" in s and "%" in s and "memory" not in s.lower():
            try:
                util = int(s.split(":")[-1].replace("%", "").strip())
            except ValueError:
                pass
        elif "GPU memory use" in s and "%" in s:
            try:
                mem = int(s.split(":")[-1].replace("%", "").strip())
            except ValueError:
                pass
    if util is None and mem is None:
        return []
    return [{
        "name": "Radeon 8060S (Strix Halo iGPU)",
        "util_pct": str(util if util is not None else 0),
        "mem_pct": str(mem if mem is not None else 0),
    }]


def cmd_check() -> int:
    target = _strix_target()
    shell = _strix_shell()
    print(f"[strix] target: {target}  shell: {shell}")
    try:
        if shell == "bash":
            probe = "uname -a"
        elif shell == "powershell":
            probe = "[System.Environment]::OSVersion.VersionString"
        else:
            probe = "ver"
        result = _ssh_run(probe, timeout=15)
        if result.returncode != 0:
            print(f"  SSH FAILED: {result.stderr.strip()}")
            print(f"  hint: set $env:STRIX_SSH_TARGET; verify: ssh {target} {probe}")
            return 1
        print(f"  SSH OK: {result.stdout.strip()}")

        rocm = _ssh_run("rocm-smi --showuse --showmemuse", timeout=15)
        if rocm.returncode == 0 and rocm.stdout.strip():
            gpus = _parse_rocm_smi(rocm.stdout)
            if gpus:
                any_busy = False
                for i, g in enumerate(gpus):
                    util = int(g["util_pct"])
                    mem_pct = int(g.get("mem_pct", "0"))
                    status = "BUSY" if (util > 5 or mem_pct > 10) else "FREE"
                    if status == "BUSY":
                        any_busy = True
                    print(f"  GPU{i} ({g['name']}): util={util}%  "
                          f"mem={mem_pct}%  -> {status}")
                if any_busy:
                    print("\n  GPU appears in use. Coordinate before submitting.")
                    return 10
                print("\n  -> Strix GPU is FREE. Safe to submit a training job.")
                return 0

        nv = _ssh_run(
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits", timeout=15,
        )
        if nv.returncode != 0:
            print(f"  rocm-smi / nvidia-smi both failed.")
            print(f"  rocm-smi stderr: {rocm.stderr.strip()[:200]}")
            print(f"  nvidia-smi stderr: {nv.stderr.strip()[:200]}")
            print(f"  hint: confirm ROCm-on-Windows is installed and rocm-smi is on PATH")
            return 2
        gpus = _parse_nvidia_smi_csv(nv.stdout)
        if not gpus:
            print(f"  no GPUs reported by nvidia-smi -- check driver state on Strix")
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
            print(f"  GPU{i} ({g['name']}): util={util}%  "
                  f"mem={mem_used}/{mem_total} MiB ({mem_pct:.1f}%)  -> {status}")
        if any_busy:
            print("\n  GPU appears in use. Active processes:")
            result = _ssh_run(
                "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                "--format=csv,noheader,nounits", timeout=15,
            )
            print(f"  {result.stdout.strip() or '(none reported)'}")
            return 10
        print("\n  -> Strix GPU is FREE. Safe to submit a training job.")
        return 0
    except subprocess.TimeoutExpired:
        print(f"  SSH TIMEOUT to {target}")
        return 4


def _have_rsync() -> bool:
    return shutil.which("rsync") is not None


def cmd_copy_up(args: argparse.Namespace) -> int:
    excludes = [".venv", "__pycache__", ".git/objects", "outputs",
                "report/raw", "checkpoints"]
    if not args.include_data:
        excludes.append("data")
    dst = f"{_strix_target()}:{_strix_workdir()}/"

    if _have_rsync():
        rsync_excludes: List[str] = [f"--exclude={e}/" for e in excludes]
        cmd = (["rsync", "-az", "--info=progress2", "--delete"]
               + rsync_excludes + ["./", dst])
        print(f"[strix] rsync -> {dst}")
        print(f"  excludes: {' '.join(rsync_excludes)}")
        return subprocess.run(cmd).returncode

    print(f"[strix] rsync not found; falling back to scp -r (slower, no --delete)")
    print(f"[strix] hint: install Git for Windows or MSYS2 to get rsync")
    rc = 0
    skip_hidden = {".github", ".gitignore", ".claude", ".venv"}
    for entry in sorted(Path(".").iterdir()):
        if entry.name in excludes or entry.name in skip_hidden:
            continue
        if entry.name.startswith(".") and entry.name not in (".planning",):
            continue
        target_path = f"{dst}{entry.name}"
        cmd = [_scp_exe(), "-r", str(entry), target_path]
        print(f"  scp {entry} -> {target_path}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            rc = result.returncode
    return rc


def cmd_copy_down(args: argparse.Namespace) -> int:
    remote_rel = args.remote_path.replace("\\", "/")
    workdir = _strix_workdir().replace("\\", "/")
    remote_path = f"{_strix_target()}:{workdir}/{remote_rel}"
    local_path = args.remote_path
    local_parent = os.path.dirname(local_path) or "."
    os.makedirs(local_parent, exist_ok=True)

    if _have_rsync():
        cmd = ["rsync", "-az", "--info=progress2",
               remote_path + "/", local_path + "/"]
        print(f"[strix] rsync from {remote_path}")
        return subprocess.run(cmd).returncode

    print(f"[strix] rsync not found; falling back to scp -r (slower)")
    cmd = [_scp_exe(), "-r", remote_path, local_path]
    print(f"[strix] scp -r {remote_path} -> {local_path}")
    return subprocess.run(cmd).returncode


def cmd_tail_log(args: argparse.Namespace) -> int:
    shell = _strix_shell()
    log_pattern = args.log_pattern
    if shell == "bash":
        if not log_pattern:
            log_pattern = "logs/train_*.log"
        remote_cmd = (
            f"cd {shlex.quote(_strix_workdir())} && "
            f"ls -t {log_pattern} 2>/dev/null | head -1 | xargs tail -F"
        )
        wrapped = _wrap_for_remote_shell(remote_cmd)
    else:
        if not log_pattern:
            log_pattern = r"logs\train_*.log"
        ps_expr = (
            f"Set-Location '{_strix_workdir()}'; "
            f"$f = Get-ChildItem -Path '{log_pattern}' -ErrorAction SilentlyContinue "
            f"| Sort-Object LastWriteTime -Descending | Select-Object -First 1; "
            f"if ($null -eq $f) {{ Write-Host '[strix] no log matched {log_pattern}'; exit 1 }}; "
            f"Write-Host \"[strix] tailing $($f.FullName)\"; "
            f"Get-Content -Path $f.FullName -Wait -Tail 30"
        )
        if shell == "cmd":
            escaped = ps_expr.replace('"', '""')
            wrapped = f'powershell.exe -NoProfile -Command "{escaped}"'
        else:
            wrapped = _wrap_for_remote_shell(ps_expr)

    print(f"[strix] tailing: {log_pattern}  (Ctrl-C to stop)")
    return subprocess.run([_ssh_exe(), "-t", _strix_target(), wrapped]).returncode


def cmd_run(args: argparse.Namespace) -> int:
    shell = _strix_shell()
    workdir = _strix_workdir()
    if shell == "bash":
        remote_cmd = f"cd {shlex.quote(workdir)} && {args.cmd}"
    elif shell == "powershell":
        remote_cmd = f"Set-Location '{workdir}'; {args.cmd}"
    else:
        remote_cmd = f'cd /d "{workdir}" && {args.cmd}'
    wrapped = _wrap_for_remote_shell(remote_cmd)
    return subprocess.run([_ssh_exe(), "-t", _strix_target(), wrapped]).returncode


def main() -> int:
    ap = argparse.ArgumentParser(prog="tools.strix_ssh", description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)

    sub.add_parser("check", help="confirm SSH + GPU free")

    p_up = sub.add_parser("copy-up", help="rsync/scp repo to Strix")
    p_up.add_argument("--include-data", action="store_true",
                      help="also send data/ (default skipped; large)")
    p_up.set_defaults(func=cmd_copy_up)

    p_down = sub.add_parser("copy-down", help="rsync/scp a checkpoint dir back from Strix")
    p_down.add_argument("remote_path", help="path relative to Strix repo")
    p_down.set_defaults(func=cmd_copy_down)

    p_tail = sub.add_parser("tail-log", help="tail -F the latest training log on Strix")
    p_tail.add_argument("--log-pattern", default=None,
                        help="glob pattern relative to repo")
    p_tail.set_defaults(func=cmd_tail_log)

    p_run = sub.add_parser("run", help="run an arbitrary command in the Strix repo dir")
    p_run.add_argument("cmd", help="the command to run. Wrapped per STRIX_SHELL.")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    if args.action == "check":
        return cmd_check()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
