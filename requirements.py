"""
requirements.py - helper to check/install project dependencies and emit requirements.txt

Usage:
    python requirements.py                # print status
    python requirements.py --write       # write requirements.txt
    python requirements.py --install     # try to pip-install missing packages
    python requirements.py --install --write
"""
from __future__ import annotations

import sys
import subprocess
import argparse
from typing import List, Dict, Optional

try:
    # Python 3.8+
    from importlib import metadata as importlib_metadata  # type: ignore
except Exception:
    import importlib_metadata  # type: ignore

REQUIRED_PACKAGES: List[Dict[str, Optional[str]]] = [
    # runtime libs used throughout the project
    {"name": "numpy", "pip": "numpy", "notes": None},
    {"name": "opencv-python", "pip": "opencv-python", "notes": "used as cv2"},
    {"name": "scipy", "pip": "scipy", "notes": None},
    {"name": "matplotlib", "pip": "matplotlib", "notes": None},
    {"name": "Pillow", "pip": "Pillow", "notes": "used as PIL"},
    {"name": "pyserial", "pip": "pyserial", "notes": "serial port communications"},
    {"name": "msgpack", "pip": "msgpack", "notes": "msgpack for socket comms"},
    {"name": "filterpy", "pip": "filterpy", "notes": "EKF implementation"},
    {"name": "numpy-quaternion", "pip": "numpy-quaternion", "notes": "install as numpy-quaternion (import quaternion)"},
    {"name": "numba", "pip": "numba", "notes": "required for CUDA kernels in lib/CudaOperations.py"},
    {"name": "msgpack", "pip": "msgpack", "notes": None},
    {"name": "tqdm", "pip": "tqdm", "notes": None},
    # Optional / system-specific
    {"name": "PySpin", "pip": None, "notes": "FLIR Spinnaker SDK — install manually from FLIR / not pip"},
    {"name": "cupy", "pip": "cupy", "notes": "optional: GPU numpy alternative (not required)"},
]

def get_version(pkg_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(pkg_name)
    except Exception:
        return None

def check_requirements() -> Dict[str, Dict[str, Optional[str]]]:
    """
    Return dict: pip_name_or_name -> {"required":pip, "installed_version":ver, "status": "ok"/"missing"/"manual"}
    """
    report = {}
    for pkg in REQUIRED_PACKAGES:
        pip_name = pkg["pip"] or pkg["name"]
        ver = get_version(pip_name)
        status = "ok" if ver else ("manual" if pkg["pip"] is None else "missing")
        report[pip_name] = {
            "required": pip_name,
            "installed_version": ver,
            "status": status,
            "notes": pkg.get("notes"),
        }
    return report

def install_packages(packages: List[str]) -> int:
    """
    Install packages via pip. Returns pip return code (0=success).
    """
    if not packages:
        print("No packages to install.")
        return 0

    cmd = [sys.executable, "-m", "pip", "install"] + packages
    print("Running:", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except Exception as e:
        print("Failed to run pip:", e)
        return 1

def write_requirements_txt(path: str = "requirements.txt") -> None:
    """
    Write a requirements.txt with pinned installed versions when available.
    """
    lines = []
    for pkg in REQUIRED_PACKAGES:
        pip_name = pkg["pip"] or pkg["name"]
        ver = get_version(pip_name)
        if ver:
            lines.append(f"{pip_name}=={ver}")
        else:
            # keep package name if no installed version; mark optional/manual
            lines.append(f"# {pip_name}  # not installed (notes: {pkg.get('notes')})")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Project requirements helper")
    p.add_argument("--install", action="store_true", help="Attempt to pip install missing packages")
    p.add_argument("--write", action="store_true", help="Write requirements.txt with pinned versions")
    p.add_argument("--output", type=str, default="requirements.txt", help="requirements.txt output path")
    args = p.parse_args(argv)

    report = check_requirements()

    missing = []
    print("\nDependency status:")
    for name, info in sorted(report.items()):
        status = info["status"]
        ver = info["installed_version"] or "-"
        note = f" ({info['notes']})" if info.get("notes") else ""
        print(f" - {name:25} : {status:8}   installed: {ver}{note}")
        if status == "missing":
            missing.append(name)

    if args.install and missing:
        print("\nAttempting to install missing packages via pip...")
        code = install_packages(missing)
        if code != 0:
            print("pip install returned non-zero exit code:", code)
            return code
        # Re-check
        report = check_requirements()
        print("\nPost-install status:")
        for name, info in sorted(report.items()):
            print(f" - {name:25} : {info['status']:8}   installed: {info['installed_version'] or '-'}")
    elif args.install and not missing:
        print("No missing pip-installable packages found.")

    if args.write:
        write_requirements_txt(args.output)

    return 0

if __name__ == "__main__":
    sys.exit(main())