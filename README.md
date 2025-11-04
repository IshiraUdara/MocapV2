# MocapV2 — Computer Vision Motion Capture

[![License: MIT Personal Use](https://img.shields.io/badge/License-MIT%20Personal%20Use-blue.svg)](LICENSE)

MocapV2 is a Python-based motion-capture toolbox that uses computer-vision methods (OpenCV + NumPy) to detect and track IR markers and cameras. It includes camera SDK integrations (FLIR / Spinnaker, Kinect DK), IMU EKF processing, and utilities to stream or export poses to Unity, Unreal and Blender.

## Table of Contents
- [Key features](#key-features)
- [Repo layout (important folders)](#repo-layout-important-folders)
- [Prerequisites and vendor SDKs](#prerequisites-and-vendor-sdks)
- [Setup (Windows)](#setup-windows)
- [Quick run examples](#quick-run-examples)
- [Blender and Unity / Unreal](#blender-and-unity--unreal)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [License file](LICENSE)

## Key features
- Real-time marker detection and tracking (OpenCV)
- Support for multiple camera backends:
  - FLIR / Spinnaker (PySpin / PySpin SDK)
  - Kinect DK (pykinect2 + Kinect SDK)
  - Generic camera SDK macros and USB cameras
- IMU EKF for orientation/position fusion (IMU_EKF/)
- Integration examples for Unity and Unreal (Unity_Simulation/, with_unreal/)
- Blender scripts to place and extract camera/point data (blender_scripts/)
- Utilities to capture images, run calibration, solve camera intrinsics / extrinsics, and compute object poses
- Project helper script to generate / install Python requirements (main/requirements.py)

## Repo layout (important folders)
- main/ — core scripts and GUI:
  - TakePhotos_*.py, RealtimeTracking_*.py, CapturePoints.py
  - RealtimeTracking_FLIR.py, Kinect_DK_cam/ (Kinect-specific scripts)
  - IMU_EKF/ — EKF, kalman and IMU processing
  - blender_scripts/ — Blender helpers (require Blender Python)
  - Unity_Simulation/ — Unity integration examples
  - unity_communication/ — small helpers for Unity IPC
  - macro_for_cam_SDKs/ — macros and templates for camera SDKs
  - lib/ — helper modules (CudaOperations.py, ImageOperations.py, Helpers.py)
  - captured_images/, floor_images/, checkerboard/ — example data and outputs
- with_unreal/ — BEN2/Unreal integration code and dependencies
- environment.yml, main/requirements.py — environment / requirements helpers

## Prerequisites and vendor SDKs
- Python 3.8+ recommended (project tested on Windows)
- pip, venv or conda for isolated environments
- Hardware/vendor SDKs (not pip-installable):
  - FLIR Spinnaker / PySpin — install from FLIR/Teledyne downloads
  - Kinect for Windows SDK + pykinect2 — pykinect2 may require manual install/build
  - Blender's `bpy` — use Blender's bundled Python to run blender_scripts
  - CUDA / CuPy — GPU packages require matching CUDA toolkit and specific wheels
- Windows-specific: comtypes is required for Kinect / COM wrappers

## Setup (Windows)
1. Create and activate a venv (from project root)
   - PowerShell:
     .\.venv\Scripts\Activate.ps1
   - Command Prompt:
     .\.venv\Scripts\activate.bat

2. Upgrade packaging tools:
   python -m pip install --upgrade pip setuptools wheel

3. Use the helper to inspect / write requirements:
   - Print status:
     python main\requirements.py
   - Generate requirements.txt (pinned versions when available):
     python main\requirements.py --write --output requirements.txt
   - Attempt pip install missing pip-installable packages:
     python main\requirements.py --install

4. If you created requirements.txt:
   python -m pip install -r requirements.txt

Notes:
- Manual SDKs listed as "manual" in requirements.py must be installed separately.
- If you encounter Python-2 idioms in comtypes (SyntaxError / 'unicode' NameError), undo manual edits and reinstall comtypes inside the venv:
  .\.venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir comtypes

## Quick run examples
- FLIR realtime tracking (requires PySpin + camera):
  .\.venv\Scripts\Activate.ps1
  python main\RealtimeTracking_FLIR.py

- Kinect DK realtime / capture (requires Kinect SDK + pykinect2):
  python main\Kinect_DK_cam\RealtimeTracking_Kinect_DK.py

- Run IMU EKF demo:
  python main\IMU_EKF\Main.py

- Capture images for calibration (example):
  python main\TakePhotos_dual-Extrinsics.py
  Captured images saved under main\captured_images\cam0 and cam1 (or camera-specific subfolders).

- Unity integration example (local messaging):
  python main\unity_communication\ToUnity.py
  python main\Unity_Simulation\Unity_livecamera_feed.py

## Blender and Unity / Unreal
- Blender scripts require running inside Blender or using Blender's Python executable. See blender_scripts/*.py.
- with_unreal/ contains BEN2 model integration; follow its README and with_unreal/ben2/requirements.txt for model-specific deps.
- Unity_Simulation folder contains Unity import/export helpers — Unity side expects UDP/socket messages or file-based textures.

## Troubleshooting
- Confirm you are using the project's venv: python -c "import sys; print(sys.executable)"
- comtypes / pykinect2 errors:
  - Reinstall comtypes in the active venv: python -m pip install --force-reinstall comtypes
  - pykinect2 may not be on PyPI; follow its GitHub install instructions or ensure Kinect SDK is installed.
- If OpenCV is missing, install opencv-python; if you need contrib modules, install opencv-contrib-python.
- GPU packages (numba/cupy/cuda) must match your CUDA toolkit and GPU drivers.

## Contributing
- Create a branch, add tests if applicable, and open a PR. Keep changes modular (e.g., add camera backend under macro_for_cam_SDKs or new subfolder).
- Update main/requirements.py when adding new Python dependencies.

## License
This project is licensed under the MIT Personal Use License. See [LICENSE](LICENSE) for the full text.




