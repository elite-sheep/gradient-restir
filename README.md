# Gradient-domain ReSTIR Path Tracing

This repository contains the source code for the paper:

**[Gradient-domain ReSTIR Path Tracing](https://bulbaberry.xyz/publications/restir-gpt-eg-2026)**
Yu-Chen Wang, Markus Kettunen, Daqi Lin, Chris Wyman, Lifan Wu, Shuang Zhao
*Eurographics 2026*

The implementation is built on top of NVIDIA's [Falcor](https://github.com/NVIDIAGameWorks/Falcor) rendering framework (version 7.0). See [README_Falcor.md](README_Falcor.md) for the original Falcor documentation.

## Prerequisites

- Windows 10 version 20H2 or newer
- Visual Studio 2022
- [Windows 10 SDK (10.0.19041.0)](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
- [CMake](https://cmake.org) (3.20 or newer)
- A GPU with DirectX Raytracing support (e.g., NVIDIA RTX series)
- NVIDIA driver 466.11 or newer

## Building

```bash
git clone --recursive git@github.com:elite-sheep/gradient-restir.git
cd gradient-restir
setup_vs2022.bat
msbuild build/windows-vs2022/Falcor.sln /p:Configuration=Release /m
```

The output binaries are located in `build/windows-vs2022/bin/Release/`.

Note: Run the above commands from a **Developer Command Prompt for VS 2022** (or run `vcvarsall.bat` first) so that `msbuild` is available on the PATH.

## Running with Mogwai

**Mogwai** is Falcor's interactive rendering application. It loads render graphs defined as Python scripts and executes them on loaded scenes.

### Quick Start

```bash
# From the build output directory, e.g.:
build/windows-vs2022/bin/Release/Mogwai.exe -s scripts/GPathTracer.py -S <path-to-scene.pyscene>
```

This loads the GPathTracer render graph and a scene file, then starts the interactive renderer.

### Loading a Script and Scene via GUI

1. Launch `Mogwai.exe`
2. Load a render graph script: **File -> Load Script** (`Ctrl+O`) and select `scripts/GPathTracer.py`
3. Load a scene: **File -> Load Scene** (`Ctrl+Shift+O`) and select a `.pyscene` file

### Render Graph Script

The provided `scripts/GPathTracer.py` sets up a render graph with:

- **VBufferRT** -- Generates the visibility buffer (primary ray intersections)
- **GPathTracer** -- Gradient-domain path tracer with multi-point ReSTIR
- **AccumulatePass** -- Accumulates per-frame results over time

Key parameters in the script:

| Parameter | Default | Description |
|---|---|---|
| `samplesPerPixel` | 1 | Number of samples per pixel per frame |
| `maxBounces` | 5 | Maximum path length |
| `useNEE` | True | Enable next event estimation |
| `useTemporalReuse` | True | Enable temporal sample reuse |
| `useSpatialReuse` | True | Enable spatial sample reuse |
| `numSpatialNeighbours` | 1 | Number of spatial neighbors for reuse |
| `numInitialSamples` | 1 | Number of initial RIS candidates |
| `shiftMappingType` | 2 | Shift mapping strategy (0: reconnection, 1: replay, 2: hybrid) |

### Headless Rendering

For batch/offline rendering without a GUI window:

```bash
Mogwai.exe --headless -s scripts/GPathTracer.py -S <path-to-scene.pyscene>
```

### Useful Keyboard Shortcuts

| Key | Action |
|---|---|
| `F2` | Toggle GUI visibility |
| `F5` | Reload shaders |
| `F12` | Capture screenshot |
| `Shift+F12` | Start/stop video recording |
| `` ` `` (backtick) | Open Python console |
| `P` | Toggle profiler |

## Batch Rendering with Python Scripts

The `python/` directory contains Python scripts for headless batch rendering and evaluation using Falcor's Python bindings. These scripts must be run with the Falcor-embedded Python interpreter (i.e., via `Mogwai.exe` or `FalcorPython.exe`).

### Prerequisites

Install the required Python packages:

```bash
pip install numpy pyexr tqdm yacs pympler
```

For Poisson reconstruction (`recon.py`), you also need the [screen-poisson-py](https://github.com/elite-sheep/screen-poisson-py) library built and accessible.

### Scripts Overview

| Script | Description |
|---|---|
| `python/render.py` | Batch rendering for static scenes (ground truth, gradient images, primal/reconstructed images) |
| `python/render_dynamic.py` | Batch rendering for dynamic scenes with camera motion |
| `python/config.py` | Configuration system using YACS (`GPathTracerConfig`, `CameraMotionConfig`) |
| `python/common.py` | Shared utilities for testbed creation, scene loading, and render pass setup |
| `python/recon.py` | Poisson reconstruction from gradient-domain renderings |
| `python/camera_pbrt.py` | Utility to convert vertical FOV to focal length |

### Rendering Ground Truth

```bash
python python/render.py gt --scene <path-to-scene.pyscene> --w 512 --h 512 --spp 16384 --max_bounces 5 --output_dir results/gt
```

This renders a high-spp ground truth image along with gradient images (DX, DY).

### Rendering with a Config File

Create a YAML config file (see `python/config.py` for all available options):

```yaml
type: "final"
scene_path: "path/to/scene.pyscene"
gt_path: "results/gt/gt_primal.exr"
max_bounces: 5
diff_integrator: 1
shift_mapping_type: 2
use_temporal: true
use_spatial: true
num_spatial_neighbours: 4
num_initial_samples: 2
num_iters: 128
num_images: 4
resolution: [1920, 1080]
output_dir: "results/output"
method_name: "multirestir"
```

Then run:

```bash
# Static scene rendering
python python/render.py render --config_file <path-to-config.yaml>

# Dynamic scene rendering (with camera motion)
python python/render_dynamic.py render --config_file <path-to-config.yaml>
```

The `type` field in the config selects the rendering mode:

| Type | Description |
|---|---|
| `final` | Full rendering (primal + DX/DY gradients + Poisson reconstruction) |
| `dx` | Gradient (DX) image only |
| `primal` | Primal (reconstructed) image only |
| `primal_restir` | Primal image using ReSTIR |
| `primal_gpt` | Primal image using gradient-domain path tracing |
| `gpt_dx` | Gradient image using gradient-domain path tracing |

### Poisson Reconstruction

After rendering gradient-domain images, reconstruct the final image:

```bash
python python/recon.py --input_dir <rendering-output-dir> --num_images <N> --recon_type default --output_dir recon
```

This reads `primal/primal_*.exr`, `dx/dx_*.exr`, and `dy/dy_*.exr` from the input directory and writes reconstructed images to the output subdirectory.

## Project Structure

```
Source/
  RenderPasses/
    GPathTracer/           # Gradient-domain path tracer (this project)
      GPathTracer.cpp/h    # Main render pass
      DiffIntegrator.*     # Differential integrator (multi-point ReSTIR)
      MultiPoint*.slang    # Multi-point ReSTIR temporal/spatial reuse shaders
      Primal*.slang        # Primal ReSTIR shaders
      PathShift.slang      # Gradient-domain shift mapping
      PoissonSolver.*      # Poisson reconstruction
  Falcor/                  # Core rendering framework
  Mogwai/                  # Interactive rendering application
scripts/
  GPathTracer.py           # Render graph script for GPathTracer
python/
  render.py                # Batch rendering for static scenes
  render_dynamic.py        # Batch rendering for dynamic scenes
  config.py                # YACS-based configuration
  common.py                # Shared utilities
  recon.py                 # Poisson reconstruction
  camera_pbrt.py           # FOV-to-focal-length utility
```

## Citation

```bibtex
@inproceedings{Wang2026GradientReSTIR,
    title   = {Gradient-domain ReSTIR Path Tracing},
    author  = {Wang, Yu-Chen and Kettunen, Markus and Lin, Daqi and Wyman, Chris and Wu, Lifan and Zhao, Shuang},
    booktitle = {Eurographics},
    year    = {2026}
}
```

## Acknowledgments

This project is built on [Falcor](https://github.com/NVIDIAGameWorks/Falcor) by NVIDIA. See [README_Falcor.md](README_Falcor.md) for Falcor's documentation and [LICENSE.md](LICENSE.md) for license details.
