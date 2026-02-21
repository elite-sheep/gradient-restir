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

### 1. Clone the repository

```bash
git clone --recursive git@github.com:elite-sheep/gradient-restir.git
cd gradient-restir
```

### 2. Fetch dependencies

```bash
setup.bat
```

This script downloads external dependencies via packman and initializes git submodules.

### 3. Build with Visual Studio 2022

```bash
setup_vs2022.bat
```

This generates a Visual Studio solution at `build/windows-vs2022/Falcor.sln`. Open it in Visual Studio, set the build configuration to **Release**, and build the solution. The output binaries are located in `build/windows-vs2022/bin/Release/`.

Alternatively, build from the command line:

```bash
cmake --preset windows-vs2022
cmake --build build/windows-vs2022 --config Release
```

### Build with VS Code / Ninja

```bash
setup.bat
```

Open the project folder in VS Code, select the **Windows Ninja/MSVC** configure preset (`Ctrl+Shift+P` -> _CMake: Select Configure Preset_), then press `F7` to build. Binaries are output to `build/windows-ninja-msvc/bin/`.

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
