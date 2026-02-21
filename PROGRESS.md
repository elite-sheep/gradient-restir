# Release Branch Preparation Progress

## Branch: `release` (based on `yuchen/refine_spatial_reuse`)

## Goal
Simplify the GPathTracer render pass by keeping only `DIFF_INTEGRATOR_MULTI_POINT_RESTIR` and removing the two unused integrator types (`DIFF_INTEGRATOR_SINGLE_POINT_RESTIR` and `DIFF_INTEGRATOR_SINGLE_POINT_RESTIR_2`).

## Changes Made

### Files Deleted
- `Source/RenderPasses/GPathTracer/SinglePointReSTIR.slang` - Single-point ReSTIR integrator implementation
- `Source/RenderPasses/GPathTracer/SinglePointReSTIR2.slang` - Single-point ReSTIR variant 2 implementation
- `Source/RenderPasses/GPathTracer/TracePass.cs.slang` - Unused (fully commented out) compute shader
- `Source/RenderPasses/GPathTracer/TemporalReuse.cs.slang` - Unused (fully commented out) compute shader
- `Source/RenderPasses/GPathTracer/SpatialReuse.cs.slang` - Unused (fully commented out) compute shader

### Files Modified

#### `DiffIntegratorType.slangh`
- Removed `DIFF_INTEGRATOR_SINGLE_POINT_RESTIR` (0) and `DIFF_INTEGRATOR_SINGLE_POINT_RESTIR_2` (2) defines
- Only `DIFF_INTEGRATOR_MULTI_POINT_RESTIR` (1) remains

#### `DiffIntegrator.slang`
- Removed conditional imports for SinglePointReSTIR and SinglePointReSTIR2
- Default typedef now uses MultiPointReSTIR
- Removed SinglePoint branches from `DiffTemporalReuse()` and `DiffSpatialReuse()`

#### `DiffIntegrator.h`
- Removed `finalizeTemporalReuse()` declaration (was only used by SinglePoint path)
- Removed SinglePoint buffer members (`mpBaseReservoirs`, `mpBaseTmpReservoirs`, `mpBaseTemporalReservoirs`, `mpBaseSpatialReservoirs`, `mpShiftReservoirs`, `mpShiftTmpReservoirs`, `mpShiftTemporalReservoirs`, `mpShiftSpatialReservoirs`)
- Removed `createSingePointReSTIRReservoirs()` declaration

#### `DiffIntegrator.cpp`
- Removed registration of `DIFF_INTEGRATOR_SINGLE_POINT_RESTIR` and `DIFF_INTEGRATOR_SINGLE_POINT_RESTIR_2`
- Simplified `endFrame()` - removed SinglePoint resource copy logic
- Simplified `prepareSpatialReuse()` - removed SinglePoint branch
- Simplified `bindShaderData()` - removed SinglePoint buffer binding, only MultiPoint bindings remain
- Simplified `prepareBuffers()` - removed SinglePoint branch
- Removed `createSingePointReSTIRReservoirs()` implementation

#### `ReflectTypes.cs.slang`
- Removed `#if`/`#elif` conditional compilation for SinglePoint vs MultiPoint buffer declarations
- Multi-point buffers are now declared unconditionally

#### `CMakeLists.txt`
- Removed entries for deleted files: `SinglePointReSTIR.slang`, `SinglePointReSTIR2.slang`, `TracePass.cs.slang`, `TemporalReuse.cs.slang`, `SpatialReuse.cs.slang`

#### `GPathTracer.cpp`
- Removed unused shader file path constants (`kTracePassShaderFile`, `kTemporalReuseShaderFile`, `kSpatialReuseShaderFile`)
- Removed `kDiffIntegratorType` property string and its parsing in constructor
- Removed "Pixel Diff Integrator" dropdown from `renderRenderingUI()`
- Cleaned up extensive commented-out code in `temporalReuse()`, `sampleInitialSamples()`, `endFrame()`, `prepareResources()`, `prepareTracePass()`, `updatePrograms()`, and `prepareEmissiveLighting()`

#### `GPathTracer.h`
- Removed commented-out `mpTracePass`, `mpTemporalReusePass`, `mpSpatialReusePass` members

## Verification
- All files referenced in `CMakeLists.txt` exist on disk
- No remaining references to `SINGLE_POINT` or `SinglePoint` in the GPathTracer directory
- No stale references to deleted shader files
- `StaticParams::diffIntegrator` defaults to `DIFF_INTEGRATOR_MULTI_POINT_RESTIR` (hardcoded)

## Status: Complete
All SinglePoint integrator types have been removed. Only `DIFF_INTEGRATOR_MULTI_POINT_RESTIR` remains as the sole diff integrator type.
