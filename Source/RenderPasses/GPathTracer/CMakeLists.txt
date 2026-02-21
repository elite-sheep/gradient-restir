add_plugin(GPathTracer)

target_sources(GPathTracer PRIVATE
    CameraUtils.slang
    DiffIntegrator.cpp
    DiffIntegrator.h
    DiffIntegrator.slang
    DiffIntegratorInterface.slang
    DiffIntegratorType.slangh
    EvaluatePass.cs.slang
    Params.slang
    GPathTracer.cpp
    GPathTracer.h
    GPathTracer.slang
    MultiPointReSTIR.slang
    MultiPointTemporalLookForNeighbour.cs.slang
    MultiPointTemporalReuse.cs.slang
    MultiPointTemporalNegReuse.cs.slang
    MultiPointSpatialLookForNeighbour.cs.slang
    MultiPointSpatialReuse.cs.slang
    MultiPointSpatialNegReuse.cs.slang
    PathShift.slang
    PathState.slang
    PoissonSolver.cs.slang
    PrepareBasePaths.cs.slang
    PrimalSampleInitialSamples.cs.slang
    PrimalTemporalLookForNeighbour.cs.slang
    PrimalTemporalReuse.cs.slang
    PrimalSpatialReuse.cs.slang
    PrimaryEmission.cs.slang
    PrimalReSTIR.slang
    ReflectTypes.cs.slang
    RISDataTypes.slang
    TracingUtils.slang
    StaticParams.slang
    Utils.slang
)

target_copy_shaders(GPathTracer RenderPasses/GPathTracer)

target_source_group(GPathTracer "RenderPasses")
