from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("PathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("OptixDenoiser.dll")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1, 'maxSurfaceBounces': 9, 'maxDiffuseBounces': 9, 'maxSpecularBounces': 9, 'maxTransmissionBounces': 9, 'useRTXDI': True, 'useReSTIRPT': True, 'emissiveSampler': EmissiveLightSamplerType.Power, 'useLambertianDiffuse': True})
    g.addPass(PathTracer, "PathTracer")
    GBufferRT = createPass("GBufferRT", {'samplePattern': SamplePattern.Center, 'sampleCount': 1, 'useAlphaTest': True})
    g.addPass(GBufferRT, "GBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': AccumulatePrecision.Single})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    OptixDenoiser = createPass("OptixDenoiser")
    g.addPass(OptixDenoiser, "OptixDenoiser")

    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("GBufferRT.mvec", "PathTracer.mvec")

    g.addEdge("GBufferRT.mvec", "OptixDenoiser.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "OptixDenoiser.color")
    g.addEdge("PathTracer.albedo", "OptixDenoiser.albedo")
    g.addEdge("GBufferRT.normW", "OptixDenoiser.normal")

    g.addEdge("OptixDenoiser.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")

    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

# compress hit Info?
m.setMogwaiUISceneBuilderFlag(buildFlags = SceneBuilderFlags.UseCompressedHitInfo)

