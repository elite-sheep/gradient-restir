from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1, 'maxSurfaceBounces': 9, 'maxDiffuseBounces': 9, 'maxSpecularBounces': 9, 'maxTransmissionBounces': 9, 'useRTXDI': True, 'useReSTIRPT': 1, 'emissiveSampler': 'Power', 'useLambertianDiffuse': True})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Center', 'sampleCount': 1, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Double'})
    g.addPass(AccumulatePass, "AccumulatePass")
    AccumulatePass2 = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Double'})
    g.addPass(AccumulatePass2, "AccumulatePass2")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    g.markOutput("PathTracer.krgb")
    g.markOutput("PathTracer.variance");
    g.addEdge("PathTracer.krgb", "AccumulatePass2.input")
    g.markOutput("AccumulatePass2.output")
    g.markOutput("AccumulatePass.output")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None
