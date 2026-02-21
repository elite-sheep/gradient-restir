from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("WarpDiff")
    WARDiffPathTracer = createPass("WARDiffPathTracer", {'samplesPerPixel': 1})
    g.addPass(WARDiffPathTracer, "WARDiffPathTracer")
    # VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    # g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    # ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    # g.addPass(ToneMapper, "ToneMapper")
    # g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    # g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    # g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("WARDiffPathTracer.color", "AccumulatePass.input")
    # g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("AccumulatePass.output")
    return g

WarDiffPT = render_graph_PathTracer()
try: m.addGraph(WarDiffPT)
except NameError: None
