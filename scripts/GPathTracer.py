from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("GPathTracer")
    gVBufferParams = {
        'samplePattern': "Center",
        'sampleCount': 1,
        'useAlphaTest': True,
        'subPixelRandom' : "UnitQuad",
        'useDOF' : False
    }
    VBufferRT = createPass(
        "VBufferRT",
        gVBufferParams)
    g.addPass(VBufferRT, "VBufferRT")
    GPathTracer = createPass("GPathTracer", {'samplesPerPixel': 1,
                                             'numInitialSamples': 1,
                                             'diffIntegrator': 1,
                                             'maxBounces': 5,
                                             'useNEE': True,
                                             'useTemporalReuse': True,
                                             'useSpatialReuse':  True,
                                             'numSpatialNeighbours': 1,
                                             'shiftMappingType': 2})
    g.addPass(GPathTracer, "GPathTracer")
    g.add_edge("VBufferRT.vbuffer", "GPathTracer.vbuffer")
    g.add_edge("VBufferRT.vbufferCenter", "GPathTracer.vbufferCenter")
    g.add_edge("VBufferRT.viewW", "GPathTracer.viewW")
    g.addEdge("VBufferRT.subPixelUV", "GPathTracer.subPixelUV")
    g.add_edge("VBufferRT.mvec", "GPathTracer.mvec")
    # g.addPass(VBufferRT, "VBufferRT")
    AccumulatePassG = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePassG, "AccumulatePassG")
    g.addEdge("GPathTracer.DX", "AccumulatePassG.input")
    AccumulatePassPrimal = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePassPrimal, "AccumulatePassPrimal")
    g.addEdge("GPathTracer.primal", "AccumulatePassPrimal.input")
    # g.markOutput("GPathTracer.DX")
    g.markOutput("AccumulatePassG.output")
    # g.markOutput("GPathTracer.primal")
    return g

GPathTracer = render_graph_PathTracer()
try: m.addGraph(GPathTracer)
except NameError: None
