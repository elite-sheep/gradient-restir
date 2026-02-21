from config import GPathTracerConfig
from pathlib import Path
import torch
import falcor
import numpy as np

def load_scene(testbed: falcor.Testbed, scene_path: Path, aspect_ratio=1.0):
    flags = (
        falcor.SceneBuilderFlags.DontMergeMaterials
        | falcor.SceneBuilderFlags.RTDontMergeDynamic
        | falcor.SceneBuilderFlags.DontOptimizeMaterials
    )
    testbed.load_scene(scene_path, flags)
    testbed.scene.camera.aspectRatio = aspect_ratio
    testbed.scene.renderSettings.useAnalyticLights = False
    testbed.scene.renderSettings.useEnvLight = False
    return testbed.scene

def create_testbed(reso: (int, int)):
    device_id = 1
    testbed = falcor.Testbed(
        width=reso[0], height=reso[1], create_window=False, gpu=device_id
    )
    testbed.clock.time = 0
    testbed.clock.pause()
    return testbed

def create_gt_passes(testbed: falcor.Testbed, max_bounces: int):
    # Rendering graph of the WAR differentiable path tracer.
    render_graph = testbed.create_render_graph("PathTracer")
    gVBufferParams = {
        'samplePattern': "Center",
        'sampleCount': 1,
        'useAlphaTest': True,
        'subPixelRandom' : "UnitQuad",
        'useDOF' : True
    }
    VBufferRT = render_graph.createPass(
        "VBufferRT",
        "VBufferRT",
        gVBufferParams)
    primal_accumulate_pass = render_graph.create_pass(
        "AccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    pt_pass = render_graph.createPass(
                    "GPathTracer",
                    "GPathTracer",
                    {'samplesPerPixel': 1,
                     'diffIntegrator': 1,
                     'maxBounces': max_bounces,
                     'useTemporalReuse': False,
                     'useSpatialReuse':  False,
                     'numInitialSamples':1,
                     'shiftMappingType': 2})
    # pt_pass = render_graph.create_pass(
    #     "PathTracer",
    #     "PathTracer",
    #     {
    #         "samplesPerPixel": 1,
    #         "maxSurfaceBounces": max_bounces,
    #     },
    # )
    render_graph.add_edge("VBufferRT.vbuffer", "GPathTracer.vbuffer")
    render_graph.add_edge("VBufferRT.vbufferCenter", "GPathTracer.vbufferCenter")
    render_graph.addEdge("VBufferRT.viewW", "GPathTracer.viewW")
    render_graph.addEdge("VBufferRT.mvec", "GPathTracer.mvec")
    render_graph.addEdge("VBufferRT.depth", "GPathTracer.depth")
    render_graph.addEdge("VBufferRT.subPixelUV", "GPathTracer.subPixelUV")
    render_graph.add_edge("GPathTracer.color", "AccumulatePass.input")
    render_graph.mark_output("AccumulatePass.output")
    render_graph.mark_output("GPathTracer.primal")
    render_graph.mark_output("GPathTracer.DX")
    render_graph.mark_output("GPathTracer.DY")
    render_graph.mark_output("VBufferRT.albedo")
    render_graph.mark_output("VBufferRT.normW")
    render_graph.mark_output("VBufferRT.linearDepth")

    passes = {
        "primal_accumulate": primal_accumulate_pass,
        "pt": pt_pass,
    }

    testbed.render_graph = render_graph
    return passes

def create_reference_passes(testbed: falcor.Testbed, max_bounces: int,
                            spatial_neighbour_search_radius: float):
    # Rendering graph of the WAR differentiable path tracer.
    render_graph = testbed.create_render_graph("PathTracer")
    gVBufferParams = {
        'samplePattern': "Center",
        'sampleCount': 1,
        'useAlphaTest': True,
        'subPixelRandom' : "UnitQuad",
        'useDOF' : True
    }
    VBufferRT = render_graph.createPass(
        "VBufferRT",
        "VBufferRT",
        gVBufferParams)
    primal_accumulate_pass = render_graph.create_pass(
        "AccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    pt_pass = render_graph.create_pass(
        "PathTracer",
        "PathTracer",
        {
            "samplesPerPixel": 1,
            "maxSurfaceBounces": max_bounces,
            "ReSTIRPTOptions": {
                "spatialGatherRadius": spatial_neighbour_search_radius
            }
        },
    )
    render_graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    render_graph.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    render_graph.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    render_graph.mark_output("PathTracer.color")

    passes = {
        "primal_accumulate": primal_accumulate_pass,
        "pt": pt_pass,
    }

    testbed.render_graph = render_graph
    return passes

def create_primal_GPathTracer_passes(testbed: falcor.Testbed,
                                     max_bounces: int,
                                     diff_integrator: int,
                                     use_temporal: bool,
                                     use_spatial: bool,
                                     num_spatial_neighbours: int,
                                     meta_seed: int,
                                     verbose: bool = False):
    if verbose:
        print(f"""Creating primal GPathTracer pass: max_bounces={max_bounces},
              diff_integrator={diff_integrator},
              use_temporal={use_temporal},
              use_spatial={use_spatial},
              num_spatial_neighbours={num_spatial_neighbours},
              meta_seed={meta_seed}.""")
    render_graph = testbed.create_render_graph("GPathTracer")
    GPathTracer = render_graph.createPass(
                    "GPathTracer",
                    "GPathTracer",
                    {'samplesPerPixel': 1,
                     'diffIntegrator': diff_integrator,
                     'maxBounces': max_bounces,
                     'useTemporalReuse': use_temporal,
                     'useSpatialReuse': use_spatial,
                     'numSpatialNeighbours': num_spatial_neighbours,
                     'metaSeed': meta_seed})
    primal_accumulate_pass = render_graph.create_pass(
        "AccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    render_graph.add_edge("GPathTracer.color", "AccumulatePass.input")
    render_graph.mark_output("AccumulatePass.output")
    render_graph.mark_output("GPathTracer.color")

    passes = {
        "primal_accumulate": primal_accumulate_pass,
        "pt": GPathTracer,
    }

    testbed.render_graph = render_graph
    return passes

def create_GPathTracer_passes(testbed: falcor.Testbed,
                              config: GPathTracerConfig,
                              meta_seed: int = 1234):
    if config.verbose:
        print(f"""Creating DX pass: max_bounces={config.max_bounces},
              diff_integrator={config.diff_integrator},
              use_temporal={config.use_temporal},
              use_spatial={config.use_spatial},
              num_spatial_neighbours={config.num_spatial_neighbours},
              num_initial_samples={config.num_initial_samples},
              jacobian_threshold={config.jacobian_threshold},
              shift_mapping_type={config.shift_mapping_type},
              should_reject_inner_shift={config.should_reject_inner_shift},
              should_reject_outer_shift={config.should_reject_outer_shift},
              meta_seed={meta_seed}.""")
    render_graph = testbed.create_render_graph("GPathTracer")
    gVBufferParams = {
        'samplePattern': "Center",
        'sampleCount': 1,
        'useAlphaTest': True,
        'subPixelRandom' : "UnitQuad",
        'useDOF' : True
    }
    VBufferRT = render_graph.createPass(
        "VBufferRT",
        "VBufferRT",
        gVBufferParams)
    GPathTracer = render_graph.createPass(
                    "GPathTracer",
                    "GPathTracer",
                    {'samplesPerPixel': 1,
                     'diffIntegrator': config.diff_integrator,
                     'maxBounces': config.max_bounces,
                     'useTemporalReuse': config.use_temporal,
                     'useSpatialReuse': config.use_spatial,
                     'numInitialSamples': config.num_initial_samples,
                     'numSpatialNeighbours': config.num_spatial_neighbours,
                     'spatialNeighboursSearchRadius': config.spatial_neighbour_search_radius,
                     'jacobianThreshold': config.jacobian_threshold,
                     'shiftMappingType': config.shift_mapping_type,
                     'shouldRejectInnerShift': config.should_reject_inner_shift,
                     'shouldRejectOuterShift': config.should_reject_outer_shift,
                     'metaSeed': meta_seed})
    DX_accumulate_pass = render_graph.create_pass(
        "AccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    render_graph.add_edge("VBufferRT.vbuffer", "GPathTracer.vbuffer")
    render_graph.add_edge("VBufferRT.vbufferCenter", "GPathTracer.vbufferCenter")
    render_graph.add_edge("VBufferRT.viewW", "GPathTracer.viewW")
    render_graph.addEdge("VBufferRT.depth", "GPathTracer.depth")
    render_graph.add_edge("VBufferRT.mvec", "GPathTracer.mvec")
    render_graph.addEdge("VBufferRT.subPixelUV", "GPathTracer.subPixelUV")
    render_graph.add_edge("GPathTracer.color", "AccumulatePass.input")
    render_graph.mark_output("AccumulatePass.output")
    render_graph.mark_output("GPathTracer.DX")
    render_graph.mark_output("GPathTracer.DY")
    render_graph.mark_output("GPathTracer.color")
    render_graph.mark_output("GPathTracer.primal")
    render_graph.mark_output("VBufferRT.albedo")
    render_graph.mark_output("VBufferRT.normW")
    render_graph.mark_output("VBufferRT.linearDepth")

    passes = {
        "DX_accumulate": DX_accumulate_pass,
        "pt": GPathTracer,
    }

    testbed.render_graph = render_graph
    return passes

def create_DX_GPathTracer_passes(testbed: falcor.Testbed,
                                 max_bounces: int,
                                 diff_integrator: int,
                                 use_temporal: bool,
                                 use_spatial: bool,
                                 num_spatial_neighbours: int,
                                 meta_seed: int,
                                 verbose: bool = False):
    if verbose:
        print(f"""Creating DX pass: max_bounces={max_bounces},
              diff_integrator={diff_integrator},
              use_temporal={use_temporal},
              use_spatial={use_spatial},
              num_spatial_neighbours={num_spatial_neighbours},
              meta_seed={meta_seed}.""")
    render_graph = testbed.create_render_graph("GPathTracer")
    GPathTracer = render_graph.createPass(
                    "GPathTracer",
                    "GPathTracer",
                    {'samplesPerPixel': 1,
                     'diffIntegrator': diff_integrator,
                     'maxBounces': max_bounces,
                     'useTemporalReuse': use_temporal,
                     'useSpatialReuse': use_spatial,
                     'numSpatialNeighbours': num_spatial_neighbours,
                     'metaSeed': meta_seed})
    DX_accumulate_pass = render_graph.create_pass(
        "AccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    render_graph.add_edge("GPathTracer.DX", "AccumulatePass.input")
    render_graph.mark_output("AccumulatePass.output")
    render_graph.mark_output("GPathTracer.DX")

    passes = {
        "DX_accumulate": DX_accumulate_pass,
        "pt": GPathTracer,
    }

    testbed.render_graph = render_graph
    return passes

def create_wrap_passes(testbed: falcor.Testbed, max_bounces: int, use_war: bool):
    # Rendering graph of the WAR differentiable path tracer.
    render_graph = testbed.create_render_graph("WARDiffPathTracer")
    primal_accumulate_pass = render_graph.create_pass(
        "PrimalAccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    grad_accumulate_pass = render_graph.create_pass(
        "GradAccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    war_diff_pt_pass = render_graph.create_pass(
        "WARDiffPathTracer",
        "WARDiffPathTracer",
        {
            "samplesPerPixel": 1,
            "maxBounces": max_bounces,
            "diffMode": "BackwardDiff",
            "useWAR": use_war,
        },
    )
    render_graph.add_edge("WARDiffPathTracer.color", "PrimalAccumulatePass.input")
    render_graph.add_edge("WARDiffPathTracer.dColor", "GradAccumulatePass.input")
    render_graph.mark_output("PrimalAccumulatePass.output")
    render_graph.mark_output("GradAccumulatePass.output")

    passes = {
        "primal_accumulate": primal_accumulate_pass,
        "grad_accumulate": grad_accumulate_pass,
        "war_diff_pt": war_diff_pt_pass,
    }

    testbed.render_graph = render_graph
    return passes
