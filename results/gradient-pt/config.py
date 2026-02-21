# Copyright @yucwang 2024

from yacs.config import CfgNode as CN

class CameraMotionConfig:
    def __init__(self, cfg: CN):
        self.type = cfg.motion_type
        self.vec  = cfg.motion_vec

class GPathTracerConfig:
    def __init__(self,
                 type: str,
                 scene_path: str,
                 gt_path: str,
                 max_bounces: int,
                 diff_integrator: int,
                 shift_mapping_type: int,
                 should_reject_inner_shift: bool,
                 should_reject_outer_shift: bool,
                 use_temporal: bool,
                 use_spatial: bool,
                 num_spatial_neighbours: int,
                 spatial_neighbour_search_radius: float,
                 jacobian_threshold: float,
                 num_iters: int,
                 num_images: int,
                 output_dir: str,
                 method_name: str = 'multirestir',
                 verbose: bool = False):
        self.type = type
        self.scene_path = scene_path
        self.gt_path = gt_path
        self.max_bounces = max_bounces
        self.diff_integrator = diff_integrator
        self.shift_mapping_type = shift_mapping_type
        self.should_reject_inner_shift = should_reject_inner_shift
        self.should_reject_outer_shift = should_reject_outer_shift
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.num_spatial_neighbours = num_spatial_neighbours
        self.spatial_neighbour_search_radius = spatial_neighbour_search_radius
        self.jacobian_threshold = jacobian_threshold
        self.num_iters = num_iters
        self.num_images = num_images
        self.output_dir = output_dir
        self.method_name = method_name
        self.verbose = verbose

    def __init__(self, cfg: CN):
        self.type = cfg.type
        self.scene_path = cfg.scene_path
        self.gt_path = cfg.gt_path
        self.max_bounces = cfg.max_bounces
        self.diff_integrator = cfg.diff_integrator
        self.shift_mapping_type = cfg.shift_mapping_type
        self.should_reject_inner_shift = cfg.should_reject_inner_shift
        self.should_reject_outer_shift = cfg.should_reject_outer_shift
        self.use_temporal = cfg.use_temporal
        self.use_spatial = cfg.use_spatial
        self.num_spatial_neighbours = cfg.num_spatial_neighbours
        self.num_initial_samples = cfg.num_initial_samples
        self.spatial_neighbour_search_radius = cfg.spatial_neighbour_search_radius
        self.jacobian_threshold = cfg.jacobian_threshold
        self.num_iters = cfg.num_iters
        self.num_images = cfg.num_images
        self.output_dir = cfg.output_dir
        self.method_name = cfg.method_name
        self.resolution = cfg.resolution
        self.verbose = cfg.verbose

def get_default_gpath_tracer_config():
    cfg = CN()
    cfg.type = 'dx'
    cfg.scene_path = ''
    cfg.scene_file = ''
    cfg.gt_path = ''
    cfg.max_bounces = 1
    cfg.diff_integrator = 1
    cfg.shift_mapping_type = 0
    cfg.should_reject_inner_shift = True
    cfg.should_reject_outer_shift = True
    cfg.use_temporal = True
    cfg.use_spatial = True
    cfg.num_initial_samples = 2
    cfg.num_spatial_neighbours = 4
    cfg.spatial_neighbour_search_radius = 4.0
    cfg.jacobian_threshold = 0.001
    cfg.num_iters = 1
    cfg.num_images = 1
    cfg.output_dir = 'output'
    cfg.method_name = 'multirestir'
    cfg.motion_type = 'none'
    cfg.motion_vec = [0.0, 0.0, 0.0]
    cfg.resolution = [ 360, 540 ]
    cfg.verbose = False

    return cfg
