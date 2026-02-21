import argparse
from config import get_default_gpath_tracer_config, GPathTracerConfig, CameraMotionConfig
import os
import common
import falcor
import numpy as np
import pyexr
import tqdm
import random
import time

import concurrent
import gc

from pympler import muppy, summary
from pympler.tracker import SummaryTracker

def render_gt(scene_file, reso: (int, int), max_bounces: int, num_iters: int, output_dir: str):
    testbed = common.create_testbed(reso)
    scene = common.load_scene(testbed, scene_file, aspect_ratio=reso[0] / reso[1])
    pixelLength = 4.0 / 256.0
    camera = scene.camera
    camera_position = camera.position
    camera.position = camera_position + common.falcor.float3(0.004 * 127, 0.0, 0.00)
    camera.target = camera.target + common.falcor.float3(0.004 * 127, 0.0, 0.0)
    render_pass = common.create_gt_passes(testbed, max_bounces)
    # render_pass = common.create_primal_GPathTracer_passes(testbed,
    #                                                       max_bounces,
    #                                                       0,
    #                                                       False)

    img = np.zeros(shape=(reso[1], reso[0], 3))
    for _ in tqdm.tqdm(range(2048)):
        testbed.frame()
        img = img + testbed.render_graph.get_output("GPathTracer.primal").to_numpy()[:,:,:3] / 2048.0

    # dx = np.zeros_like(img)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1] - 1):
    #         dx[i,j,:] = img[i,j+1,:] - img[i,j,:]

    os.makedirs(output_dir, exist_ok=True)
    pyexr.write(f"{output_dir}/gt_dynamic.exr", img[:,:,:3])
    # pyexr.write(f"{output_dir}/dx_gt_dynamic.exr", img[:,:,:3])

def render_ref(scene_file, max_bounces: int, spatial_neighbour_search_radius: float, num_iters: int,
               gt_image_path: str, output_dir: str):
    gt_image = pyexr.read(gt_image_path)
    reso = (gt_image.shape[1], gt_image.shape[0])
    testbed = common.create_testbed(reso)
    scene = common.load_scene(testbed, scene_file, reso[0] / reso[1])
    os.makedirs(output_dir, exist_ok=True)
    for _ in range(1):
        _ = common.create_reference_passes(testbed, max_bounces, spatial_neighbour_search_radius)
        for _ in tqdm.tqdm(range(128)):
            testbed.frame()
        for j in tqdm.tqdm(range(num_iters)):
            testbed.frame()
            cur_img = testbed.render_graph.get_output("PathTracer.color").to_numpy()
            pyexr.write(f"{output_dir}/ref_{j}.exr", cur_img[:,:,:3])

def validate_primal(cfg: GPathTracerConfig):
    gt_image = pyexr.read(cfg.gt_path)
    reso = (gt_image.shape[0], gt_image.shape[1])

    mses = np.zeros(shape=(cfg.num_images))
    img = np.zeros_like(gt_image)
    os.makedirs(cfg.output_dir, exist_ok=True)
    tracker = SummaryTracker()
    testbed = common.create_testbed(reso)
    scene = common.load_scene(testbed, cfg.scene_path, reso[0] / reso[1])
    scene.animated = True
    for i in range(cfg.num_images):
        _ = common.create_GPathTracer_passes(testbed, cfg, random.randint(0, 65536))
        for _ in tqdm.tqdm(range(cfg.num_iters)):
            testbed.frame()
            cur_img = testbed.render_graph.get_output("GPathTracer.color").to_numpy()
            cur_img = cur_img[:,:,:3]

        cur_img = testbed.render_graph.get_output("GPathTracer.color").to_numpy()
        img += cur_img[:,:,:3]

        if i % 8:
            gc.collect()

        mse = np.mean(np.sqrt((img[:,:,:3] / (i + 1) - gt_image[:,:,:3]) ** 2))
        mses[i] = mse
        pyexr.write(f"{cfg.output_dir}/validate_dx_{cfg.method_name}_primal.exr", img[:,:,:3] / (i + 1))
        np.savetxt(f"{cfg.output_dir}/validate_dx_{cfg.method_name}_primal.txt", np.array(mses))

def validate_dx(config: GPathTracerConfig):
    gt_image = pyexr.read(config.gt_path)
    reso = (gt_image.shape[1], gt_image.shape[0])

    print(config.output_dir)
    mses = np.zeros(shape=(config.num_images))
    img = np.zeros_like(gt_image)
    print(img.shape)
    os.makedirs(config.output_dir, exist_ok=True)
    tracker = SummaryTracker()
    testbed = common.create_testbed(reso)
    pixelLength = 4.0 / 256.0
    for i in range(config.num_images):
        _ = common.create_GPathTracer_passes(testbed, config, random.randint(0, 65536))
        scene = common.load_scene(testbed, config.scene_path, reso[0] / reso[1])
        camera = scene.camera
        for _ in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
        for j in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
            camera.position = camera.position + common.falcor.float3(0.3 * pixelLength, 0.0, 0.0)
            camera.target   = camera.target + common.falcor.float3(0.3 * pixelLength, 0.0, 0.0)
            cur_img = testbed.render_graph.get_output("GPathTracer.DX").to_numpy()
            if j == 20:
                img += cur_img[:,:,:3]
        mse = np.mean(np.sqrt((img[:,:,:3] / (i + 1) - gt_image[:,:,:3]) ** 2))
        mses[i] = mse
        # cur_img = testbed.render_graph.get_output("GPathTracer.DX").to_numpy()

        if i % 8:
            gc.collect()

        img_to_save = img[:,:,:3] / (i + 1)
        print(img_to_save.shape)
        pyexr.write(f"{config.output_dir}/validate_dx_{config.method_name}.exr", img_to_save)
        np.savetxt(f"{config.output_dir}/validate_dx_{config.method_name}.txt", np.array(mses))

def render_dx_dynamic(config: GPathTracerConfig, motion_config: CameraMotionConfig):
    gt_image = pyexr.read(config.gt_path)
    reso = (gt_image.shape[1], gt_image.shape[0])

    mses = np.zeros(shape=(config.num_iters))
    img = np.zeros_like(gt_image)
    os.makedirs(config.output_dir, exist_ok=True)
    tracker = SummaryTracker()
    pixelLength = 4.0 / 256.0
    for i in range(config.num_images):
        testbed = common.create_testbed(reso)
        scene = common.load_scene(testbed, config.scene_path, reso[0] / reso[1])
        camera = scene.camera
        _ = common.create_GPathTracer_passes(testbed, config, random.randint(0, 65536))
        for j in tqdm.tqdm(range(128)):
            testbed.frame()
        for j in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
            camera.position = camera.position + common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            camera.target   = camera.target + common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            cur_img = testbed.render_graph.get_output("GPathTracer.DX").to_numpy()
            mse = np.mean(np.sqrt((cur_img[:,:,:3] - gt_image[:,:,:3])**2))
            mses[j] += mse / config.num_images
            pyexr.write(f"{config.output_dir}/render_dx_{config.method_name}_{j}.exr", cur_img[:,:,:3])

        cur_img = testbed.render_graph.get_output("GPathTracer.DX").to_numpy()

        if i % 8:
            gc.collect()

        pyexr.write(f"{config.output_dir}/render_dx_{config.method_name}.exr", cur_img[:,:,:3])
        np.savetxt(f"{config.output_dir}/render_dx_{config.method_name}.txt", np.array(mses))

def render_restir_gt(config: GPathTracerConfig, motion_config: CameraMotionConfig):
    gt_image = pyexr.read(config.gt_path)
    reso = (gt_image.shape[1], gt_image.shape[0])

    mses = np.zeros(shape=(config.num_iters))
    img = np.zeros_like(gt_image)
    print(img.shape)
    os.makedirs(config.output_dir, exist_ok=True)
    tracker = SummaryTracker()
    testbed = common.create_testbed(reso)
    for i in range(config.num_images):
        _ = common.create_gt_passes(testbed, config.max_bounces - 1)
        scene = common.load_scene(testbed, config.scene_path, reso[0] / reso[1])
        camera = scene.camera
        camera_position = camera.position
        for j in tqdm.tqdm(range(128)):
            testbed.frame()
        for j in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
            camera.position = camera.position + common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            camera.target   = camera.target + common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            cur_img = testbed.render_graph.get_output("PathTracer.color").to_numpy()
            mse = np.mean(np.sqrt((cur_img[:,:,:3] - gt_image[:,:,:3])**2))
            mses[j] += mse / config.num_images
            cur_img[np.where(cur_img < 0.0)] = 0.0
            pyexr.write(f"{config.output_dir}/render_primal_{config.method_name}_{j}.exr", cur_img[:,:,:3])


        cur_img = testbed.render_graph.get_output("PathTracer.color").to_numpy()
        cur_img[np.where(cur_img < 0.0)] = 0.0

        if i % 8:
            gc.collect()

        mse = np.mean(np.sqrt((img[:,:,:3] / (i + 1) - gt_image[:,:,:3]) ** 2))
        mses[i] = mse
        pyexr.write(f"{config.output_dir}/render_primal_{config.method_name}.exr", cur_img[:,:,:3])
        np.savetxt(f"{config.output_dir}/render_primal_{config.method_name}.txt", np.array(mses))

def render(config: GPathTracerConfig, motion_config: CameraMotionConfig):
    reso = config.resolution
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "primal"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "dx"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "dy"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "recon"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "albedo"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "depth"), exist_ok=True)

    tracker = SummaryTracker()
    testbed = common.create_testbed(reso)
    # testbed.clock.framerate = 30
    # testbed.clock.play()

    primals = np.zeros(shape=(2 * config.num_iters, reso[1], reso[0], 4))
    dxs = np.zeros(shape=(2 * config.num_iters, reso[1], reso[0], 4))
    dys = np.zeros(shape=(2 * config.num_iters, reso[1], reso[0], 4))
    recons = np.zeros(shape=(2 * config.num_iters, reso[1], reso[0], 4))
    albedos = np.zeros(shape=(2 * config.num_iters, reso[1], reso[0], 4))
    depths = np.zeros(shape=(2 * config.num_iters, reso[1], reso[0], 4))
    normals = np.zeros(shape=(2 * config.num_iters, reso[1], reso[0], 4))

    total_time = 0.0
    for i in range(config.num_images):
        scene = common.load_scene(testbed, config.scene_path, reso[0] / reso[1])
        scene.animated = True
        _ = common.create_GPathTracer_passes(testbed, config, random.randint(0, 65536))
        camera = scene.camera
        testbed.frame()
        for j in tqdm.tqdm(range(64)):
            s = time.perf_counter()
            testbed.frame()
            t = time.perf_counter()
            total_time += (t - s)
        for j in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
            camera.position = camera.position + common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            camera.target   = camera.target + common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            primals[j] += testbed.render_graph.get_output("GPathTracer.primal").to_numpy() / config.num_images
            dxs[j] += testbed.render_graph.get_output("GPathTracer.DX").to_numpy() / config.num_images
            dys[j] += testbed.render_graph.get_output("GPathTracer.DY").to_numpy() / config.num_images
            recons[j] += testbed.render_graph.get_output("GPathTracer.color").to_numpy() / config.num_images
            albedos[j] += testbed.render_graph.get_output("VBufferRT.albedo").to_numpy() / config.num_images
            depths[j] += testbed.render_graph.get_output("VBufferRT.linearDepth").to_numpy() / config.num_images
            normals[j] += testbed.render_graph.get_output("VBufferRT.normW").to_numpy() / config.num_images

        for j in tqdm.tqdm(range(config.num_iters)):
            camera.position = camera.position - common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            camera.target   = camera.target - common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            testbed.frame()
            primals[j + config.num_iters] += testbed.render_graph.get_output("GPathTracer.primal").to_numpy() / config.num_images
            dxs[j + config.num_iters] += testbed.render_graph.get_output("GPathTracer.DX").to_numpy() / config.num_images
            dys[j + config.num_iters] += testbed.render_graph.get_output("GPathTracer.DY").to_numpy() / config.num_images
            recons[j + config.num_iters] += testbed.render_graph.get_output("GPathTracer.color").to_numpy() / config.num_images
            albedos[j + config.num_iters] += testbed.render_graph.get_output("VBufferRT.albedo").to_numpy() / config.num_images
            depths[j + config.num_iters] += testbed.render_graph.get_output("VBufferRT.linearDepth").to_numpy() / config.num_images
            normals[j + config.num_iters] += testbed.render_graph.get_output("VBufferRT.normW").to_numpy() / config.num_images
        if i % 8:
            gc.collect()

    recons[np.where(recons < 0.0)] = 0.0
    for i in tqdm.tqdm(range(2*config.num_iters)):
        pyexr.write(os.path.join(config.output_dir, f"primal/primal_{i}.exr"), primals[i,:,:,:3])
        pyexr.write(os.path.join(config.output_dir, f"dx/dx_{i}.exr"), dxs[i,:,:,:3])
        pyexr.write(os.path.join(config.output_dir, f"dy/dy_{i}.exr"), dys[i,:,:,:3])
        pyexr.write(os.path.join(config.output_dir, f"recon/recon_{i}.exr"), recons[i,:,:,:3])
        pyexr.write(os.path.join(config.output_dir, f"albedo/albedo_{i}.exr"), albedos[i,:,:,:3])
        pyexr.write(os.path.join(config.output_dir, f"depth/depth_{i}.exr"), depths[i,:,:,:3])
        pyexr.write(os.path.join(config.output_dir, f"normal/normal_{i}.exr"), normals[i,:,:,:3])
    print(f"Avg Frame Time: {total_time / 64.0 * 1000.0:.6f}ms.")

def render_primal(config: GPathTracerConfig, motion_config: CameraMotionConfig):
    gt_image = pyexr.read(config.gt_path)
    reso = (gt_image.shape[1], gt_image.shape[0])

    mses = np.zeros(shape=(config.num_iters))
    img = np.zeros_like(gt_image)
    print(img.shape)
    os.makedirs(config.output_dir, exist_ok=True)
    tracker = SummaryTracker()
    pixelLength = 4.0 / 256.0
    testbed = common.create_testbed(reso)
    for i in range(config.num_images):
        scene = common.load_scene(testbed, config.scene_path, reso[0] / reso[1])
        _ = common.create_GPathTracer_passes(testbed, config, random.randint(0, 65536))
        camera = scene.camera
        for j in tqdm.tqdm(range(128)):
            testbed.frame()
        for j in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
            camera.position = camera.position + common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            camera.target   = camera.target + common.falcor.float3(motion_config.vec[0], motion_config.vec[1], motion_config.vec[2])
            cur_img = testbed.render_graph.get_output("GPathTracer.primal").to_numpy()
            mse = np.mean(np.sqrt((cur_img[:,:,:3] - gt_image[:,:,:3])**2))
            mses[j] += mse / config.num_images
            cur_img[np.where(cur_img < 0.0)] = 0.0
            pyexr.write(f"{config.output_dir}/render_primal_{config.method_name}_{j}.exr", cur_img[:,:,:3])


        cur_img = testbed.render_graph.get_output("GPathTracer.primal").to_numpy()
        cur_img[np.where(cur_img < 0.0)] = 0.0

        if i % 8:
            gc.collect()

        mse = np.mean(np.sqrt((img[:,:,:3] / (i + 1) - gt_image[:,:,:3]) ** 2))
        mses[i] = mse
        pyexr.write(f"{config.output_dir}/render_primal_{config.method_name}.exr", cur_img[:,:,:3])
        np.savetxt(f"{config.output_dir}/render_primal_{config.method_name}.txt", np.array(mses))

def render_primal_restir(config: GPathTracerConfig):
    gt_image = pyexr.read(config.gt_path)
    reso = (gt_image.shape[1], gt_image.shape[0])

    mses = np.zeros(shape=(config.num_iters))
    img = np.zeros_like(gt_image)
    print(img.shape)
    os.makedirs(config.output_dir, exist_ok=True)
    tracker = SummaryTracker()
    testbed = common.create_testbed(reso)
    _ = common.load_scene(testbed, config.scene_path, reso[0] / reso[1])
    for i in range(config.num_images):
        _ = common.create_GPathTracer_passes(testbed, config, random.randint(0, 65536))
        for j in tqdm.tqdm(range(128)):
            testbed.frame()
        for j in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
            cur_img = testbed.render_graph.get_output("GPathTracer.primal").to_numpy()
            img += cur_img[:,:,:3] / config.num_iters
            mse = np.mean(np.sqrt((cur_img[:,:,:3] - gt_image[:,:,:3])**2))
            mses[j] += mse / config.num_images

        if i % 8:
            gc.collect()

        mse = np.mean(np.sqrt((img[:,:,:3] / (i + 1) - gt_image[:,:,:3]) ** 2))
        mses[i] = mse
        pyexr.write(f"{config.output_dir}/render_primal_{config.method_name}.exr", img[:,:,:3])
        # np.savetxt(f"{config.output_dir}/render_primal_{config.method_name}.txt", np.array(mses))

def render_primal_gpt(config: GPathTracerConfig):
    gt_image = pyexr.read(config.gt_path)
    reso = (gt_image.shape[1], gt_image.shape[0])

    mses = np.zeros(shape=(config.num_iters))
    img = np.zeros_like(gt_image)
    print(img.shape)
    os.makedirs(config.output_dir, exist_ok=True)
    tracker = SummaryTracker()
    testbed = common.create_testbed(reso)
    _ = common.load_scene(testbed, config.scene_path, reso[0] / reso[1])
    for i in range(config.num_images):
        _ = common.create_GPathTracer_passes(testbed, config, random.randint(0, 65536))
        for j in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
            cur_img = testbed.render_graph.get_output("GPathTracer.color").to_numpy()
            img += cur_img[:,:,:3] / config.num_iters
            mse = np.mean(np.sqrt((cur_img[:,:,:3] - gt_image[:,:,:3])**2))
            mses[j] += mse / config.num_images

        if i % 8:
            gc.collect()

        mse = np.mean(np.sqrt((img[:,:,:3] / (i + 1) - gt_image[:,:,:3]) ** 2))
        mses[i] = mse
        img[np.where(img < 0.0)] = 0.0
        pyexr.write(f"{config.output_dir}/render_primal_{config.method_name}.exr", img[:,:,:3])
        # np.savetxt(f"{config.output_dir}/render_primal_{config.method_name}.txt", np.array(mses))

def render_primal_gpt_dx(config: GPathTracerConfig):
    gt_image = pyexr.read(config.gt_path)
    reso = (gt_image.shape[1], gt_image.shape[0])

    mses = np.zeros(shape=(config.num_iters))
    img = np.zeros_like(gt_image)
    print(img.shape)
    os.makedirs(config.output_dir, exist_ok=True)
    tracker = SummaryTracker()
    testbed = common.create_testbed(reso)
    _ = common.load_scene(testbed, config.scene_path, reso[0] / reso[1])
    for i in range(config.num_images):
        _ = common.create_GPathTracer_passes(testbed, config, random.randint(0, 65536))
        for j in tqdm.tqdm(range(config.num_iters)):
            testbed.frame()
            cur_img = testbed.render_graph.get_output("GPathTracer.DX").to_numpy()
            img += cur_img[:,:,:3] / config.num_iters
            mse = np.mean(np.sqrt((cur_img[:,:,:3] - gt_image[:,:,:3])**2))
            mses[j] += mse / config.num_images

        if i % 8:
            gc.collect()

        mse = np.mean(np.sqrt((img[:,:,:3] / (i + 1) - gt_image[:,:,:3]) ** 2))
        mses[i] = mse
        img[np.where(img < 0.0)] = 0.0
        pyexr.write(f"{config.output_dir}/render_dx_{config.method_name}.exr", img[:,:,:3])
        # np.savetxt(f"{config.output_dir}/render_primal_{config.method_name}.txt", np.array(mses))

def render_with_args(args):
    config_file = args.config_file
    cfg = get_default_gpath_tracer_config()
    cfg.merge_from_file(config_file)
    gPathTracerConfig = GPathTracerConfig(cfg)
    cameraMotionConfig = CameraMotionConfig(cfg)
    if gPathTracerConfig.type == 'dx':
        render_dx_dynamic(gPathTracerConfig, cameraMotionConfig)
    elif gPathTracerConfig.type == 'primal':
        render_primal(gPathTracerConfig, cameraMotionConfig)
    elif gPathTracerConfig.type == 'primal_restir':
        render_primal_restir(gPathTracerConfig)
    elif gPathTracerConfig.type == 'primal_gpt':
        render_primal_gpt(gPathTracerConfig)
    elif gPathTracerConfig.type == 'gpt_dx':
        render_primal_gpt_dx(gPathTracerConfig)
    elif gPathTracerConfig.type == 'restir_ref':
        render_restir_gt(gPathTracerConfig, cameraMotionConfig)
    elif gPathTracerConfig.type == 'final':
        render(gPathTracerConfig, cameraMotionConfig)
    else:
        raise NotImplementedError(f"Unknown rendering type: {gPathTracerConfig.type}")

def validate_with_args(args):
    config_file = args.config_file
    cfg = get_default_gpath_tracer_config()
    cfg.merge_from_file(config_file)
    gPathTracerConfig = GPathTracerConfig(cfg)
    if gPathTracerConfig.type == 'dx':
        validate_dx(gPathTracerConfig)
    elif gPathTracerConfig.type == 'primal':
        validate_primal(gPathTracerConfig)
    else:
        raise NotImplementedError(f"Unknown validation type: {args.type}")

def gt_with_args(args):
    render_gt(os.path.abspath(args.scene), (args.w, args.h),
              args.max_bounces, 16384,
              args.output_dir)

def ref_with_args(args):
    render_ref(os.path.abspath(args.scene), args.max_bounces,
              args.spatial_neighbour_search_radius, 256, args.gt,
              args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gradient-PT rendering experiments')
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    subparsers = parser.add_subparsers()

    parser_render = subparsers.add_parser('render', help='Render images')
    parser_render.add_argument('--config_file', type=str, help='Path to the config file')
    parser_render.set_defaults(func=render_with_args)

    parser_val = subparsers.add_parser('validate', help='Render images')
    parser_val.add_argument('--config_file', type=str, help='Path to the config file')
    parser_val.set_defaults(func=validate_with_args)

    parser_gt = subparsers.add_parser('gt', help='Render ground truth')
    parser_gt.add_argument('--scene', type=str, help='Path to the scene')
    parser_gt.add_argument('--max_bounces', type=int, default=1, help='Max bounces')
    parser_gt.add_argument('--w', type=int, default=512, help='Width')
    parser_gt.add_argument('--h', type=int, default=512, help='Height')
    parser_gt.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser_gt.set_defaults(func=gt_with_args)

    parser_ref = subparsers.add_parser('ref', help='Render ground truth')
    parser_ref.add_argument('--scene', type=str, help='Path to the scene')
    parser_ref.add_argument('--max_bounces', type=int, default=1, help='Max bounces')
    parser_ref.add_argument('--spatial_neighbour_search_radius', type=float, default=32.0, help='Spatial neighbour search radius')
    parser_ref.add_argument('--gt', type=str, default="", help='Path to the ground truth image')
    parser_ref.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser_ref.set_defaults(func=ref_with_args)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
