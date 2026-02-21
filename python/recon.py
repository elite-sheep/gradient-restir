# Copyright @yucwang 2025

import pyexr

import os
import sys
screen_poisson_path = os.path.abspath('../../../screen-poisson-py/build/Release/')
sys.path.append(screen_poisson_path)
import screenpoissonpy as sp

import argparse
import numpy as np

def reconstruct_batch(input_dir, num_images, recon_type, output_dir):
    params = sp.PoissonParams()
    params.setConfigPreset(recon_type)
    output_path = os.path.join(input_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    solver = sp.PoissonSolver(params)
    for i in range(num_images):
        primal_path = os.path.join(input_dir, 'primal', f'primal_{i}.exr')
        primal = pyexr.read(primal_path)

        dx_path = os.path.join(input_dir, 'dx', f'dx_{i}.exr')
        dx = pyexr.read(dx_path)

        dy_path = os.path.join(input_dir, 'dy', f'dy_{i}.exr')
        dy = pyexr.read(dy_path)

        solver.load(primal, dx, dy)
        solver.setupBackend()
        solver.solve()
        recon = solver.getFinalImage()
        recon[np.where(recon < 0)] = 0.0

        pyexr.write(os.path.join(output_path, f'recon_{i}.exr'), recon)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, required=True)
    parser.add_argument('--recon_type', type=str, default='default')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    reconstruct_batch(args.input_dir, args.num_images, args.recon_type, args.output_dir)
