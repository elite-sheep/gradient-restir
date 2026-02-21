# Copyright @yucwang 2022

import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyexr
import numpy as np
import os

def read_exr(filename):
    image = pyexr.read(filename)
    return image

def mat_visualization(mat, vmin, vmax, output_name: str):
    img_show = mat
    color_map = LinearSegmentedColormap.from_list("cubicL", np.loadtxt("./CubicL.txt"), N=256)
    f, axes = plt.subplots(1, 1)
    im1 = axes.imshow(img_show, cmap=color_map, interpolation='bilinear', vmin=vmin, vmax=vmax)
    f.colorbar(im1, orientation='vertical')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    f.savefig(output_name, bbox_inches='tight', pad_inches=0.2)

def img_visualization(img, vmin, vmax, output_name: str):
    img_show = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0
    color_map = LinearSegmentedColormap.from_list("cubicL", np.loadtxt("./CubicL.txt"), N=256)
    f, axes = plt.subplots(1, 1)
    im1 = axes.imshow(img_show, cmap=color_map, interpolation='bilinear', vmin=vmin, vmax=vmax)
    f.colorbar(im1, orientation='vertical')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    f.savefig(output_name, bbox_inches='tight', pad_inches=0.2)

def relmse_visualization(source_image, target_image):
    print(source_image.shape)
    source_image_1 = source_image[:,:,0]
    target_image_1 = target_image[:,:,0]
    relmse = (source_image_1 - target_image_1)**2 / (target_image_1**2 + 0.001)
    print("RelMSE: {:3f}".format(relmse.clip(0.0, 8.0).mean()))

    # relmse_show = relmse[:,:,0]
    relmse_show = relmse
    color_map = LinearSegmentedColormap.from_list("cubicL", np.loadtxt("./CubicL.txt"), N=256)
    plt.imshow(relmse_show, cmap=color_map, interpolation='bilinear', vmin=0.0, vmax=2.0)
    plt.axis('off')
    plt.savefig('out.png', bbox_inches='tight', pad_inches = 0)

def batch_img_remap(data_dir):
    n_iters = 128
    color_map = LinearSegmentedColormap.from_list("cubicL", np.loadtxt("./CubicL.txt"), N=256)
    for i in range(n_iters):
        img = read_exr(os.path.join(data_dir, "TR_renderD_{:04d}.exr".format(i)))
        fig, ax = plt.subplots()
        ax.imshow(img[:,:,0], cmap=color_map, interpolation='bilinear', vmin=-1.0, vmax=1.0)
        ax.axis('off')
        fig.savefig(os.path.join(data_dir, "vis/TR_renderD_{}.png".format(i)), bbox_inches='tight', pad_inches=0)
        plt.close()
        print('done: {}'.format(i))

def img_remap(img):
    img = img[:,:,0]
    color_map = LinearSegmentedColormap.from_list("cubicL", np.loadtxt("./CubicL.txt"), N=256)
    img_show = plt.imshow(img / 2.0, cmap=color_map, interpolation='bilinear', vmin=-0.08, vmax=0.08)
    plt.axis('off')
    plt.savefig('out.png', bbox_inches='tight', pad_inches = 0)

def remap(args):
    img = read_exr(args.img)
    file_path_without_ext = os.path.splitext(args.img)[0]
    img_visualization(img, args.vmin, args.vmax, file_path_without_ext + '.png')

def diff(args):
    source_image = read_exr(args.img)
    target_image = read_exr(args.gt)

    rmse = np.zeros((source_image.shape[0], source_image.shape[1]))
    rmse[:,:] = np.sqrt(np.mean((source_image - target_image)**2, axis=2))

    mat_visualization(rmse, 0.0, 0.1, 'out.png')

    file_path_without_ext = os.path.splitext(args.img)[0]
    mat_visualization(rmse, args.vmin, args.vmax, file_path_without_ext + '_diff.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXR visualization')
    parser.add_argument('--img',  type=str, help='Path to the image')
    parser.add_argument('--vmin', type=float, default=-0.1, help='Min for the colormap')
    parser.add_argument('--vmax', type=float, default=0.1, help='Max for the colormap')
    subparsers = parser.add_subparsers()

    # Single EXR image visualization
    parser_remap = subparsers.add_parser('remap', help='Recolor image')
    parser_remap.set_defaults(func=remap)

    # Difference visualization
    parser_diff = subparsers.add_parser('diff', help='Difference visualization')
    parser_diff.add_argument('--gt', help='path to ground truth image')
    parser_diff.set_defaults(func=diff)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
