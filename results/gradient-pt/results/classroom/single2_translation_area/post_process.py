# Copyright @yucwang 2024

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyexr
from PIL import Image
import numpy as np
import os

def read_exr(filename):
    image = pyexr.read(filename)
    return image

def img_remap(img, vmin, vmax, filepath):
    img = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0
    color_map = LinearSegmentedColormap.from_list("cubicL", np.loadtxt("./CubicL.txt"), N=256)
    img_show = plt.imshow(img / 2.0, cmap=color_map, interpolation='bilinear', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight', pad_inches = 0)
    plt.close()

def gamma_correction(image, gamma):
    image = image.clip(0.0, 1.0)
    # Apply gamma correction
    corrected_image = np.power(image, 1.0 / gamma)

    # Scale back to the range [0, 255] and convert to unsigned 8-bit integer
    corrected_image = np.uint8(corrected_image * 255)

    return corrected_image

def save_png(image, filepath):
    image = Image.fromarray(image)
    image.save(filepath)

for i in range(128):
    print(f"frame {i}.")
    primal = read_exr(os.path.join("./primal/", f"primal_{i}.exr"))
    primal = gamma_correction(primal, 2.2)
    save_png(primal, os.path.join("./primal/", f"primal_{i}.png"))

    dx = read_exr(os.path.join("./dx/", f"dx_{i}.exr"))
    dx = img_remap(dx, -0.02, 0.02, os.path.join("./dx/", f"dx_{i}.png"))

    dy = read_exr(os.path.join("./dy/", f"dy_{i}.exr"))
    dy = img_remap(dy, -0.02, 0.02, os.path.join("./dy/", f"dy_{i}.png"))

    recon = read_exr(os.path.join("./recon/", f"recon_{i}.exr"))
    recon = gamma_correction(recon, 2.2)
    save_png(recon, os.path.join("./recon/", f"recon_{i}.png"))
