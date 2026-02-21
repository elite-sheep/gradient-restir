import pyexr
import imageio
import numpy as np
import os

os.makedirs('./png', exist_ok=True)
for i in range(128):
    image_i = pyexr.open('./render_primal_multirestir_{}.exr'.format(i))
    image_np = np.zeros(shape=(image_i.height, image_i.width, 3), dtype=np.float32)
    image_np[:,:,0] = image_i.get('R').squeeze()
    image_np[:,:,1] = image_i.get('G').squeeze()
    image_np[:,:,2] = image_i.get('B').squeeze()
    image_np = image_np.clip(0.0, 1.0)
    image_np = (image_np * 255.0).astype(np.uint8)
    imageio.imwrite('./png/render_primal_multirestir_{:04d}.png'.format(i), image_np)
