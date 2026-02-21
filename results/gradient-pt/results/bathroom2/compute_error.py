import pyexr
import numpy as np

gt = pyexr.read("./hybrid_gt_still/gt_primal.exr")
multi = pyexr.read("./single2_still/recon/recon.exr")

mse = ((gt - multi)** 2) / (gt + 0.1)
mse = np.clip(mse, 0.0, 10.0)
print(f"MSE: {np.mean(mse)}")
