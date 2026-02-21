import numpy as np

def fovYToFocalLength(fovY):
    return 12.0 / np.tan(0.5 * np.radians(fovY))

focalLength = fovYToFocalLength(19.5)
print(focalLength)
