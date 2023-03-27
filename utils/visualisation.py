import random
import numpy as np

def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b]).astype(np.float32)

def remove_ceiling(points):
    points_mask = points[..., 2] < np.max(points[..., 2]) - 1
    points = points[points_mask]
    return points
    