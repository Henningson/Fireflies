import numpy as np
import math

def build_projection_matrix(fov: float, near_clip: float, far_clip: float) -> np.array:
    S = 1.0 / math.tan((fov / 2.0) * (math.pi / 180))
    z_far = -far_clip / (far_clip - near_clip)
    z_near = - (far_clip * near_clip) / (far_clip - near_clip)

    projection_matrix = np.array([4, 4], dtype=float)
    projection_matrix[0, 0] = S
    projection_matrix[1, 1] = S
    projection_matrix[2, 2] = z_far
    projection_matrix[2, 3] = -1.0
    projection_matrix[3, 2] = z_near

    return projection_matrix