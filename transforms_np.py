import numpy as np


def convert_points_to_homogeneous(points: np.array) -> np.array:
    return np.pad(points, pad_width=((0,0), (0, 1)), mode="constant", constant_values=(0.0, 1.0))


def convert_points_from_homogeneous(points: np.array) -> np.array:
    return points[..., :-1] / points[..., -1:]


def convert_points_to_nonhomogeneous(points: np.array) -> np.array:
    return np.pad(points, pad_width=((0,0), (0, 1)), mode="constant", constant_values=(0.0, 0.0))


def transform_points(points: np.array, transform: np.array) -> np.array:
    points_1_h = convert_points_to_homogeneous(points)

    points_0_h = np.matmul(
          np.expand_dims(transform, 0), 
          np.expand_dims(points_1_h, -1))
    
    points_0_h = np.squeeze(points_0_h, axis=-1)

    points_0 = convert_points_from_homogeneous(points_0_h)
    return points_0


def transform_directions(points: np.array, transform: np.array) -> np.array:
    points = convert_points_to_nonhomogeneous(points)
    points = (np.expand_dims(transform, 0) @ np.expand_dims(points, -1))
    points = np.squeeze(points, axis=-1)
    return points[..., :-1]