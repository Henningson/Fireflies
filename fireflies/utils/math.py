import random
import numpy as np
import torch
import torch.nn.functional as F
import math


def uniformBetweenValues(a: float, b: float) -> float:
    return random.uniform(a, b)


def getZTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    return getYawTransform(alpha, _device)


def getYTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    return getPitchTransform(alpha, _device)


def getXTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    return getRollTransform(alpha, _device)


def getYawTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    rotZ = torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha), math.cos(alpha), 0],
            [0, 0, 1],
        ],
        device=_device,
    )

    return rotZ


def getPitchTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    rotY = torch.tensor(
        [
            [math.cos(alpha), 0, math.sin(alpha)],
            [0, 1, 0],
            [-math.sin(alpha), 0, math.cos(alpha)],
        ],
        device=_device,
    )

    return rotY


def getRollTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    rotX = torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(alpha), -math.sin(alpha)],
            [0, math.sin(alpha), math.cos(alpha)],
        ],
        device=_device,
    )

    return rotX


def vector_dot(A: torch.tensor, B: torch.tensor) -> torch.tensor:
    return torch.sum(A * B, dim=-1)


def rotation_matrix_from_vectors(v1, v2):
    """
    Calculates the rotation matrix that transforms vector v1 to v2.

    Args:
    - v1 (torch.Tensor): The source 3D vector (3x1).
    - v2 (torch.Tensor): The target 3D vector (3x1).

    Returns:
    - torch.Tensor: The 3x3 rotation matrix.
    """
    v1 = F.normalize(v1, dim=0)
    v2 = F.normalize(v2, dim=0)

    # Compute the cross product and dot product
    cross_product = torch.cross(v1, v2)
    dot_product = torch.dot(v1, v2)

    # Skew-symmetric matrix for cross product
    skew_sym_matrix = torch.tensor(
        [
            [0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0],
        ],
        dtype=torch.float32,
        device=v1.device,
    )

    # Rotation matrix using Rodrigues' formula
    rotation_matrix = (
        torch.eye(3, device=v1.device)
        + skew_sym_matrix
        + torch.mm(skew_sym_matrix, skew_sym_matrix)
        * (1 - dot_product)
        / torch.norm(cross_product) ** 2
    )

    return rotation_matrix


def rotation_matrix_from_vectors_with_fixed_up(
    v1, v2, up_vector=torch.tensor([0.0, 0.0, 1.0])
):
    """
    Calculates the rotation matrix that transforms vector v1 to v2, while keeping an "up" direction fixed.

    Args:
    - v1 (torch.Tensor): The source 3D vector (3x1).
    - v2 (torch.Tensor): The target 3D vector (3x1).
    - up_vector (torch.Tensor): The fixed "up" direction (3x1). Default is [0, 0, 1].

    Returns:
    - torch.Tensor: The 3x3 rotation matrix.
    """
    v1 = F.normalize(v1, dim=0)
    v2 = F.normalize(v2, dim=0)
    up_vector = F.normalize(up_vector, dim=0)

    # Compute the cross product and dot product
    cross_product = torch.cross(v1, v2)
    dot_product = torch.dot(v1, v2)

    # Skew-symmetric matrix for cross product
    skew_sym_matrix = torch.tensor(
        [
            [0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0],
        ],
        dtype=torch.float32,
        device=v1.device,
    )

    # Rotation matrix using Rodrigues' formula
    rotation_matrix = (
        torch.eye(3, device=v1.device)
        + skew_sym_matrix
        + torch.mm(skew_sym_matrix, skew_sym_matrix)
        * (1 - dot_product)
        / torch.norm(cross_product) ** 2
    )

    # Ensure the "up" direction is fixed
    rotated_up_vector = torch.mv(rotation_matrix, up_vector)
    correction_axis = torch.cross(rotated_up_vector, up_vector)
    correction_angle = torch.acos(torch.dot(rotated_up_vector, up_vector))

    # Apply the correction to the rotation matrix
    correction_matrix = F.normalize(skew_sym_matrix, dim=0) * correction_angle
    corrected_rotation_matrix = torch.eye(3, device=v1.device) + correction_matrix

    return corrected_rotation_matrix


def singleRandomBetweenTensors(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    assert a.size() == b.size()
    assert a.device == b.device

    rands = random.uniform(0, 1)
    return rands * (b - a) + b


def randomBetweenTensors(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    assert a.size() == b.size()
    assert a.device == b.device

    rands = torch.rand(a.shape, device=a.device)
    return rands * (b - a) + a


def normalize(tensor: torch.tensor) -> torch.tensor:
    tensor = tensor - tensor.amin()
    tensor = tensor / tensor.amax()
    return tensor


def normalize_channelwise(
    tensor: torch.tensor,
    dim: int = -1,
    device: torch.cuda.device = torch.device("cuda"),
) -> torch.tensor:
    indices = torch.arange(0, len(tensor.shape), device=device)
    mask = torch.ones(indices.shape, dtype=torch.bool, device=device)
    mask[dim] = False
    indices = indices[mask].tolist()

    tensor = tensor - tensor.amin(indices)
    tensor = tensor / tensor.amax(indices)
    return tensor


def convert_points_to_homogeneous(points: torch.tensor) -> torch.tensor:
    return torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=1.0)


def toMat4x4(mat: torch.tensor, addOne: bool = True) -> torch.tensor:
    mat4x4 = torch.nn.functional.pad(mat, pad=(0, 1, 0, 1), mode="constant", value=0.0)

    if addOne:
        mat4x4[3, 3] = 1.0

    return mat4x4


def convert_points_from_homogeneous(points: torch.tensor) -> torch.tensor:
    return points[..., :-1] / points[..., -1:]


def convert_points_to_nonhomogeneous(points: torch.tensor) -> torch.tensor:
    return torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=0.0)


def transform_points(points: torch.tensor, transform: torch.tensor) -> torch.tensor:
    points_1_h = convert_points_to_homogeneous(points)

    points_0_h = torch.matmul(transform.unsqueeze(0), points_1_h.unsqueeze(-1))

    points_0_h = points_0_h.squeeze(dim=-1)

    points_0 = convert_points_from_homogeneous(points_0_h)
    return points_0


def transform_directions(points: torch.tensor, transform: torch.tensor) -> torch.tensor:
    points = convert_points_to_nonhomogeneous(points)
    points = transform.unsqueeze(0) @ points.unsqueeze(-1)
    points = torch.squeeze(points, axis=-1)
    return points[..., :-1]
