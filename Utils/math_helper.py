import random
import torch
import torch.nn.functional as F
import math


def uniformBetweenValues(a: float, b: float):
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


def test():
    import Utils.transforms as transforms

    roll = 0.0
    pitch = 3.141 / 2.0
    yaw = 0.0

    vec = torch.tensor([[1.0, 0.0, 0.0]])

    yaw_mat = getYawTransform(yaw, vec.device)
    pitch_mat = getPitchTransform(pitch, vec.device)
    roll_mat = getRollTransform(roll, vec.device)

    rot = yaw_mat @ pitch_mat @ roll_mat

    rot = transforms.toMat4x4(rot)
    rotVec = transforms.transform_points(vec, rot)
    print(rotVec)


if __name__ == "__main__":
    # Example usage
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0])

    rotation_matrix = rotation_matrix_from_vectors(v1, v2)
    print("Rotation Matrix:")
    print(rotation_matrix @ v1.T)
