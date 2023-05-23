import torch
import math
from typing import List


def normalize(tensor: torch.tensor) -> torch.tensor:
    tensor = tensor - tensor.amin()
    tensor = tensor / tensor.amax()
    return tensor


def normalize_channelwise(tensor: torch.tensor, dim: int = -1, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    indices = torch.arange(0, len(tensor.shape), device=device)
    mask = torch.ones(indices.shape, dtype=torch.bool, device=device)
    mask[dim] = False
    indices = indices[mask].tolist()

    tensor = tensor - tensor.amin(indices)
    tensor = tensor / tensor.amax(indices)
    return tensor


def retain_grads(non_leaf_tensor: List[torch.tensor]) -> None:
    for tensor in non_leaf_tensor:
        tensor.retain_grad()


# From: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html#FoVPerspectiveCameras.compute_projection_matrix

def build_projection_matrix(fov: float, near_clip: float, far_clip: float, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    """
    Compute the calibration matrix K of shape (N, 4, 4)

    Args:
        znear: near clipping plane of the view frustrum.
        zfar: far clipping plane of the view frustrum.
        fov: field of view angle of the camera.
        aspect_ratio: aspect ratio of the image pixels.
            1.0 indicates square pixels.
        degrees: bool, set to True if fov is specified in degrees.

    Returns:
    torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
    """
    K = torch.zeros((4, 4), dtype=torch.float32, device=device)
    fov = (math.pi / 180) * fov

    if not torch.is_tensor(fov):
        fov = torch.tensor(fov, device=device)

    tanHalfFov = torch.tan((fov / 2))
    max_y = tanHalfFov * near_clip
    min_y = -max_y
    max_x = max_y * 1.0
    min_x = -max_x

    # NOTE: In OpenGL the projection matrix changes the handedness of the
    # coordinate frame. i.e the NDC space positive z direction is the
    # camera space negative z direction. This is because the sign of the z
    # in the projection matrix is set to -1.0.
    # In pytorch3d we maintain a right handed coordinate system throughout
    # so the so the z sign is 1.0.
    z_sign = 1.0

    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    K[0, 0] = 2.0 * near_clip / (max_x - min_x)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    K[1, 1] = 2.0 * near_clip / (max_y - min_y)
    K[0, 2] = (max_x + min_x) / (max_x - min_x)
    K[1, 2] = (max_y + min_y) / (max_y - min_y)
    K[3, 2] = z_sign

    # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
    # is at the near clipping plane and z = 1 when the point is at the far
    # clipping plane.
    K[2, 2] = z_sign * far_clip / (far_clip - near_clip)
    K[2, 3] = -(far_clip * near_clip) / (far_clip - near_clip)

    return K