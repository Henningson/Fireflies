import torch
import Utils.utils as utils
import Utils.math as math
import numpy as np

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

    points_0_h = torch.matmul(
          transform.unsqueeze(0), 
          points_1_h.unsqueeze(-1))
    
    points_0_h = points_0_h.squeeze(dim=-1)

    points_0 = convert_points_from_homogeneous(points_0_h)
    return points_0


def transform_directions(points: torch.tensor, transform: torch.tensor) -> torch.tensor:
    points = convert_points_to_nonhomogeneous(points)
    points = (transform.unsqueeze(0) @ points.unsqueeze(-1))
    points = torch.squeeze(points, axis=-1)
    return points[..., :-1]


def matToBlender(mat, device):
    
    coordinate_shift = torch.eye(4, device=device)
    coordinate_shift[1, 1] = 0.0
    coordinate_shift[2, 2] = 0.0
    coordinate_shift[2, 1] = -1.0
    coordinate_shift[1, 2] = 1.0

    return coordinate_shift.inverse() @ mat #@ init_rot.inverse()


def matToMitsuba(mat):
    #init_rot = torch.zeros(4, device=mat.device)
    #init_rot[0, 0] = -1.0
    #init_rot[1, 2] = -1.0
    
    rotmat = toMat4x4(math.getPitchTransform(np.pi * 0.5, mat.device))

    #coordinate_shift = torch.eye(4, device=mat.device)
    #coordinate_shift[1, 1] = 0.0
    #coordinate_shift[2, 2] = 0.0
    #coordinate_shift[2, 1] = 1.0
    #coordinate_shift[1, 2] = 1.0

    return mat
    #return coordinate_shift @ mat @ coordinate_shift @ init_rot 
    #return mat @ init_rot


def project_to_camera_space(params, points) -> torch.tensor:
    x_fov = params['PerspectiveCamera.x_fov']
    near_clip = params['PerspectiveCamera.near_clip']
    far_clip = params['PerspectiveCamera.far_clip']

    # TODO: Refactor
    perspective = utils.build_projection_matrix(x_fov, near_clip, far_clip).to('cuda')
    
    camera_to_world = params["PerspectiveCamera.to_world"].matrix.torch()

    view_space_points = transform_points(points, camera_to_world.inverse())
    ndc_points = transform_points(view_space_points, perspective)
    return ndc_points