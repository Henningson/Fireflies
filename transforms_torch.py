import torch

def convert_points_to_homogeneous(points: torch.tensor) -> torch.tensor:
    return torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=1.0)

def toMat4x4(mat: torch.tensor, addOne: bool=True) -> torch.tensor:
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