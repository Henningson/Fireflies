import random
import torch
import math

def uniformBetweenValues(a: float, b: float):
    return random.uniform(a, b)

def getYawTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    rotZ = torch.tensor([[math.cos(alpha), -math.sin(alpha), 0],
                         [math.sin(alpha),  math.cos(alpha), 0],
                         [         0,                     0, 1]],
                         device=_device)

    return rotZ

def getPitchTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    rotY = torch.tensor([[math.cos(alpha),  0, math.sin(alpha)],
                         [              0,  1,               0],
                         [-math.sin(alpha), 0, math.cos(alpha)]],
                         device=_device)

    return rotY

def getRollTransform(alpha: float, _device: torch.cuda.device) -> torch.tensor:
    rotX = torch.tensor([[1,               0,                0],
                         [0, math.cos(alpha), -math.sin(alpha)],
                         [0, math.sin(alpha),  math.cos(alpha)]],
                         device=_device)

    return rotX




def test():
    import transforms_torch

    roll = 0.0
    pitch = 3.141/2.0
    yaw = 0.0


    vec = torch.tensor([[1.0, 0.0, 0.0]])

    yaw_mat   = getYawTransform(yaw, vec.device)
    pitch_mat = getPitchTransform(pitch, vec.device)
    roll_mat  = getRollTransform(roll, vec.device)

    rot = yaw_mat @ pitch_mat @ roll_mat

    rot = transforms_torch.toMat4x4(rot)
    rotVec = transforms_torch.transform_points(vec, rot)
    print(rotVec)


if __name__ == "__main__":
    test()