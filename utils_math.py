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