import torch

def rayPlane(laserOrigin, laserDirection, planeOrigin, planeNormal):
    denom = torch.sum(planeNormal * laserDirection, axis=1)

    denom = torch.where(torch.abs(denom) < 0.000001, denom/denom, denom)
    t = torch.sum((planeOrigin - laserOrigin) * planeNormal, axis=1) / denom

    return t[:, None]