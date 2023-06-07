import numpy as np

def rayPlane(laserOrigin, laserDirection, planeOrigin, planeNormal):
    denom = np.sum(planeNormal * laserDirection, axis=1)

    denom = np.where(np.abs(denom) < 0.000001, denom/denom, denom)
    t = np.sum((planeOrigin - laserOrigin) * planeNormal, axis=1) / denom

    return t[:, None]