import torch


# Batchwise ray-plane intersection
def rayPlane(laserOrigin, laserDirection, planeOrigin, planeNormal):
    denom = torch.sum(planeNormal * laserDirection, axis=1)

    denom = torch.where(torch.abs(denom) < 0.000001, denom/denom, denom)
    t = torch.sum((planeOrigin - laserOrigin) * planeNormal, axis=1) / denom

    return t[:, None]



# Batchwise Sphere-Sphere intersection:
# Inputs: 
#   - a_coords: Tensor of size NxD where N is the number of circles, and D the number of dimensions
#   - a_radii: radii of the spheres in a_xy
#   - b_coords: Tensor of size NxD where N is the number of circles, and D the number of dimensions
#   - b_radii: radii of the spheres in a_xy
# Outputs:
#   - Tensor of size Nx1, with boolean values being 
#       - TRUE when a hit occured
#       - FALSE otherwise

def sphereSphere(a_coords, a_radius, b_coords, b_radius):
    dist_ab = a_coords - b_coords
    squared_dist = dist_ab.pow(2).sum(dim=1, keepdim=True)
    
    sum_radii = a_radius + b_radius
    squared_radii = sum_radii.pow(2)

    return squared_dist <= squared_radii




if __name__ == "__main__":
    import numpy as np
    import cv2


    # Here we create random circles and intersect them
    # They're rendered green, if they do not intersect
    # And red, if they intersect
    for _ in range(1000):
        a_coords = torch.rand(1, 2)
        a_radius = torch.rand(1, 1) / 2.0
        b_coords = torch.rand(1, 2)
        b_radius = torch.rand(1, 1) / 2.0

        intersections = sphereSphere(a_coords, a_radius, b_coords, b_radius)

        image = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(a_coords.shape[0]):
            cv2.circle(image, (a_coords[i].detach().cpu().numpy()*512).astype(np.int), (a_radius[i,0].detach().cpu().numpy()*512).astype(np.int), color= (0, 0, 255) if intersections[i, 0] else (0, 255, 0))
            cv2.circle(image, (b_coords[i].detach().cpu().numpy()*512).astype(np.int), (b_radius[i,0].detach().cpu().numpy()*512).astype(np.int), color= (0, 0, 255) if intersections[i, 0] else (0, 255, 0))

        cv2.imshow("Test", image)
        cv2.waitKey(0)