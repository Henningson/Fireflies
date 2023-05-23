import mitsuba as mi
import drjit as dr
import torch
import utils_torch
import numpy as np
import math
import transforms_np
import transforms_torch

class Laser:
    def __init__(self, num_beams_x: int, num_beams_y: int, intra_ray_angle: float, to_world: torch.tensor, origin: torch.tensor, max_fov: float = None, near_clip: float = 0.01, far_clip: float = 1000.0, device: torch.cuda.device = torch.device("cuda")):
        
        self.device = device
        intra_ray_angle = intra_ray_angle * math.pi / 180
        self._to_world = to_world.to(self.device)
        self._rays = self.computeRays(intra_ray_angle, num_beams_x, num_beams_y).to(self.device)
        self._origin = origin.to(self.device)
        max_fov = num_beams_x*intra_ray_angle if max_fov is None else max_fov
        self._perspective = utils_torch.build_projection_matrix(max_fov, near_clip, far_clip).to(self.device)

    
    def rays(self) -> torch.tensor:
        return transforms_torch.transform_directions(self._rays, self._to_world)

    
    def origin(self) -> torch.tensor:
        return self._origin


    def computeRays(self, intra_ray_angle: float, num_beams_x: int, num_beams_y) -> torch.tensor:
        laserRays = torch.zeros((num_beams_y*num_beams_x, 3), device=self.device)

        for x in range(num_beams_x):
            for y in range(num_beams_y):
                laserRays[x * num_beams_x + y, :] = torch.tensor([
                    math.tan((x-(num_beams_x - 1) / 2) * intra_ray_angle), 
                    math.tan((y-(num_beams_y - 1) / 2) * intra_ray_angle), 
                    1.0])

        return self.normalize(laserRays)
    

    def initRandomRays(self):
        # Spawn random points in [-1.0, 1.0]
        spawned_points = torch.rand(self._rays.shape, device=self.device) * 2.0 - 1.0

        # Set Z to 1
        spawned_points[:, 2] = 1.0

        # Project to world
        rand_rays = self.projectNDCPointsToWorld(spawned_points)
        self._rays = self.normalize(rand_rays)


    def clamp_to_fov(self, clamp_val: float = 0.95, epsilon: float = 0.001) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, clamp it back.
        # If randomize is set, spawn a new random laser inside NDC.
        # Else, clamp it to the edge.
        ndc_coords = self.projectRaysToNDC()
        xy_coords = ndc_coords[:, 0:2]
        out_of_bounds_indices = ((xy_coords > 1.0 - epsilon) | (xy_coords < -1.0 + epsilon)).any(dim=1)
        
        out_of_bounds_points = ndc_coords[out_of_bounds_indices]
        
        if out_of_bounds_points.nelement() == 0:
            return 0
        
        clamped_ndc_points = torch.clamp(xy_coords, -clamp_val, clamp_val)
        clamped_rays = self.projectNDCPointsToWorld(transforms_torch.convert_points_to_homogeneous(clamped_ndc_points))
        self._rays[:] = self.normalize(clamped_rays)

    def randomize_out_of_bounds(self) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, clamp it back.
        # If randomize is set, spawn a new random laser inside NDC.
        # Else, clamp it to the edge.
        new_rays = self._rays.clone()
        ndc_coords = transforms_torch.transform_points(new_rays, self._perspective)
        xy_coords = ndc_coords[:, 0:2]
        out_of_bounds_indices = ((xy_coords > 1.0) | (xy_coords < -1.0)).any(dim=1)
        
        out_of_bounds_points = ndc_coords[out_of_bounds_indices]
        
        if out_of_bounds_points.nelement() == 0:
            return 0
        
        out_of_bounds_points = torch.rand(out_of_bounds_points.shape, device=self.device) * 2.0 - 1.0
        
        clamped_rays = self.projectNDCPointsToWorld(out_of_bounds_points)
        new_rays[out_of_bounds_indices] = clamped_rays
        new_rays = self.normalize(new_rays)


        self._rays[:] = new_rays

    def normalize(self, tensor: torch.tensor) -> torch.tensor:
        return tensor / torch.linalg.norm(tensor, dim=-1, keepdims=True)
    

    def setToWorld(self, to_world: torch.tensor) -> None:
        self._to_world = to_world


    def projectRaysToNDC(self) -> torch.tensor:
        return transforms_torch.transform_points(self._rays, self._perspective)
    
    def projectNDCPointsToWorld(self, points: torch.tensor) -> torch.tensor:
        return transforms_torch.transform_points(points, self._perspective.inverse())


if __name__ == "__main__":
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    import rasterization
    import matplotlib.pyplot as plt
    

    laser = Laser(20, 20, 0.5, torch.eye(4), torch.tensor([0.0, 0.0, 0.0]), max_fov=9)
    laser.initRandomRays()

    points = laser.projectRaysToNDC()[:, 0:2]
    sigma = 0.001
    texture_size = torch.tensor([512, 512])


    texture = rasterization.rasterize_points(points, sigma, texture_size)
    scene_init = mi.load_file("scenes/proj_cbox.xml", spp=1024)
    params = mi.traverse(scene_init)
    
    params["tex.data"] = mi.TensorXf(texture.cuda().unsqueeze(-1).repeat(1, 1, 3))
    params.update()

    render_init = mi.render(scene_init, spp=1024)
    image_init = mi.util.convert_to_bitmap(render_init)



    print("Init | GT | Depth")
    plt.axis("off")
    plt.title("GT")
    plt.imshow(image_init)
    plt.show(block=True)
