import drjit as dr
import torch
import Utils.utils as utils
import numpy as np
import math
import Graphics.rasterization as rasterization
import Utils.transforms as transforms
import Objects.Transformable as Transformable
import Objects.Camera as Camera
import yaml

from typing import List



class Laser(Camera.Camera):
    # Static Convenience Function
    def generate_uniform_rays(intra_ray_angle: float, num_beams_x: int, num_beams_y:int, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
        laserRays = torch.zeros((num_beams_y*num_beams_x, 3), device=device)

        for x in range(num_beams_x):
            for y in range(num_beams_y):
                laserRays[x * num_beams_x + y, :] = torch.tensor([
                    math.tan((x-(num_beams_x - 1) / 2) * intra_ray_angle), 
                    math.tan((y-(num_beams_y - 1) / 2) * intra_ray_angle), 
                    1.0])

        return laserRays / torch.linalg.norm(laserRays, dim=-1, keepdims=True)


    def __init__(self, 
                 transformable: Transformable.Transformable, 
                 ray_directions, 
                 max_fov: float, 
                 near_clip: float = 0.01, 
                 far_clip: float = 1000.0, 
                 device: torch.cuda.device = torch.device("cuda")):
        
        Camera.Camera.__init__(self, transformable, max_fov, near_clip, far_clip, device)
        self._rays = ray_directions.to(self.device)
    

    def rays(self) -> torch.tensor:
        return transforms.transform_directions(self._rays, self._transformable.world())

    
    def origin(self) -> torch.tensor:
        return self._transformable.world()


    def originPerRay(self) -> torch.tensor:
        return self._transformable.world()[0:3, 3].unsqueeze(0).repeat(self._rays.shape[0], 1)


    def near_clip(self) -> float:
        return self._near_clip
    
    def far_clip(self) -> float:
        return self._far_clip
    

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
        clamped_rays = self.projectNDCPointsToWorld(transforms.convert_points_to_homogeneous(clamped_ndc_points))
        self._rays[:] = self.normalize(clamped_rays)


    def randomize_out_of_bounds(self) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, spawn a new randomly in NDC in (-1, 1).
        new_rays = self._rays.clone()
        ndc_coords = transforms.transform_points(new_rays, self._perspective)
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


    def normalize_rays(self) -> None:
        self._rays[:] = self.normalize(self._rays)
    

    def setToWorld(self, to_world: torch.tensor) -> None:
        self._to_world = self._transformable.setWorld(to_world)


    def projectRaysToNDC(self) -> torch.tensor:
        #rays_in_world = transforms_torch.transform_directions(self._rays, self._to_world)
        return transforms.transform_points(self._rays, self._perspective)
    

    def projectNDCPointsToWorld(self, points: torch.tensor) -> torch.tensor:
        return transforms.transform_points(points, self._perspective.inverse())
    

    def generateTexture(self, sigma: float, texture_size: List[int]) -> torch.tensor:
        points = self.projectRaysToNDC()[:, 0:2]
        return rasterization.rasterize_points(points, sigma, texture_size)


    def render_epipolar_lines(self, sigma: float, texture_size: torch.tensor) -> torch.tensor:
        epipolar_min = self.originPerRay() + self._near_clip * self.rays()
        epipolar_max = self.originPerRay() + self._far_clip  * self.rays()

        CAMERA_TO_WORLD = self._transformable.world()
        WORLD_TO_CAMERA = CAMERA_TO_WORLD.inverse()

        epipolar_max = transforms.transform_points(epipolar_max, WORLD_TO_CAMERA)
        epipolar_max = transforms.transform_points(epipolar_max, self._perspective)[:, 0:2]
        #epipolar_max = transforms.convert_points_from_homogeneous(epipolar_max)

        epipolar_min = transforms.transform_points(epipolar_min, WORLD_TO_CAMERA)
        epipolar_min = transforms.transform_points(epipolar_min, self._perspective)[:, 0:2]
        #epipolar_min = transforms.convert_points_from_homogeneous(epipolar_min)
        
        lines = torch.stack([epipolar_min, epipolar_max], dim=1)

        return rasterization.rasterize_lines(lines, sigma, texture_size)


    def save(self, filepath: str):
        save_dict = {
            'rays': self._rays.detach().cpu().numpy().tolist(),
            'fov': self._fov.torch().detach().cpu().numpy()[0],
            'near_clip': self._near_clip,
            'far_clip': self._far_clip
        }

        with open(filepath, 'w') as file:
            yaml.dump(save_dict, file)
        






class DeprecatedLaser:
    def generate_uniform_rays(intra_ray_angle: float, num_beams_x: int, num_beams_y:int, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
        laserRays = torch.zeros((num_beams_y*num_beams_x, 3), device=device)

        for x in range(num_beams_x):
            for y in range(num_beams_y):
                laserRays[x * num_beams_x + y, :] = torch.tensor([
                    math.tan((x-(num_beams_x - 1) / 2) * intra_ray_angle), 
                    math.tan((y-(num_beams_y - 1) / 2) * intra_ray_angle), 
                    1.0])

        return laserRays / torch.linalg.norm(laserRays, dim=-1, keepdims=True)


    def __init__(self, 
                 to_world: torch.tensor, 
                 ray_directions, 
                 max_fov: float, 
                 near_clip: float = 0.01, 
                 far_clip: float = 1000.0, 
                 device: torch.cuda.device = torch.device("cuda")):
        
        self.device = device
        self._to_world = to_world.to(self.device)
        self._rays = ray_directions.to(self.device)
        self._origin = self._to_world[0:3, 3]
        self._perspective = utils.build_projection_matrix(max_fov, near_clip, far_clip).to(self.device)
        self._near_clip = near_clip
        self._far_clip = far_clip

    
    def rays(self) -> torch.tensor:
        return transforms.transform_directions(self._rays, self._to_world)

    
    def origin(self) -> torch.tensor:
        return self._origin


    def originPerRay(self) -> torch.tensor:
        return self._origin.unsqueeze(0).repeat(self._rays.shape[0], 1)


    def near_clip(self) -> float:
        return self._near_clip
    

    def far_clip(self) -> float:
        return self._far_clip
    

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
        clamped_rays = self.projectNDCPointsToWorld(transforms.convert_points_to_homogeneous(clamped_ndc_points))
        self._rays[:] = self.normalize(clamped_rays)


    def randomize_out_of_bounds(self) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, spawn a new randomly in NDC in (-1, 1).
        new_rays = self._rays.clone()
        ndc_coords = transforms.transform_points(new_rays, self._perspective)
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


    def normalize_rays(self) -> None:
        self._rays[:] = self.normalize(self._rays)
    

    def setToWorld(self, to_world: torch.tensor) -> None:
        self._to_world = to_world


    def projectRaysToNDC(self) -> torch.tensor:
        #rays_in_world = transforms_torch.transform_directions(self._rays, self._to_world)
        return transforms.transform_points(self._rays, self._perspective)
    

    def projectNDCPointsToWorld(self, points: torch.tensor) -> torch.tensor:
        return transforms.transform_points(points, self._perspective.inverse())
    

    def generateTexture(self, sigma: float, texture_size: List[int]) -> torch.tensor:
        points = self.projectRaysToNDC()[:, 0:2]
        return rasterization.rasterize_points(points, sigma, texture_size)

    def render_epipolar_lines(self, params, sigma: float, texture_size: torch.tensor) -> torch.tensor:
        epipolar_min = self.originPerRay() + self._near_clip * self.rays()
        epipolar_max = self.originPerRay() + self._far_clip  * self.rays()

        K = utils.build_projection_matrix(params['PerspectiveCamera.x_fov'], params['PerspectiveCamera.near_clip'], params['PerspectiveCamera.far_clip'])
        CAMERA_TO_WORLD = params["PerspectiveCamera.to_world"].matrix.torch()
        
        
        # Project points into NDC
        CAMERA_TO_WORLD = CAMERA_TO_WORLD.inverse()

        epipolar_max = transforms.transform_points(epipolar_max, CAMERA_TO_WORLD)
        epipolar_max = transforms.transform_points(epipolar_max, K)
        epipolar_max = transforms.convert_points_from_homogeneous(epipolar_max)[0]

        epipolar_min = transforms.transform_points(epipolar_min, CAMERA_TO_WORLD)
        epipolar_min = transforms.transform_points(epipolar_min, K)
        epipolar_min = transforms.convert_points_from_homogeneous(epipolar_min)[0]
        
        lines = torch.stack([epipolar_min, epipolar_max], dim=1)

        return rasterization.rasterize_lines(lines, sigma, texture_size)


class OldDeprecatedLaser:
    def __init__(self, num_beams_x: int, num_beams_y: int, intra_ray_angle: float, to_world: torch.tensor, origin: torch.tensor, max_fov: float = None, near_clip: float = 0.01, far_clip: float = 1000.0, device: torch.cuda.device = torch.device("cuda")):
        self.device = device
        intra_ray_angle = intra_ray_angle * math.pi / 180
        self._to_world = to_world.to(self.device)
        self._rays = self.computeRays(intra_ray_angle, num_beams_x, num_beams_y).to(self.device)
        self._origin = origin.to(self.device)
        max_fov = num_beams_x*intra_ray_angle if max_fov is None else max_fov
        self._perspective = utils.build_projection_matrix(max_fov, near_clip, far_clip).to(self.device)

    
    def rays(self) -> torch.tensor:
        return transforms.transform_directions(self._rays, self._to_world)

    
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
        clamped_rays = self.projectNDCPointsToWorld(transforms.convert_points_to_homogeneous(clamped_ndc_points))
        self._rays[:] = self.normalize(clamped_rays)


    def randomize_out_of_bounds(self) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, clamp it back.
        # If randomize is set, spawn a new random laser inside NDC.
        # Else, clamp it to the edge.
        new_rays = self._rays.clone()
        ndc_coords = transforms.transform_points(new_rays, self._perspective)
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
        return transforms.transform_points(self._rays, self._perspective)
    

    def projectNDCPointsToWorld(self, points: torch.tensor) -> torch.tensor:
        return transforms.transform_points(points, self._perspective.inverse())




if __name__ == "__main__":
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    import Graphics.rasterization as rasterization
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
