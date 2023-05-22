import mitsuba as mi
import drjit as dr
import torch
import utils_torch
import numpy as np
import math
import transforms_np
import transforms_torch

class TorchLaser:
    def __init__(self, num_beams_x: int, num_beams_y: int, intra_ray_angle: float, to_world: torch.tensor, origin: torch.tensor, max_fov: float = None, near_clip: float = 0.01, far_clip: float = 1000.0):
        
        intra_ray_angle = intra_ray_angle * math.pi / 180
        self._to_world = to_world
        self._rays = self.computeRays(intra_ray_angle, num_beams_x, num_beams_y)
        self._origin = origin
        max_fov = num_beams_x*intra_ray_angle if max_fov is None else max_fov
        self._perspective = utils_torch.build_projection_matrix(max_fov, near_clip, far_clip)

    
    def rays(self) -> torch.tensor:
        return transforms_torch.transform_directions(self._rays, self._to_world)

    
    def origin(self) -> torch.tensor:
        return self._origin


    def computeRays(self, intra_ray_angle: float, num_beams_x: int, num_beams_y) -> torch.tensor:
        laserRays = torch.zeros((num_beams_y*num_beams_x, 3))

        for x in range(num_beams_x):
            for y in range(num_beams_y):
                laserRays[x * num_beams_x + y, :] = torch.tensor([
                    math.tan((x-(num_beams_x - 1) / 2) * intra_ray_angle), 
                    math.tan((y-(num_beams_y - 1) / 2) * intra_ray_angle), 
                    1.0])

        return self.normalize(laserRays)
    

    def clamp_to_fov(self, randomize: bool = False) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, clamp it back.
        # If randomize is set, spawn a new random laser inside NDC.
        # Else, clamp it to the edge.
        ndc_coords = self.projectRaysToNDC()
        xy_coords = ndc_coords[:, 0:2]
        out_of_bounds_indices = ((xy_coords > 1.0) | (xy_coords < -1.0)).any(dim=1)
        
        out_of_bounds_points = ndc_coords[out_of_bounds_indices]
        
        if out_of_bounds_points.nelement() == 0:
            return 0

        if not randomize:
            out_of_bounds_points = torch.clamp(out_of_bounds_points, -1.0, 1.0)
        else:
            out_of_bounds_points = torch.rand(out_of_bounds_points.shape) * 2.0 - 1.0
        
        clamped_rays = self.projectNDCPointsToWorld(out_of_bounds_points)
        self._rays[out_of_bounds_indices] = clamped_rays

        self._rays = self.normalize(self._rays)

        import matplotlib.pyplot as plt
        test = self.projectRaysToNDC()
        plt.scatter(test[:, 0], test[:, 1])
        plt.show()


    def normalize(self, tensor: torch.tensor) -> torch.tensor:
        return tensor / torch.linalg.norm(tensor, dim=-1, keepdims=True)
    

    def setToWorld(self, to_world: torch.tensor) -> None:
        self._to_world = to_world


    def projectRaysToNDC(self) -> torch.tensor:
        return transforms_torch.transform_points(self._rays, self._perspective)
    
    def projectNDCPointsToWorld(self, points: torch.tensor) -> torch.tensor:
        return transforms_torch.transform_points(points, self._perspective.inverse())



class Laser:
    def __init__(self, num_beams_x: int, num_beams_y: int, intra_ray_angle: float, to_world: np.array, origin: np.array):
        self._to_world = to_world
        self._rays = self.computeRays(intra_ray_angle, num_beams_x, num_beams_y)
        self._origin = origin

    
    def rays(self) -> np.array:
        return transforms_np.transform_directions(self._rays, self._to_world)

    
    def origin(self) -> np.array:
        return self._origin


    def computeRays(self, intra_ray_angle: float, num_beams_x: int, num_beams_y) -> np.array:
        laserRays = np.zeros((num_beams_y*num_beams_x, 3))

        for x in range(num_beams_x):
            for y in range(num_beams_y):
                laserRays[x * num_beams_x + y, :] = np.array([
                    math.tan((x-(num_beams_x - 1) / 2) * intra_ray_angle), 
                    math.tan((y-(num_beams_y - 1) / 2) * intra_ray_angle), 
                    1.0])

        return self.normalize(laserRays)
    

    def clamp_to_fov(self) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, clamp it back.
        pass


    def normalize(self, tensor: np.array) -> np.array:
        return tensor / tensor.sum(axis=-1, keepdims=True)
    

    def setToWorld(self, to_world: np.array) -> None:
        self._to_world = to_world



if __name__ == "__main__":
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    import rasterization
    import matplotlib.pyplot as plt
    

    laser = TorchLaser(20, 20, 0.5, torch.eye(4), torch.tensor([0.0, 0.0, 0.0]), max_fov=9)
    laser.clamp_to_fov(randomize=True)

    points = laser.projectRaysToNDC()
    sigma = 0.005
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
