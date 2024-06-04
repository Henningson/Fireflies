import torch
import numpy as np
import math
import fireflies.graphics.rasterization
import fireflies.utils.transforms
import fireflies.sampling.poisson
import fireflies.entity.base
import projector
import yaml

from typing import List


class Laser(projector.Projector):
    # Static Convenience Function
    @staticmethod
    def generate_uniform_rays(
        intra_ray_angle: float,
        num_beams_x: int,
        num_beams_y: int,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> torch.tensor:
        laserRays = torch.zeros((num_beams_y * num_beams_x, 3), device=device)

        for x in range(num_beams_x):
            for y in range(num_beams_y):
                laserRays[x * num_beams_x + y, :] = torch.tensor(
                    [
                        math.tan((x - (num_beams_x - 1) / 2) * intra_ray_angle),
                        math.tan((y - (num_beams_y - 1) / 2) * intra_ray_angle),
                        -1.0,
                    ]
                )

        return laserRays / torch.linalg.norm(laserRays, dim=-1, keepdims=True)

    @staticmethod
    def generate_uniform_rays_by_count(
        num_beams_x: int,
        num_beams_y: int,
        intrinsic_matrix: torch.tensor,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> torch.tensor:
        laserRays = torch.zeros((num_beams_y * num_beams_x, 3), device=device)

        x_steps = torch.arange((1 / num_beams_x) / 2, 1, 1 / num_beams_x)
        y_steps = torch.arange((1 / num_beams_y) / 2, 1, 1 / num_beams_y)

        xy = torch.stack(torch.meshgrid(x_steps, y_steps))
        xy = xy.movedim(0, -1).reshape(-1, 2)

        # Set Z to 1
        laserRays[:, 0:2] = xy
        laserRays[:, 2] = -1.0

        # Project to world
        rays = fireflies.utils.transforms.transform_points(
            laserRays, intrinsic_matrix.inverse()
        )

        # Normalize
        rays = rays / torch.linalg.norm(rays, dim=-1, keepdims=True)
        rays[:, 2] *= -1.0
        return rays

    @staticmethod
    def generate_random_rays(
        num_beams: int,
        intrinsic_matrix: torch.tensor,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> torch.tensor:

        # Random points and move into NDC
        spawned_points = (
            torch.ones([num_beams, 3], device=device) * 0.5
            + (torch.rand([num_beams, 3], device=device) - 0.5) / 10.0
        )

        # Set Z to 1
        spawned_points[:, 2] = -1.0

        # Project to world
        rays = fireflies.utils.transforms.transform_points(
            spawned_points, intrinsic_matrix.inverse()
        )

        # Normalize
        rays = rays / torch.linalg.norm(rays, dim=-1, keepdims=True)
        rays[:, 2] *= -1.0
        return rays

    @staticmethod
    def generate_blue_noise_rays(
        image_size_x: int,
        image_size_y: int,
        num_beams: int,
        intrinsic_matrix: torch.tensor,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> torch.tensor:

        # We want to know the radius of the poisson disk so that we get roughly N beams
        #
        # So we say N < (X*Y) / PI*r*r <=> sqrt(X*Y / PI*N) ~ r
        #

        poisson_radius = math.sqrt(
            (image_size_x * image_size_y) / (math.pi * num_beams)
        )
        poisson_radius += poisson_radius / 4.0
        im = np.ones([image_size_x, image_size_y]) * poisson_radius
        num_samples, poisson_samples = fireflies.sampling.poisson.poissonDiskSampling(
            im
        )
        # print(len(poisson_samples))
        poisson_samples = torch.tensor(poisson_samples, device=device)

        # Remove random points from poisson samples such that num_beams is correct again.
        # indices = torch.linspace(0, poisson_samples.shape[0] - 1, poisson_samples.shape[0], device=device)
        # indices = torch.multinomial(indices, num_beams, replacement=False).long()
        # poisson_samples = poisson_samples[indices]

        # From image space to 0 1
        poisson_samples /= torch.tensor([image_size_x, image_size_y], device=device)

        # Create empty tensor for copying
        temp = torch.ones([poisson_samples.shape[0], 3], device=device) * -1.0

        # Copy to temp and add 1 for z coordinate
        temp[:, 0:2] = poisson_samples
        # temp[:, 0:2] = poisson_samples

        # Project to world
        rays = fireflies.utils.transforms.transform_points(
            temp, intrinsic_matrix.inverse()
        )

        # rays = fireflies.utils.transforms.transform_points(
        #    rays,
        #    fireflies.utils.transforms.toMat4x4(
        #        utils_math.getZTransform(0.5 * np.pi, intrinsic_matrix.device)
        #    ),
        # )

        # Normalize
        rays = rays / torch.linalg.norm(rays, dim=-1, keepdims=True)
        rays[:, 2] *= -1.0
        return rays

    def __init__(
        self,
        transformable: fireflies.entity.base.Transformable,
        ray_directions,
        perspective: torch.tensor,
        max_fov: float,
        near_clip: float = 0.01,
        far_clip: float = 1000.0,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        super(Laser, self).__init__(
            self, transformable, perspective, None, max_fov, near_clip, far_clip, device
        )
        self._rays = ray_directions.to(self.device)
        self.device = device

    def rays(self) -> torch.tensor:
        return fireflies.utils.transforms.transform_directions(
            self._rays,
            self._fireflies.transformable.transformable.Transformable.world(),
        )

    def origin(self) -> torch.tensor:
        return self._fireflies.transformable.transformable.Transformable.world()

    def originPerRay(self) -> torch.tensor:
        return (
            self._fireflies.transformable.transformable.Transformable.world()[0:3, 3]
            .unsqueeze(0)
            .repeat(self._rays.shape[0], 1)
        )

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

    def initPoissonDiskSamples(self, width, height, radius):
        return None

    def clamp_to_fov(self, clamp_val: float = 0.95, epsilon: float = 0.0001) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, clamp it back.
        # If randomize is set, spawn a new random laser inside NDC.
        # Else, clamp it to the edge.
        ndc_coords = self.projectRaysToNDC()
        ndc_coords[:, 0:2] = torch.clamp(ndc_coords[:, 0:2], 1 - clamp_val, clamp_val)
        clamped_rays = self.projectNDCPointsToWorld(ndc_coords)
        self._rays[:] = self.normalize(clamped_rays)

    def randomize_laser_out_of_bounds(self) -> None:
        # TODO: Check, if laser beam falls out of fov. If it does, spawn a new randomly in NDC in (-1, 1).
        new_rays = self._rays.clone()

        # No need to transform as rays are in laser space anyway
        ndc_coords = fireflies.utils.transforms.transform_points(
            new_rays, self._perspective
        )
        xy_coords = ndc_coords[:, 0:2]
        out_of_bounds_indices = ((xy_coords >= 1.0) | (xy_coords <= 0.0)).any(dim=1)

        out_of_bounds_points = ndc_coords[out_of_bounds_indices]

        if out_of_bounds_points.nelement() == 0:
            return 0

        new_ray_point = torch.rand(out_of_bounds_points.shape, device=self.device)
        new_ray_point[:, 2] = -1.0

        clamped_rays = self.projectNDCPointsToWorld(new_ray_point)
        new_rays[out_of_bounds_indices] = clamped_rays
        new_rays = self.normalize(new_rays)

        self._rays[:] = new_rays

    def randomize_camera_out_of_bounds(self, ndc_coords) -> None:
        new_rays = self._rays.clone()
        xy_coords = ndc_coords[:, 0:2]
        out_of_bounds_indices = ((xy_coords >= 1.0) | (xy_coords <= -1.0)).any(dim=1)
        out_of_bounds_points = ndc_coords[out_of_bounds_indices]

        if out_of_bounds_points.nelement() == 0:
            return 0

        new_ray_point = torch.rand(out_of_bounds_points.shape, device=self.device)
        new_ray_point[:, 2] = -1.0

        clamped_rays = self.projectNDCPointsToWorld(new_ray_point)
        new_rays[out_of_bounds_indices] = clamped_rays
        new_rays = self.normalize(new_rays)

        self._rays[:] = new_rays

    def normalize(self, tensor: torch.tensor) -> torch.tensor:
        return tensor / torch.linalg.norm(tensor, dim=-1, keepdims=True)

    def normalize_rays(self) -> None:
        self._rays[:] = self.normalize(self._rays)

    def setToWorld(self, to_world: torch.tensor) -> None:
        self._to_world = (
            self._fireflies.transformable.transformable.Transformable.setWorld(to_world)
        )

    def projectRaysToNDC(self) -> torch.tensor:
        # rays_in_world = fireflies.utils.transforms_torch.transform_directions(self._rays, self._to_world)
        FLIP_Y = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        return fireflies.utils.transforms.transform_points(
            self._rays, self._perspective @ FLIP_Y
        )

    def projectNDCPointsToWorld(self, points: torch.tensor) -> torch.tensor:
        FLIP_Y = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        return fireflies.utils.transforms.transform_points(
            points, (self._perspective @ FLIP_Y).inverse()
        )

    def generateTexture(self, sigma: float, texture_size: List[int]) -> torch.tensor:
        points = self.projectRaysToNDC()[:, 0:2]
        return fireflies.graphics.rasterization.rasterize_points(
            points, sigma, texture_size
        )

    def render_epipolar_lines(
        self, sigma: float, texture_size: torch.tensor
    ) -> torch.tensor:
        epipolar_min = self.originPerRay() + self._near_clip * self.rays()
        epipolar_max = self.originPerRay() + self._far_clip * self.rays()

        CAMERA_TO_WORLD = (
            self._fireflies.transformable.transformable.Transformable.world()
        )
        WORLD_TO_CAMERA = CAMERA_TO_WORLD.inverse()

        epipolar_max = fireflies.utils.transforms.transform_points(
            epipolar_max, WORLD_TO_CAMERA
        )
        epipolar_max = fireflies.utils.transforms.transform_points(
            epipolar_max, self._perspective
        )[:, 0:2]

        epipolar_min = fireflies.utils.transforms.transform_points(
            epipolar_min, WORLD_TO_CAMERA
        )
        epipolar_min = fireflies.utils.transforms.transform_points(
            epipolar_min, self._perspective
        )[:, 0:2]

        lines = torch.stack([epipolar_min, epipolar_max], dim=1)

        return fireflies.graphics.rasterization.rasterize_lines(
            lines, sigma, texture_size
        )

    def save(self, filepath: str):
        save_dict = {
            "rays": self._rays.detach().cpu().numpy().tolist(),
            "fov": self._fov,
            "near_clip": self._near_clip,
            "far_clip": self._far_clip,
        }

        with open(filepath, "w") as file:
            yaml.dump(save_dict, file)
