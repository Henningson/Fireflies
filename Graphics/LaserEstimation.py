import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import cv2
import numpy as np
import Objects.intersections as intersections
import torch

import Utils.transforms as transforms
import Utils.utils as utils

import Graphics.depth as depth
import Objects.laser as laser

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
import Utils.math as utils_math
import Utils.bridson as bridson


import math


def probability_distribution_from_depth_maps(
    depth_maps: np.array, uniform_weight: float = 0.0
) -> np.array:

    variance_map = depth_maps.std(axis=0)
    variance_map += uniform_weight

    return variance_map


def points_from_probability_distribution(
    prob_distribution: torch.tensor, num_samples: int
) -> torch.tensor:

    p = prob_distribution.flatten()
    chosen_points = p.multinomial(num_samples, replacement=False)

    return chosen_points


def get_camera_direction(sensor, device: torch.cuda.device) -> torch.tensor:
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.size()
    total_samples = 1

    if sampler.wavefront_size() != total_samples:
        sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    # pos = mi.UInt32(points.split(split_size=1))

    # scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(0.5), mi.Float(0.5))

    # pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0, sample1=sampler.next_1d(), sample2=pos, sample3=0
    )

    return rays.o.torch(), rays.d.torch()


def get_camera_frustum(sensor, device: torch.cuda.device) -> torch.tensor:
    # film = sensor.film()
    sampler = sensor.sampler()
    # film_size = film.size()
    # total_samples = 4

    # if sampler.wavefront_size() != total_samples:
    #    sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    # pos = mi.UInt32(points.split(split_size=1))

    # scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float([0.0, 1.0, 0.0, 1.0]), mi.Float([0.0, 0.0, 1.0, 1.0]))

    # pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0, sample1=sampler.next_1d(), sample2=pos, sample3=0
    )

    ray_origins = rays.o.torch()
    ray_directions = rays.d.torch()

    # x_transform = transforms.toMat4x4(utils_math.getXTransform(np.pi*0.5, ray_origins.device))
    # ray_origins = transforms.transform_points(ray_origins, x_transform)
    # ray_directions = transforms.transform_directions(ray_directions, x_transform)

    return ray_origins, ray_directions


def getRayFromSensor(sensor, ray_coordinate_in_ndc):
    sampler = sensor.sampler()

    # scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(
        mi.Float([ray_coordinate_in_ndc[0]]), mi.Float([ray_coordinate_in_ndc[1]])
    )

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0, sample1=sampler.next_1d(), sample2=pos, sample3=0
    )

    return rays.o.torch(), rays.d.torch()


def create_rays(sensor, points) -> torch.tensor:
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.size()
    total_samples = points.shape[0]

    if sampler.wavefront_size() != total_samples:
        sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    pos = mi.UInt32(points.split(split_size=1))

    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(
        mi.Float(pos % int(film_size[0])), mi.Float(pos // int(film_size[0]))
    )

    # pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0, sample1=sampler.next_1d(), sample2=pos * scale, sample3=0
    )

    return rays.o.torch(), rays.d.torch()


def laser_from_ndc_points(
    sensor, laser_origin, depth_maps, chosen_points, device: torch.cuda.device("cuda")
) -> torch.tensor:
    ray_origins, ray_directions = create_rays(sensor, chosen_points)

    # Get camera origin and direction
    camera_origin, camera_direction = get_camera_direction(sensor, device)

    camera_origin = sensor.world_transform().translation().torch()

    camera_direction = camera_direction / torch.linalg.norm(
        camera_direction, dim=-1, keepdims=True
    )

    # Build plane from depth map
    plane_origin = camera_origin + camera_direction * depth_maps.mean()
    plane_normal = -camera_direction

    # Compute intersections inbetween mean plane and randomly sampled rays
    intersection_distances = intersections.rayPlane(
        ray_origins, ray_directions, plane_origin, plane_normal
    )
    world_points = ray_origins + ray_directions * intersection_distances

    laser_dir = world_points - laser_origin
    laser_dir = laser_dir / torch.linalg.norm(laser_dir, dim=-1, keepdims=True)
    return laser_dir


def draw_lines(ax, rayOrigin, rayDirection, ray_length=1.0, color="g"):
    for i in range(rayDirection.shape[0]):
        ax.plot(
            [rayOrigin[i, 0], rayOrigin[i, 0] + ray_length * rayDirection[i, 0]],
            [rayOrigin[i, 1], rayOrigin[i, 1] + ray_length * rayDirection[i, 1]],
            [rayOrigin[i, 2], rayOrigin[i, 2] + ray_length * rayDirection[i, 2]],
            color=color,
        )


def generate_epipolar_constraints(scene, params, device):

    camera_sensor = scene.sensors()[0]

    projector_sensor = scene.sensors()[1]
    proj_xwidth, proj_ywidth = projector_sensor.film().size()

    ray_origins, ray_directions = get_camera_frustum(projector_sensor, device)
    camera_origins, camera_directions = get_camera_frustum(camera_sensor, device)

    near_clip = params["PerspectiveCamera_1.near_clip"]
    far_clip = params["PerspectiveCamera_1.far_clip"]
    # steps = 1
    # delta = (far_clip - near_clip / steps)

    projection_points = ray_origins + far_clip * ray_directions
    epipolar_points = projection_points

    # K = utils.build_projection_matrix(params['PerspectiveCamera.x_fov'], params['PerspectiveCamera.near_clip'], params['PerspectiveCamera.far_clip'])
    K = mi.perspective_projection(
        camera_sensor.film().size(),
        camera_sensor.film().crop_size(),
        camera_sensor.film().crop_offset(),
        params["PerspectiveCamera.x_fov"],
        params["PerspectiveCamera.near_clip"],
        params["PerspectiveCamera.far_clip"],
    ).matrix.torch()[0]
    CAMERA_WORLD = params["PerspectiveCamera.to_world"].matrix.torch()[0]
    # CAMERA_WORLD[0:3, 0:3] = CAMERA_WORLD[0:3, 0:3] @ utils_math.getYTransform(np.pi, CAMERA_WORLD.device)

    # mi.perspective_transformation(scene.sensors()[0].film.size())

    epipolar_points = transforms.transform_points(
        epipolar_points, CAMERA_WORLD.inverse()
    )
    epipolar_points = transforms.transform_points(epipolar_points, K)[:, 0:2]

    # Is in [0 -> 1]
    epi_points_np = epipolar_points.detach().cpu().numpy()

    hull = ConvexHull(epi_points_np)
    line_segments = epipolar_points[hull.vertices]

    # We could also calculate the fundamental matrix
    # and use this to estimate epipolar lines here
    # However, we
    # Find closest point between min and max
    # Replace this point by the epipolar minimum
    # This gives us the convex hull of the epipolar constraints
    # In clockwise order

    camera_size = np.array(camera_sensor.film().crop_size())
    # camera_size = camera_size[[1, 0]] # swap image size to Y,X

    epi_points_np = line_segments.cpu().numpy()
    # epi_points_np = epi_points_np[:, [1, 0]]
    epi_points_np *= camera_size

    image = np.zeros(camera_size[[1, 0]], dtype=np.uint8)
    image = cv2.fillPoly(image, [epi_points_np.astype(int)], color=1)
    cv2.imshow("Epipolar Image", image * 255)
    cv2.waitKey(0)

    return torch.from_numpy(image).to(device)


def initialize_laser(
    mitsuba_scene, mitsuba_params, firefly_scene, config, mode, device
):
    projector_sensor = mitsuba_scene.sensors()[1]

    near_clip = mitsuba_scene.sensors()[1].near_clip()
    far_clip = mitsuba_scene.sensors()[1].far_clip()
    laser_fov = float(mitsuba_params["PerspectiveCamera_1.x_fov"][0])
    near_clip = mitsuba_scene.sensors()[1].near_clip()

    radians = math.pi / 180.0

    image_size = torch.tensor(mitsuba_scene.sensors()[1].film().size(), device=device)
    LASER_K = mi.perspective_projection(
        projector_sensor.film().size(),
        projector_sensor.film().crop_size(),
        projector_sensor.film().crop_offset(),
        laser_fov,
        near_clip,
        far_clip,
    ).matrix.torch()[0]
    n_beams = config.n_beams

    local_laser_dir = None
    if mode == "RANDOM":
        local_laser_dir = laser.Laser.generate_random_rays(
            num_beams=n_beams, intrinsic_matrix=LASER_K, device=device
        )
    elif mode == "POISSON":
        local_laser_dir = laser.Laser.generate_blue_noise_rays(
            image_size_x=image_size[0],
            image_size_y=image_size[1],
            num_beams=n_beams,
            intrinsic_matrix=LASER_K,
            device=device,
        )
    elif mode == "GRID":
        grid_width = int(math.sqrt(config.n_beams))
        local_laser_dir = laser.Laser.generate_uniform_rays(
            laser_fov * radians / grid_width,
            num_beams_x=grid_width,
            num_beams_y=grid_width,
            device=device,
        )
    elif mode == "SMARTY":
        # Doesnt work, IDK why
        constraint_map = generate_epipolar_constraints(
            mitsuba_scene, mitsuba_params, device
        )

        # Generate random depth maps by uniformly sampling from scene parameter ranges
        # print(config.n_depthmaps)
        depth_maps = depth.random_depth_maps(
            firefly_scene, mitsuba_scene, num_maps=config.n_depthmaps
        )

        # Given depth maps, generate probability distribution
        variance_map = probability_distribution_from_depth_maps(
            depth_maps, config.variational_epsilon
        )
        variance_map = utils.normalize(variance_map)
        vm = (variance_map.cpu().numpy() * 255).astype(np.uint8)
        vm = cv2.applyColorMap(vm, cv2.COLORMAP_INFERNO)
        cv2.imshow("Variance Map", vm)
        cv2.waitKey(0)

        # Final multiplication and normalization
        final_sampling_map = variance_map  # * constraint_map
        final_sampling_map /= final_sampling_map.sum()

        # Gotta flip this in y direction, since apparently I can't program
        # final_sampling_map = torch.fliplr(final_sampling_map)
        # final_sampling_map = torch.flip(final_sampling_map, (0,))

        # sample points for laser rays

        min_radius = config.sigma / 2
        max_radius = 7 * min_radius
        normalized_sampling = 1 - utils.normalize(final_sampling_map)
        normalized_sampling = (
            min_radius + (max_radius - min_radius) * normalized_sampling
        )
        n_points, points = bridson.poissonDiskSampling(
            normalized_sampling.detach().cpu().numpy(), 50
        )
        points = torch.from_numpy(points).to(device).floor().int()
        chosen_points = points[:, 0] * final_sampling_map.shape[1] + points[:, 1]

        # chosen_points = points_from_probability_distribution(final_sampling_map, config.n_beams)

        vm = variance_map.cpu().numpy()
        cp = chosen_points.cpu().numpy()
        cm = constraint_map.cpu().numpy()
        if config.save_images:
            vm = (vm * 255).astype(np.uint8)
            vm = cv2.applyColorMap(vm, cv2.COLORMAP_VIRIDIS)
            vm.reshape(-1, 3)[cp, :] = ~vm.reshape(-1, 3)[cp, :]
            cv2.imwrite("sampling_map.png", vm)
            cm = cm * 255
            cv2.imwrite("constraint_map.png", cm)

        laser_world = firefly_scene.projector.world()
        laser_origin = laser_world[0:3, 3]
        # Sample directions of laser beams from variance map
        laser_dir = laser_from_ndc_points(
            mitsuba_scene.sensors()[0],
            laser_origin,
            depth_maps,
            chosen_points,
            device=device,
        )

        # Apply inverse rotation of the projector, such that we get a normalized direction
        # The laser direction up until now is in world coordinates!
        local_laser_dir = transforms.transform_directions(
            laser_dir, laser_world.inverse()
        )

    # Flip Y
    # I really gotta fix those coordinate systems...
    local_laser_dir[:, 1] *= -1.0

    return laser.Laser(
        firefly_scene.projector,
        local_laser_dir,
        LASER_K,
        laser_fov,
        near_clip,
        far_clip,
    )
