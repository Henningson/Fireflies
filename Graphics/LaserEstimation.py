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


def probability_distribution_from_depth_maps(
    depth_maps: np.array, 
    uniform_weight: float = 0.0) -> np.array:

    variance_map = depth_maps.std(axis=0)
    variance_map += uniform_weight

    return variance_map

def points_from_probability_distribution(
        prob_distribution: torch.tensor,
        num_samples: int) -> torch.tensor:
    
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
    #pos = mi.UInt32(points.split(split_size=1))

    #scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(0.5), mi.Float(0.5))

    #pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0,
        sample1=sampler.next_1d(),
        sample2=pos,
        sample3=0
    )

    return rays.o.torch(), rays.d.torch()


def get_camera_frustum(sensor, device: torch.cuda.device) -> torch.tensor:
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.size()
    total_samples = 4

    if sampler.wavefront_size() != total_samples:
        sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    #pos = mi.UInt32(points.split(split_size=1))

    #scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float([0.0, 1.0, 0.0, 1.0]), mi.Float([0.0, 0.0, 1.0, 1.0]))

    #pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0,
        sample1=sampler.next_1d(),
        sample2=pos,
        sample3=0
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
    pos = mi.Vector2f(mi.Float(pos % int(film_size[0])),
                mi.Float(pos // int(film_size[1])))

    #pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0,
        sample1=sampler.next_1d(),
        sample2=pos * scale,
        sample3=0
    )

    return rays.o.torch(), rays.d.torch()


def laser_from_ndc_points(sensor,
                            laser_origin,
                            depth_maps,
                            chosen_points,
                            device: torch.cuda.device('cuda')) -> torch.tensor:
    ray_origins, ray_directions = create_rays(sensor, chosen_points)

    # Get camera origin and direction
    camera_origin, camera_direction = get_camera_direction(sensor, device)
    
    camera_origin = sensor.world_transform().translation().torch()

    camera_direction = camera_direction / torch.linalg.norm(camera_direction, dim=-1, keepdims=True)

    # Build plane from depth map
    plane_origin = camera_origin + camera_direction * depth_maps.mean()
    plane_normal = -camera_direction

    # Compute intersections inbetween mean plane and randomly sampled rays
    intersection_distances = intersections.rayPlane(ray_origins, ray_directions, plane_origin, plane_normal)
    world_points = ray_origins + ray_directions*intersection_distances

    laser_dir = world_points - laser_origin
    laser_dir = laser_dir / torch.linalg.norm(laser_dir, dim=-1, keepdims=True)
    return laser_dir



def draw_lines(ax, rayOrigin, rayDirection, ray_length=1.0, color='g'):
    for i in range(rayDirection.shape[0]):
        ax.plot([rayOrigin[i, 0], rayOrigin[i, 0] + ray_length*rayDirection[i, 0]],
                [rayOrigin[i, 1], rayOrigin[i, 1] + ray_length*rayDirection[i, 1]],
                [rayOrigin[i, 2], rayOrigin[i, 2] + ray_length*rayDirection[i, 2]],
                color=color)


def generate_epipolar_constraints(scene, params, device):
    camera_sensor = scene.sensors()[0]

    projector_sensor = scene.sensors()[1]
    proj_xwidth, proj_ywidth = projector_sensor.film().size()
    
    ray_origins, ray_directions = get_camera_frustum(projector_sensor, device)

    near_clip = params['PerspectiveCamera_1.near_clip']
    far_clip = params['PerspectiveCamera_1.far_clip']
    steps = 5
    delta = (far_clip - near_clip / steps)

    projection_points = [ray_origins + (params['PerspectiveCamera_1.near_clip'] + delta*i) * ray_directions for i in range(steps)]
    projection_points = torch.vstack(projection_points)
    epipolar_points   = projection_points

    K = utils.build_projection_matrix(params['PerspectiveCamera.x_fov'], params['PerspectiveCamera.near_clip'], params['PerspectiveCamera.far_clip'])
    CAMERA_TO_WORLD = params["PerspectiveCamera.to_world"].matrix.torch()[0]

    epipolar_points = transforms.transform_points(epipolar_points, CAMERA_TO_WORLD.inverse())
    epipolar_points = transforms.transform_points(epipolar_points, K)[:, 0:2]

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
    #camera_size = camera_size[[1, 0]] # swap image size to Y,X
    
    epi_points_np = line_segments.cpu().numpy()
    epi_points_np = (epi_points_np + 1.0) * 0.5
    epi_points_np = epi_points_np[:, [1, 0]]
    epi_points_np *= camera_size


    image = np.zeros(camera_size[[1, 0]], dtype=np.uint8)
    image = cv2.fillPoly(image, [epi_points_np.astype(int)], color=1)
    cv2.imshow("Epipolar Image", image*255)
    cv2.waitKey(0)
    
    return torch.from_numpy(image).to(device)


def initialize_laser(mitsuba_scene, mitsuba_params, firefly_scene, config, device):
    # Doesnt work, IDK why
    constraint_map = generate_epipolar_constraints(mitsuba_scene, mitsuba_params, device)

    # Generate random depth maps by uniformly sampling from scene parameter ranges
    depth_maps = depth.random_depth_maps(firefly_scene, mitsuba_scene, num_maps=config.n_depthmaps)

    # Given depth maps, generate probability distribution
    variance_map = probability_distribution_from_depth_maps(depth_maps, config.variational_epsilon)
    vm = (variance_map.cpu().numpy()*255).astype(np.uint8)
    vm = cv2.applyColorMap(vm, cv2.COLORMAP_VIRIDIS)
    cv2.imshow("Variance Map", vm)
    cv2.waitKey(1)

    # Final multiplication and normalization
    final_sampling_map = variance_map * constraint_map
    final_sampling_map /= final_sampling_map.sum()

    # Gotta flip this in y direction, since apparently I can't program
    #final_sampling_map = torch.fliplr(final_sampling_map)
    #final_sampling_map = torch.flip(final_sampling_map, (0,))

    # sample points for laser rays
    chosen_points = points_from_probability_distribution(final_sampling_map, config.n_beams)

    vm = variance_map.cpu().numpy()
    cp = chosen_points.cpu().numpy()
    cm = constraint_map.cpu().numpy()
    if config.save_images:
        vm = (vm*255).astype(np.uint8)
        vm = cv2.applyColorMap(vm, cv2.COLORMAP_VIRIDIS)
        vm.reshape(-1, 3)[cp, :] = ~vm.reshape(-1, 3)[cp, :]
        cv2.imwrite("sampling_map.png", vm)
        cm = cm*255
        cv2.imwrite("constraint_map.png", cm)

    # Build laser from Projector constraints
    #tex_size = torch.tensor(mitsuba_scene.sensors()[1].film().size(), device=device)
    near_clip = mitsuba_scene.sensors()[1].near_clip()
    far_clip = mitsuba_scene.sensors()[1].far_clip()
    fov = mitsuba_params['PerspectiveCamera_1.x_fov']

    laser_world = firefly_scene.projector.world()
    laser_origin = laser_world[0:3, 3]
    # Sample directions of laser beams from variance map
    laser_dir = laser_from_ndc_points(mitsuba_scene.sensors()[0],
                            laser_origin,
                            depth_maps,
                            chosen_points,
                            device=device)


    # Apply inverse rotation of the projector, such that we get a normalized direction
    # The laser direction up until now is in world coordinates!
    local_laser_dir = transforms.transform_directions(laser_dir, laser_world.inverse())
    return laser.Laser(firefly_scene.projector, local_laser_dir, fov, near_clip, far_clip)



if __name__ == "__main__":
    test()