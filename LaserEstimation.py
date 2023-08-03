import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import cv2
import numpy as np
import intersections
import torch
import utils_torch
import transforms


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
    # TODO: Add device
    center_point = torch.tensor([(sensor.film().size()[0] * sensor.film().size()[1]) // 2], device='cuda')
    return create_rays(sensor, center_point)


def create_rays(sensor, points) -> torch.tensor:
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    total_samples = points.shape[0]

    if sampler.wavefront_size() != total_samples:
        sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    pos = mi.UInt32(points.split(split_size=1))

    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(pos  % int(film_size[1])),
                mi.Float(pos // int(film_size[0])))

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
    proj_ywidth, proj_xwidth = projector_sensor.film().crop_size()
    
    # These values correspond to a flattened array.
    # Upper-Left
    # Upper-Right
    # Lower-Right
    # Lower-Left
    proj_frame_bounds = torch.tensor([0,
                                proj_xwidth - 1,
                                proj_ywidth*proj_xwidth - 1,
                                proj_ywidth*proj_xwidth - proj_xwidth],
                                device=device)



    ray_origins, ray_directions = create_rays(projector_sensor, proj_frame_bounds)
    #ray_origins = torch.tensor(ray_origins, device=device)
    #ray_directions = torch.tensor(ray_directions, device=device)


    # TODO: Inlcude projector near and far clip here,
    # to ensure optimization in specific epipolar range.
    epipolar_min = ray_origins.mean(dim=0, keepdims=True)
    epipolar_max = ray_origins + 10000 * ray_directions

    K = utils_torch.build_projection_matrix(params['PerspectiveCamera.x_fov'], params['PerspectiveCamera.near_clip'], params['PerspectiveCamera.far_clip'])
    CAMERA_TO_WORLD = params["PerspectiveCamera.to_world"].matrix.torch()
    
    
    # Project points into NDC
    CAMERA_TO_WORLD = CAMERA_TO_WORLD.inverse()

    epipolar_max = transforms.transform_points(epipolar_max, CAMERA_TO_WORLD)
    epipolar_max = transforms.transform_points(epipolar_max, K)
    epipolar_max = transforms.convert_points_from_homogeneous(epipolar_max)[0]

    epipolar_min = transforms.transform_points(epipolar_min, CAMERA_TO_WORLD)
    epipolar_min = transforms.transform_points(epipolar_min, K)
    epipolar_min = transforms.convert_points_from_homogeneous(epipolar_min)[0]


    # We could also calculate the fundamental matrix 
    # and use this to estimate epipolar lines here
    # However, we
    # Find closest point between min and max
    # Replace this point by the epipolar minimum
    # This gives us the convex hull of the epipolar constraints
    # In clockwise order
    closest_index = (epipolar_max - epipolar_min).norm(dim=1).argmin()
    epipolar_max[closest_index] = epipolar_min[0]

    camera_size = torch.tensor(camera_sensor.film().crop_size(), device=device)

    epipolar_max = (epipolar_max + 1.0) * 0.5
    epipolar_max *= camera_size[None, ...]


    image = np.zeros(camera_size.cpu().numpy(), dtype=np.uint8)
    image = cv2.fillPoly(image, pts=[epipolar_max.cpu().numpy().astype(int)], color=1)
    image = cv2.flip(image, 0)
    
    return torch.from_numpy(image).to(device)





if __name__ == "__main__":
    test()