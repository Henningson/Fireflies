import torch
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import Graphics.LaserEstimation as LaserEstimation
import cv2

from tqdm import tqdm


@dr.wrap_ad(source='torch', target='drjit')
def from_laser(scene, params, laser):
    sensor = scene.sensors()[0]
    film = sensor.film()
    # TODO: Add device
    size = torch.tensor(film.size(), device='cuda')

    hit_points = LaserEstimation.cast_laser(scene, laser=laser)
    ndc_coords = LaserEstimation.project_to_camera_space(params, hit_points)
    pixel_coords = ndc_coords * 0.5 + 0.5
    pixel_coords = pixel_coords[0, :, 0:2]
    pixel_coords = torch.floor(pixel_coords * size).int()

    mask = torch.zeros(size.tolist(), device=size.device)
    mask[pixel_coords[:, 0], pixel_coords[:, 1]] = 1.0

    depth_map = from_camera_non_wrapped(scene, spp=1)
    depth_map = depth_map.reshape(film.size()[0], film.size()[1])

    return depth_map * mask





def from_camera_non_wrapped(scene, spp=64):
    sensor = scene.sensors()[0]
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    total_samples = dr.prod(film_size) * spp

    if sampler.wavefront_size() != total_samples:
        sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    pos = dr.arange(mi.UInt32, total_samples)

    pos //= spp
    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(pos  % int(film_size[0])),
                mi.Float(pos // int(film_size[0])))

    pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0,
        sample1=sampler.next_1d(),
        sample2=pos * scale,
        sample3=0
    )

    # Intersect rays with the scene geometry
    surface_interaction = scene.ray_intersect(rays)

    # Given intersection, compute the final pixel values as the depth t
    # of the sampled surface interaction
    result = surface_interaction.t

    # Set to zero if no intersection was found
    result[~surface_interaction.is_valid()] = 0

    return result



@dr.wrap_ad(source='drjit', target='torch')
def from_camera(scene, spp=64):
    sensor = scene.sensors()[0]
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    total_samples = dr.prod(film_size) * spp

    if sampler.wavefront_size() != total_samples:
        sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    pos = dr.arange(mi.UInt32, total_samples)

    pos //= spp
    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(pos  % int(film_size[0])),
                mi.Float(pos // int(film_size[0])))

    pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray(
        time=0,
        sample1=sampler.next_1d(),
        sample2=pos * scale,
        sample3=0
    )

    # Intersect rays with the scene geometry
    surface_interaction = scene.ray_intersect(rays)

    # Given intersection, compute the final pixel values as the depth t
    # of the sampled surface interaction
    result = surface_interaction.t

    # Set to zero if no intersection was found
    result[~surface_interaction.is_valid()] = 0

    return result


def random_depth_maps(firefly_scene, mi_scene, num_maps: int = 100, spp: int = 1) -> torch.tensor:
    stacked_depth_maps = []
    im_size = mi_scene.sensors()[0].film().size()


    for i in tqdm(range(num_maps)):
        firefly_scene.randomize()

        depth_map = from_camera_non_wrapped(mi_scene, spp=1)

        vis_depth = depth_map.torch().reshape(im_size[0], im_size[1])
        vis_depth /= vis_depth.max()
        cv2.imshow("Depth", vis_depth.detach().cpu().numpy())
        cv2.waitKey(1)


        depth_map = depth_map.torch().reshape(im_size[0], im_size[1], spp).mean(dim=-1)
        stacked_depth_maps.append(depth_map)

    return torch.stack(stacked_depth_maps)