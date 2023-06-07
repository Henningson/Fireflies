import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import hello_world
import cv2
import numpy as np
import intersections
import torch
import entity
from tqdm import tqdm


def sample_probability_distribution(
    depth_maps: np.array, 
    num_samples: int, 
    uniform_weight: float = 0.0) -> np.array:

    variance_map = depth_maps.var(axis=0)
    variance_map += uniform_weight
    prob_distribution = variance_map / variance_map.sum()

    candidates = np.arange(0, variance_map.size)
    chosen_points = np.random.choice(candidates, num_samples, p=prob_distribution.flatten(), replace=False)

    return chosen_points, variance_map


def random_depth_maps(project_path, num_maps: int = 100) -> np.array:
    scene = mi.load_file(project_path + "scene.xml")
    params = mi.traverse(scene)

    # TODO: Load scene randomizer configs
    # scene_randomizer = SceneRandomizer(project_path, scene)
    # scene_randomizer.generateScene()

    rand_obj = entity.RandomizableEntity("test_random_object.yaml")
    verts_base = params["PLYMesh.vertex_positions"]
    rand_obj.setVertices(torch.tensor(verts_base).reshape(-1, 3).tolist())

    # Init once to not get nan values
    #vertex_data = rand_obj.getVertexData()
    #params["PLYMesh.vertex_positions"] = mi.Float32(vertex_data.flatten())
    #params.update()


    stacked_depth_maps = []
    for i in tqdm(range(num_maps)):
        # TODO: Generate randomized scene!

        vertex_data = rand_obj.getVertexData()
        params["PLYMesh.vertex_positions"] = mi.Float32(vertex_data.flatten())
        params.update()

        depth_map = hello_world.get_depth_map(scene, spp=10)
        stacked_depth_maps.append(depth_map)

        #if i % 10 == 0 and i < num_maps - 10:
        #    dr.registry_clear()

    return np.stack(stacked_depth_maps)


def get_camera_direction(sensor) -> np.array:
    center_point = np.array([(sensor.film().size()[0] * sensor.film().size()[1]) // 2])
    return create_rays(sensor, center_point)

def create_rays(sensor, points) -> np.array:
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    total_samples = points.shape[0]

    if sampler.wavefront_size() != total_samples:
        sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    pos = mi.UInt32(points)

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

    return np.array(rays.o), np.array(rays.d)

def laser_from_variance_map(sensor,
                            laser_origin,
                            depth_maps,
                            var_map,
                            chosen_points) -> np.array:
    ray_origins, ray_directions = create_rays(sensor, chosen_points)

    # Get camera origin and direction
    camera_origin, camera_direction = get_camera_direction(sensor)
    camera_direction = camera_direction / np.linalg.norm(camera_direction, axis=-1, keepdims=True)

    # Build plane from depth map
    # TODO: Use weighted depths based on variance maps!
    plane_origin = camera_origin + camera_direction * depth_maps.mean()
    plane_normal = -camera_direction

    # Compute intersections inbetween mean plane and randomly sampled rays
    intersection_distances = intersections.rayPlane(ray_origins, ray_directions, plane_origin, plane_normal)
    world_points = ray_origins + ray_directions*intersection_distances

    # TODO: Apply inverse transformations, to get local coordinate system
    return world_points - laser_origin


def test():
    project_path = "TestScene/"
    num_depth_maps = 5
    num_point_samples = 15000
    laser_origin = np.array([[5.0, 0.0, 0.0]])
    weight = 0.001
    save_images = False
    scene = mi.load_file(project_path + "scene.xml")

    # Generate random depth maps by uniformly sampling from scene parameter ranges
    depth_maps = random_depth_maps(project_path, num_maps=num_depth_maps)

    # Given depth maps, sample linearized random coordinates for laser pattern
    chosen_points, variance_map = sample_probability_distribution(depth_maps, num_point_samples, weight)

    if save_images:
        variance_map = (variance_map*255).astype(np.uint8)
        variance_map = cv2.applyColorMap(variance_map, cv2.COLORMAP_VIRIDIS)
        variance_map.reshape(-1, 3)[chosen_points, :] = ~variance_map.reshape(-1, 3)[chosen_points, :]
        cv2.imwrite("var_map.png", variance_map)

    laser_dir = laser_from_variance_map(scene.sensors()[0],
                            laser_origin,
                            depth_maps,
                            variance_map,
                            chosen_points)
    laser_dir = laser_dir / np.linalg.norm(laser_dir, axis=-1, keepdims=True)

    print(laser_dir)






if __name__ == "__main__":
    test()