import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import drjit as dr
import matplotlib.pyplot as plt
import imageio
import numpy as np
import torch
import transforms_torch
from laser_torch import Laser
import math
import rasterization
import utils_torch



def zero_normalize(ref: np.array) -> np.array:
    return (ref - ref.mean()) / ref.std()


def random_scene(scene_base: list):
    # TODO: Implement me
    return scene_base[0]

def cast_laser(scene, laser: Laser) -> torch.tensor:
    rays = laser.rays()
    origin = laser.origin()

    ts = (scene.ray_intersect(mi.Ray3f(o=mi.Vector3f(origin.unsqueeze(0)), d=rays))).t

    hit_points = origin + rays*ts.torch().unsqueeze(-1)
    hit_points = hit_points[~torch.isinf(hit_points).any(axis=1)]

    return hit_points

def cast_rays(scene, ray_origin, rays):
    ts = np.zeros((rays.shape[0]))
    si = scene.ray_intersect(mi.Ray3f(o=ray_origin, d=rays))
    ts = si.t

    hit_points = ray_origin + rays*np.expand_dims(ts, -1)
    hit_points = hit_points[ts > 0]

    return hit_points

def build_projection_matrix(params) -> np.array:
    fov = params["sensor.x_fov"][0]
    near_clip = params["sensor.near_clip"]
    far_clip = params["sensor.far_clip"]

    S = 1.0 / math.tan((fov / 2.0) * (math.pi / 180))
    z_far = -far_clip / (far_clip - near_clip)
    z_near = - (far_clip * near_clip) / (far_clip - near_clip)

    projection_matrix = np.zeros([4, 4], dtype=float)
    projection_matrix[0, 0] = S
    projection_matrix[1, 1] = S
    projection_matrix[2, 2] = z_far
    projection_matrix[2, 3] = -1.0
    projection_matrix[3, 2] = z_near

    return projection_matrix


# TODO: Refactor
def project_to_camera_space(params, points) -> torch.tensor:
    x_fov = params['PerspectiveCamera.x_fov']
    near_clip = params['PerspectiveCamera.near_clip']
    far_clip = params['PerspectiveCamera.far_clip']
    # TODO: Refactor
    perspective = utils_torch.build_projection_matrix(x_fov, near_clip, far_clip).to('cuda')
    
    camera_to_world = params["PerspectiveCamera.to_world"].matrix.torch()

    view_space_points = transforms_torch.transform_points(points, camera_to_world.inverse())
    ndc_points = transforms_torch.transform_points(view_space_points, perspective)
    return ndc_points


def optimization_task(scene_base, laser, camera, model, batch_size, optimizer):
    # We need to make explicit gradient steps :(
    # At least I guess
    model_gradients = []
    for i in range(batch_size):
        randomized_scene = random_scene(scene_base)
        dense_gt_depth = generate_depth_map(randomized_scene)

        sparse_depth_map = ray_cast(randomized_scene, laser, camera)
        dense_depth_map = model(sparse_depth_map)

        loss = mse(dense_gt_depth, dense_depth_map)
        loss.backward()

# Using zero-normalized cross correlation
# We assume image and projection are rectified and in binary format
# For unbatched input
def compute_disparity(image: torch.tensor, 
                      projection: torch.tensor, 
                      window_size: int = 5,
                      pixel_displacement: int = 9):
    
    disparity = torch.zeros(image.size())
    image_norm = image
    proj_norm = projection
    pad = (pixel_displacement + window_size) // 2
    window_pad = window_size // 2

    proj_padded = torch.nn.functional.pad(proj_norm, (pad, pad, window_pad, window_pad), mode="constant", value=0.0)
    proj_patches = proj_padded.unfold(1, pixel_displacement // 2 + window_size, 1).unfold(0, window_size, 1)

    image_padded = torch.nn.functional.pad(image_norm, (window_pad, window_pad, window_pad, window_pad), mode="constant", value=0.0)
    image_patches = image_padded.unfold(0, window_size, 1).unfold(1, window_size, 1)

    zncc = torch.diagonal(torch.nn.functional.conv2d(proj_patches, image_patches)).squeeze().transpose(0, 1)

    # Check this here
    result = None

    # Generate final disparity map somehow

    return disparity


def get_sparse_depth_map(scene, params, laser):
    sensor = scene.sensors()[0]
    film = sensor.film()
    # TODO: Add device
    size = torch.tensor(film.size(), device='cuda')

    hit_points = cast_laser(scene, laser=laser)
    ndc_coords = project_to_camera_space(params, hit_points)
    pixel_coords = ndc_coords * 0.5 + 0.5
    pixel_coords = pixel_coords[0, :, 0:2]
    pixel_coords = torch.floor(pixel_coords * size).int()

    mask = torch.zeros(size.tolist(), device=size.device)
    mask[pixel_coords[:, 0], pixel_coords[:, 1]] = 1.0

    depth_map = get_depth_map(scene, spp=1)
    depth_map = depth_map.torch().reshape(256, 256)

    return depth_map * mask

@dr.wrap_ad(source='drjit', target='torch')
def get_depth_map(scene, spp=64):
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
    rays, weights = sensor.sample_ray_differential(
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

def mse(image, image_ref):
    return dr.mean(dr.sqr(image - image_ref))


#image = (torch.rand((512, 512)) > 0.5) * 1.0
#proj = torch.nn.functional.pad(image, (2, 0, 0, 0), mode="constant", value=0.0)[:, 0:512]
#disp_map = compute_disparity(image, proj)

def main():
    scene_init = mi.load_file("TestScene/scene.xml", spp=1)
    params = mi.traverse(scene_init)



    render_init = mi.render(scene_init, spp=1)
    image_init = torch.tensor(render_init)[:, :, -1].detach().cpu().numpy()
    #image_init = mi.util.convert_to_bitmap(render_init)

    print("Init | GT | Depth")
    plt.axis("off")
    plt.title("GT")
    plt.imshow(image_init)
    plt.show(block=True)


    exit()
    '''laser = Laser(100, 100, 0.005, 
                np.array([[1.0, 0.0, 0.0, 0.0], 
                                [0.0, 1.0, 0.0, 0.0], 
                                [0.0, 0.0, -1.0, 0.0], 
                                [0.0, 0.0, 0.0, 1.0]]),
                np.array([0.0, 0.0, 6.0]))
    '''

    #laser_hit_points = cast_laser(scene_init, laser)
    #projected_points = project_to_camera_space(mi.traverse(scene_init), laser_hit_points)
    #projected_points /= projected_points[:, 2:]

    scene_gt   = mi.load_file("TestScene/test_scene.xml", spp=256)
    render_gt = mi.render(scene_gt, spp=256)
    image_gt = mi.util.convert_to_bitmap(render_gt)

    depth = np.expand_dims(depth, -1).repeat(3, 2)
    image_init = np.array(image_init).astype(float) / 255
    image_gt = np.array(image_gt).astype(float) / 255




    optimizer = mi.ad.Adam(lr=0.05)
    #optimizer[key_red] = mi.Color3f(params[key_red])
    #optimizer[key_green] = mi.Color3f(params[key_green])   
    optimizer["tex.data"] = mi.TensorXf(params["tex.data"])
    params.update(optimizer)

    iterations = 300
    writer = imageio.get_writer('test.mp4', fps=25)

    for it in range(iterations):
        image = mi.render(scene_init, params, spp=32)
        loss = mse(image, render_gt)
        dr.backward(loss)

        optimizer.step()
        
        #optimizer[key_red] = dr.clamp(optimizer[key_red], 0.0, 1.0)
        #optimizer[key_green] = dr.clamp(optimizer[key_green], 0.0, 1.0)

        params.update(optimizer)
        #err_red = dr.sum(dr.sqr(ref_red - params[key_red]))
        #err_green = dr.sum(dr.sqr(ref_green - params[key_green]))
        print(f"Iteration {it:02d}", end='\r')

        bitmap = mi.util.convert_to_bitmap(image)
        bitmap = np.array(bitmap)
        writer.append_data(bitmap)

        opt_texture = mi.util.convert_to_bitmap(params["tex.data"])
        mi.util.write_bitmap("opt_tex.png", opt_texture)
    writer.close()




    print('\nOptimization complete.')









if __name__ == "__main__":
    main()