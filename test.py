import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
dr.set_flag(dr.JitFlag.LoopRecord, False)
import hello_world
import cv2
import numpy as np
import intersections
import torch
import utils_torch
import transforms_torch
import entity
import Firefly
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import laser_torch
import UNet


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


def random_depth_maps(firefly_scene, mitsuba_scene, num_maps: int = 100) -> np.array:
    stacked_depth_maps = []
    for i in tqdm(range(num_maps)):
        firefly_scene.randomize()


        # TODO: Add image size here
        spp = 10
        depth_map = hello_world.get_depth_map(mitsuba_scene, spp=spp)
        depth_map = depth_map.torch().reshape(256, 256, spp).mean(dim=-1)
        stacked_depth_maps.append(depth_map)


    return torch.stack(stacked_depth_maps)


def get_camera_direction(sensor) -> torch.tensor:
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

def laser_from_variance_map(sensor,
                            laser_origin,
                            depth_maps,
                            var_map,
                            chosen_points) -> np.array:
    ray_origins, ray_directions = create_rays(sensor, chosen_points)

    # Get camera origin and direction
    camera_origin, camera_direction = get_camera_direction(sensor)
    # TODO: Multiply by near clip distance
    
    camera_origin = sensor.world_transform().translation().torch()

    camera_direction = camera_direction / torch.linalg.norm(camera_direction, dim=-1, keepdims=True)

    # Build plane from depth map
    # TODO: Use weighted depths based on variance maps!
    plane_origin = camera_origin + camera_direction * depth_maps.mean()
    plane_normal = -camera_direction

    # Compute intersections inbetween mean plane and randomly sampled rays
    intersection_distances = intersections.rayPlane(ray_origins, ray_directions, plane_origin, plane_normal)
    world_points = ray_origins + ray_directions*intersection_distances

    # TODO: Apply inverse transformations, to get local coordinate system
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
    ray_origins = torch.tensor(ray_origins, device=device)
    ray_directions = torch.tensor(ray_directions, device=device)


    # TODO: Inlcude projector near and far clip here,
    # to ensure optimization in specific epipolar range.
    epipolar_min = ray_origins.mean(dim=0, keepdims=True)
    epipolar_max = ray_origins + 10000 * ray_directions

    K = utils_torch.build_projection_matrix(params['PerspectiveCamera.x_fov'], params['PerspectiveCamera.near_clip'], params['PerspectiveCamera.far_clip'])
    CAMERA_TO_WORLD = params["PerspectiveCamera.to_world"].matrix.torch()
    
    
    # Project points into NDC
    CAMERA_TO_WORLD = CAMERA_TO_WORLD.inverse()

    epipolar_max = transforms_torch.transform_points(epipolar_max, CAMERA_TO_WORLD)
    epipolar_max = transforms_torch.transform_points(epipolar_max, K)
    epipolar_max = transforms_torch.convert_points_from_homogeneous(epipolar_max)[0]

    epipolar_min = transforms_torch.transform_points(epipolar_min, CAMERA_TO_WORLD)
    epipolar_min = transforms_torch.transform_points(epipolar_min, K)
    epipolar_min = transforms_torch.convert_points_from_homogeneous(epipolar_min)[0]


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

def test():
    base_path = "scenes/RotObject/"
    num_depth_maps = 150
    steps_per_frame = 5
    sequentially_updated = True
    num_point_samples = 150
    weight = 0.001
    save_images = True

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    mitsuba_params['Projector.to_world'] = mitsuba_params['PerspectiveCamera_1.to_world']
    mitsuba_params.update()

    constraint_map = generate_epipolar_constraints(mitsuba_scene, mitsuba_params, 'cuda')

    firefly_scene = Firefly.Scene(mitsuba_params, 
                                  base_path, 
                                  sequential_animation=sequentially_updated, 
                                  steps_per_frame=steps_per_frame)

    # Generate random depth maps by uniformly sampling from scene parameter ranges
    depth_maps = random_depth_maps(firefly_scene, mitsuba_scene, num_maps=num_depth_maps)

    # Given depth maps, generate probability distribution
    variance_map = probability_distribution_from_depth_maps(depth_maps, weight)
    
    # Final multiplication and normalization
    final_sampling_map = variance_map * constraint_map
    final_sampling_map /= final_sampling_map.sum()

    # sample points for laser rays
    chosen_points = points_from_probability_distribution(final_sampling_map, num_point_samples)

    vm = variance_map.cpu().numpy()
    cp = chosen_points.cpu().numpy()
    cm = constraint_map.cpu().numpy()
    if save_images:
        vm = (vm*255).astype(np.uint8)
        vm = cv2.applyColorMap(vm, cv2.COLORMAP_VIRIDIS)
        vm.reshape(-1, 3)[cp, :] = ~vm.reshape(-1, 3)[cp, :]
        cv2.imwrite("sampling_map.png", vm)
        cm = cm*255
        cv2.imwrite("constraint_map.png", cm)

    # Build laser from Projector constraints
    laser_to_world = mitsuba_scene.sensors()[1].world_transform().matrix.torch()[0]
    laser_origin = laser_to_world[0:3, 3]
    tex_size = torch.tensor(mitsuba_scene.sensors()[1].film().size(), device=laser_to_world.device)
    near_clip = mitsuba_scene.sensors()[1].near_clip()
    far_clip = mitsuba_scene.sensors()[1].far_clip()
    # TODO: Can we find a better way to get this fov?
    fov = mitsuba_params['PerspectiveCamera_1.x_fov']

    # Sample directions of laser beams from variance map
    laser_dir = laser_from_variance_map(mitsuba_scene.sensors()[0],
                            laser_origin,
                            depth_maps,
                            variance_map,
                            chosen_points)


    # Apply inverse rotation of the projector, such that we get a normalized direction
    # The laser direction up until now is in world coordinates!
    local_laser_dir = transforms_torch.transform_directions(laser_dir, laser_to_world.inverse())
    Laser = laser_torch.Laser(laser_to_world, local_laser_dir, fov, near_clip, far_clip)
    laser_texture = Laser.generateTexture(0.005, tex_size)

    # Change projector texture to laser texture
    mitsuba_params["tex.data"] = mi.TensorXf(laser_texture.unsqueeze(-1))
    mitsuba_params.update()

    # TODO: Implement U-Net
    model = UNet.Model(device='cuda').cuda()
    loss_func = torch.nn.MSELoss()
    Laser._rays.requires_grad = True
    batch_size = 4
    num_batches = 10000

    optim = torch.optim.Adam([
        {'params': model.parameters(),  'lr': 0.01}, 
        {'params': Laser._rays,         'lr': 10.0 }])

    for i in range(num_batches):
        optim.zero_grad()
        gt_depths = []
        sparse_depths = []


        laser_texture = Laser.generateTexture(0.005, tex_size)

        # Change projector texture to laser texture
        mitsuba_params["tex.data"] = mi.TensorXf(laser_texture.unsqueeze(-1))
        mitsuba_params.update()

        rendered_im = None
        
        for b in range(batch_size):
            firefly_scene.randomize()

            if b == 0:
                with torch.no_grad():
                    render_init = mi.render(mitsuba_scene, spp=2)
                    
                    rendered_im = mi.TensorXf(mi.util.convert_to_bitmap(render_init)).torch()
                    rendered_im /= 255.0
                    rendered_im = rendered_im.detach().cpu().numpy()


            spp = 4
            depth = hello_world.get_depth_map(mitsuba_scene, spp)
            depth = depth.torch().reshape(256, 256, spp).mean(dim=-1)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            gt_depths.append(depth)

            '''
            plt.axis("off")
            plt.title("GT")
            plt.imshow(image_init)
            plt.show(block=True)
            '''

            # Generate sparse depth map
            sparse_depth = hello_world.get_sparse_depth_map(mitsuba_scene, mitsuba_params, Laser)
            sparse_depths.append(sparse_depth)
            '''
            print("Init | GT | Depth")
            plt.axis("off")
            plt.title("Sparse Depth")
            plt.imshow(sparse_depth.detach().cpu().numpy())
            plt.show(block=True)
            '''
            
            # Use U-Net to interpolate
        pred_depth_maps = model(torch.stack(sparse_depths).unsqueeze(1))

        loss = loss_func(torch.stack(gt_depths), pred_depth_maps.squeeze())
        loss.backward()

        optim.step()



        pred_depth_map = pred_depth_maps[0, 0].unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
        gt_depth_map = gt_depths[0].unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
        concat_im = np.hstack([rendered_im, pred_depth_map, gt_depth_map])
        cv2.imshow("Predicted Depth Map", concat_im)
        cv2.waitKey(1)
        print(loss)


if __name__ == "__main__":
    test()