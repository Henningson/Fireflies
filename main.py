import os
import cv2
import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr

import Firefly
import LaserEstimation
import depth
import transforms
import laser_torch
import UNet
import rasterization

from tqdm import tqdm


global_scene = None
global_params = None
global_key = None


@dr.wrap_ad(source='torch', target='drjit')
def render(texture, spp=256, seed=1):
    global_params[global_key] = texture
    global_params.update()
    return mi.render(global_scene, global_params, spp=spp, seed=seed, seed_grad=seed+1)


@dr.wrap_ad(source='torch', target='drjit')
def cast_laser(origin, direction):
    surface_interaction = global_scene.ray_intersect(mi.Ray3f(mi.Point3f(origin), mi.Vector3f(direction)))
    result = surface_interaction.t
    result[~surface_interaction.is_valid()] = 0
    return mi.TensorXf(result, shape=(1, len(result)))


def main():
    global global_scene
    global global_params
    global global_key

    base_path = "scenes/RotObject/"
    num_depth_maps = 5
    steps_per_frame = 5
    sequentially_updated = True
    num_point_samples = 150
    weight = 0.001
    save_images = False
    spp=4
    sigma=0.004
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    global_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    global_params = mi.traverse(global_scene)
    global_params['Projector.to_world'] = global_params['PerspectiveCamera_1.to_world']
    global_params.update()
    global_key = "tex.data"

    global_scene = global_scene
    global_params = global_params
    global_key = "tex.data"


    constraint_map = LaserEstimation.generate_epipolar_constraints(global_scene, global_params, DEVICE)

    firefly_scene = Firefly.Scene(global_params, 
                                  base_path, 
                                  sequential_animation=sequentially_updated, 
                                  steps_per_frame=steps_per_frame,
                                  device=DEVICE)

    # Generate random depth maps by uniformly sampling from scene parameter ranges
    depth_maps = depth.random_depth_maps(firefly_scene, global_scene, num_maps=num_depth_maps)

    # Given depth maps, generate probability distribution
    variance_map = LaserEstimation.probability_distribution_from_depth_maps(depth_maps, weight)
    
    # Final multiplication and normalization
    final_sampling_map = variance_map * constraint_map
    final_sampling_map /= final_sampling_map.sum()

    # sample points for laser rays
    chosen_points = LaserEstimation.points_from_probability_distribution(final_sampling_map, num_point_samples)

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
    laser_to_world = global_scene.sensors()[1].world_transform().matrix.torch()[0]
    laser_origin = laser_to_world[0:3, 3]
    tex_size = torch.tensor(global_scene.sensors()[1].film().size(), device=laser_to_world.device)
    near_clip = global_scene.sensors()[1].near_clip()
    far_clip = global_scene.sensors()[1].far_clip()
    # TODO: Can we find a better way to get this fov?
    fov = global_params['PerspectiveCamera_1.x_fov']

    # Sample directions of laser beams from variance map
    laser_dir = LaserEstimation.laser_from_ndc_points(global_scene.sensors()[0],
                            laser_origin,
                            depth_maps,
                            chosen_points,
                            device=DEVICE)


    # Apply inverse rotation of the projector, such that we get a normalized direction
    # The laser direction up until now is in world coordinates!
    local_laser_dir = transforms.transform_directions(laser_dir, laser_to_world.inverse())
    Laser = laser_torch.Laser(laser_to_world, local_laser_dir, fov, near_clip, far_clip)

    # Init U-Net and params
    model = UNet.Model(device=DEVICE).to(DEVICE)
    model.train()
    
    loss_func = torch.nn.MSELoss()
    Laser._rays.requires_grad = True
    iterations = 10000

    optim = torch.optim.Adam([
        {'params': model.parameters(),  'lr': 0.01}, 
        {'params': Laser._rays,         'lr': 0.005}
        ])


    for i in (progress_bar := tqdm(range(iterations))):
        optim.zero_grad()

        #print(Laser._rays[0])

        firefly_scene.randomize()

        points = Laser.projectRaysToNDC()[:, 0:2]
        texture_init = rasterization.rasterize_points(points, sigma, tex_size)

        hitpoints = cast_laser(Laser.originPerRay(), Laser.rays())




        input = render(texture_init.unsqueeze(-1), spp=spp, seed=i)
        
        dense_depth = depth.from_camera_non_wrapped(global_scene, spp).torch()
        dense_depth = dense_depth.reshape(256, 256, spp).mean(dim=-1)
        dense_depth = (dense_depth - dense_depth.min()) / (dense_depth.max() - dense_depth.min())

        '''
        plt.axis("off")
        plt.title("GT")
        plt.imshow(image_init)
        plt.show(block=True)
        '''

        # Generate sparse depth map
        #sparse_depth = depth.from_laser(global_scene, global_params, Laser)
        #sparse_depths.append(torch.stack([sparse_depth, render_init]))
        '''
        print("Init | GT | Depth")
        plt.axis("off")
        plt.title("Sparse Depth")
        plt.imshow(sparse_depth.detach().cpu().numpy())
        plt.show(block=True)
        '''

        # Use U-Net to interpolate
        pred_depth = model(input.moveaxis(-1, 0).unsqueeze(0))

        loss = loss_func(pred_depth.squeeze(), dense_depth)
        loss.backward()

        optim.step()


        with torch.no_grad():
            Laser.randomize_out_of_bounds()
            Laser.normalize_rays()


        progress_bar.set_description("Loss: {0:.4f}".format(loss.item()))
        pred_depth_map = pred_depth[0, 0].unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
        gt_depth_map = dense_depth.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
        rendering = torch.clamp(input, 0, 1).detach().cpu().numpy()
        texture = texture_init.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()

        concat_im = np.hstack([rendering, texture, pred_depth_map, gt_depth_map])

        scale_percent = 300 # percent of original size
        width = int(concat_im.shape[1] * scale_percent / 100)
        height = int(concat_im.shape[0] * scale_percent / 100)
        dim = (width, height)

        concat_im = cv2.resize(concat_im, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Predicted Depth Map", concat_im)
        cv2.waitKey(1)
        cv2.imwrite("ims/{0:05d}.png".format(i), (concat_im*255).astype(np.uint8))


if __name__ == "__main__":
    main()