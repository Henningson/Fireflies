import os
import cv2
import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_rgb")
import drjit as dr

import Firefly
import LaserEstimation
import depth
import transforms
import laser_torch
import UNet
import rasterization
import Losses

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
    origin_point = mi.Point3f(origin[:, 0].array, origin[:, 1].array, origin[:, 2].array)
    rays_vector = mi.Vector3f(direction[:, 0].array, direction[:, 1].array, direction[:, 2].array)
    surface_interaction = global_scene.ray_intersect(mi.Ray3f(origin_point, rays_vector))
    result = surface_interaction.t
    result[~surface_interaction.is_valid()] = 0
    return mi.TensorXf(result, shape=(len(result), 1))


def main():
    global global_scene
    global global_params
    global global_key


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = "scenes/RotObject/"
    num_depth_maps = 150
    steps_per_frame = 5
    sequentially_updated = True
    num_point_samples = 150
    weight = 0.001
    save_images = True
    spp=4
    sigma=torch.tensor([0.008], device=DEVICE)
    num_upsamples = 3
    upsample_at_iter = 2000
    iterations = 10000


    global_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    global_params = mi.traverse(global_scene)
    global_params['Projector.to_world'] = global_params['PerspectiveCamera_1.to_world']
    global_params.update()
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
    UNET_CONFIG = {
        'in_channels': num_point_samples + 3, 
        'out_channels': 1, 
        'features': [32, 64, 128, 256, 512]}
    model = UNet.Model(config=UNET_CONFIG, device=DEVICE).to(DEVICE)
    model.train()
    
    losses = Losses.Handler([
            [Losses.VGGPerceptual().to(DEVICE), 0.0],
            [torch.nn.MSELoss().to(DEVICE), 1.0],
            #[torch.nn.L1Loss().to(DEVICE),  1.0]
            ])
    
    Laser._rays.requires_grad = True
    sigma.requires_grad = True

    optim = torch.optim.Adam([
        {'params': model.parameters(),  'lr': 0.0001}, 
        {'params': Laser._rays,         'lr': 0.005},
        {'params': sigma,               'lr': 0.00001}
        ])


    upsampling = [global_scene.sensors()[0].film().size() // 2**i for i in range(num_upsamples-1, -1, -1)]
    upsampling_step = 0
    for i in (progress_bar := tqdm(range(iterations))):
        #if i % upsample_at_iter == 0 and upsampling_step + 1 != num_upsamples:
        #    global_params['PerspectiveCamera.film.size'] = upsampling[upsampling_step]
        #    upsampling_step += 1
        #    global_params.update()


        optim.zero_grad()

        #print(Laser._rays[0])


        firefly_scene.randomize()

        points = Laser.projectRaysToNDC()[:, 0:2]
        texture_init = rasterization.rasterize_points(points, sigma, tex_size)
        texture_init = rasterization.softor(texture_init)

        hitpoints = cast_laser(Laser.originPerRay(), Laser.rays())
        world_points = Laser.originPerRay() + hitpoints * Laser.rays()




        # with torch.no_grad():
        #     import matplotlib.pyplot as plt
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     np_world = world_points.detach().cpu().numpy()
        #     ax.scatter(np_world[:, 0], np_world[:, 1], np_world[:, 2])
        #     plt.show()


        ndc_points = transforms.project_to_camera_space(global_params, world_points).squeeze()

        sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)
        sparse_depth = rasterization.rasterize_depth(ndc_points[:, 0:2], ndc_points[:, 2:3], sigma, sensor_size)

        rendered_image = render(texture_init.unsqueeze(-1), spp=spp, seed=i)
        

        # WHERE'S THE MEMORY LEAK
        dense_depth = depth.from_camera_non_wrapped(global_scene, spp).torch()

        dense_depth = dense_depth.reshape(sensor_size[0], sensor_size[1], spp).mean(dim=-1)
        dense_depth = dense_depth - near_clip
        dense_depth = dense_depth / far_clip
        dense_depth = 1 - dense_depth
        
        dense_depth = (dense_depth - dense_depth.min()) / (dense_depth.max() - dense_depth.min())

        sparse_depth = (sparse_depth - sparse_depth.min()) / (sparse_depth.max() - sparse_depth.min())


        '''
        import matplotlib.pyplot as plt
        plt.axis("off")
        plt.title("GT")
        plt.imshow(sparse_depth.detach().cpu().numpy())
        plt.show(block=True)

        # Generate sparse depth map
        #sparse_depth = depth.from_laser(global_scene, global_params, Laser)
        #sparse_depths.append(torch.stack([sparse_depth, render_init]))

        print("Init | GT | Depth")
        plt.axis("off")
        plt.title("Sparse Depth")
        plt.imshow(dense_depth.detach().cpu().numpy())
        plt.show(block=True)
        '''

        # Use U-Net to interpolate
        #pred_depth = model(sparse_depth.moveaxis(-1, 0).unsqueeze(0))
        final_input = torch.vstack([sparse_depth, rendered_image.moveaxis(-1, 0)])
        pred_depth = model(final_input.unsqueeze(0))

        loss = losses(pred_depth.repeat(1, 3, 1, 1), dense_depth.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1))

        # lines = getEpipolarConstraintLines()
        # epc_regularization = torch.nn.MSELoss()(rasterization.softor(lines), lines.sum(dim=0))
        # loss += epc_regularization

        loss.backward()

        optim.step()

        #if i % 100:
        #    dr.registry_clear()

        progress_bar.set_description("Loss: {0:.4f}, Sigma: {1:.4f}".format(loss.item(), sigma.detach().cpu().numpy()[0]))
        with torch.no_grad():
            #torch.cuda.empty_cache()
            Laser.randomize_out_of_bounds()
            Laser.normalize_rays()

            if i % 1 == 0:
                pred_depth_map = pred_depth[0, 0].unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                gt_depth_map = dense_depth.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                #rendering = torch.clamp(input, 0, 1).detach().cpu().numpy()
                rendering = torch.clamp(sparse_depth, 0, 1).sum(dim=0).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                texture = texture_init.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()

                concat_im = np.hstack([rendering, texture, pred_depth_map, gt_depth_map])

                scale_percent = 200 # percent of original size
                width = int(concat_im.shape[1] * scale_percent / 100)
                height = int(concat_im.shape[0] * scale_percent / 100)
                dim = (width, height)

                concat_im = cv2.resize(concat_im, dim, interpolation=cv2.INTER_AREA)
                cv2.imshow("Predicted Depth Map", concat_im)
                cv2.waitKey(1)
                cv2.imwrite("ims/{0:05d}.png".format(i), (concat_im*255).astype(np.uint8))


if __name__ == "__main__":
    main()