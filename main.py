import os
import cv2
import torch

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import mitsuba as mi

if DEVICE == "cuda":
    mi.set_variant("cuda_ad_rgb")
else:
    mi.set_variant("llvm_ad_rgb")

import drjit as dr

import Graphics.Firefly as Firefly
import Graphics.LaserEstimation as LaserEstimation
import Graphics.depth as depth
import Utils.transforms as transforms
import Utils.math as math
import Objects.laser as laser
import Models.GatedUNet as GatedUNet
import Models.UNet as UNet
import Models.UNetWithMultiInput as UNetWithMultiInput
import Graphics.rasterization as rasterization
import Metrics.Losses as Losses
import Utils.ConfigArgsParser as CAP
import Utils.Args as Args
import Utils.utils as utils

import Utils.printer as printer
from tqdm import tqdm

import imageio


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


def subsampled_point_raster(ndc_points, num_subsamples, sigma, sensor_size):
    subsampled_rastered_depth = []
    for i in range(num_subsamples):
        rastered_depth = rasterization.rasterize_depth(ndc_points[:, 0:2], ndc_points[:, 2:3], sigma, sensor_size // 2**i)
        rastered_depth = rasterization.softor(rastered_depth, keepdim=True)
        rastered_depth = (rastered_depth - rastered_depth.min()) / (rastered_depth.max() - rastered_depth.min())
        subsampled_rastered_depth.append(rastered_depth)
    return subsampled_rastered_depth


def vis_schmexy_depth(ndc_points, depth_image):
    head_depth = (depth_image < 1.3).nonzero()

    
    head_depth = depth_image[head_depth[:, 0], head_depth[:, 1]]
    min_depth = head_depth.min()
    max_depth = head_depth.max()

    depth_image = (depth_image - min_depth) / (max_depth - min_depth)
    depth_image = 1 - torch.clamp(depth_image, 0, 1)
    depth_image = (depth_image.detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imshow("Schmexay Depth", depth_image)
    cv2.waitKey(0)


def main():
    global global_scene
    global global_params
    global global_key

    parser = Args.GlobalArgumentParser()
    args = parser.parse_args()
    config = CAP.ConfigArgsParser(utils.read_config_yaml(os.path.join(args.scene_path, "config.yml")), args)
    config.printFormatted()
    config = config.asNamespace()
    
    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(os.path.join(args.scene_path, "scene.xml"))
    global_params = mi.traverse(global_scene)

    if hasattr(config, 'downscale_factor'):
        global_params['PerspectiveCamera.film.size'] = global_params['PerspectiveCamera.film.size'] // config.downscale_factor
        global_params['PerspectiveCamera_1.film.size'] = global_params['PerspectiveCamera_1.film.size'] // config.downscale_factor

    global_params['Projector.to_world'] = global_params['PerspectiveCamera_1.to_world']
    global_params.update()
    global_key = "tex.data"


    firefly_scene = Firefly.Scene(global_params, 
                                  args.scene_path, 
                                  sequential_animation=config.sequential, 
                                  steps_per_frame=config.steps_per_anim,
                                  device=DEVICE)
    firefly_scene.randomize()

    # Doesnt work, IDK why
    constraint_map = LaserEstimation.generate_epipolar_constraints(global_scene, global_params, DEVICE)

    # Generate random depth maps by uniformly sampling from scene parameter ranges
    depth_maps = depth.random_depth_maps(firefly_scene, global_scene, num_maps=config.n_depthmaps)

    # Given depth maps, generate probability distribution
    variance_map = LaserEstimation.probability_distribution_from_depth_maps(depth_maps, config.variational_epsilon)
    vm = (variance_map.cpu().numpy()*255).astype(np.uint8)
    vm = cv2.applyColorMap(vm, cv2.COLORMAP_VIRIDIS)
    cv2.imshow("Variance Map", vm)
    cv2.waitKey(1)

    # Final multiplication and normalization
    final_sampling_map = variance_map * constraint_map
    final_sampling_map /= final_sampling_map.sum()

    # sample points for laser rays
    chosen_points = LaserEstimation.points_from_probability_distribution(final_sampling_map, config.n_beams)

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
    tex_size = torch.tensor(global_scene.sensors()[1].film().size(), device=DEVICE)
    near_clip = global_scene.sensors()[1].near_clip()
    far_clip = global_scene.sensors()[1].far_clip()
    fov = global_params['PerspectiveCamera_1.x_fov']

    laser_world = firefly_scene.projector.world()
    laser_origin = laser_world[0:3, 3]
    # Sample directions of laser beams from variance map
    laser_dir = LaserEstimation.laser_from_ndc_points(global_scene.sensors()[0],
                            laser_origin,
                            depth_maps,
                            chosen_points,
                            device=DEVICE)


    # Apply inverse rotation of the projector, such that we get a normalized direction
    # The laser direction up until now is in world coordinates!
    local_laser_dir = transforms.transform_directions(laser_dir, laser_world.inverse())
    Laser = laser.Laser(firefly_scene.projector, local_laser_dir, fov, near_clip, far_clip)

    # Init U-Net and params
    UNET_CONFIG = {
        'in_channels': 1, 
        'out_channels': 1, 
        'features': [32, 64, 128, 256, 512]}
    model = UNetWithMultiInput.Model(config=UNET_CONFIG, device=DEVICE).to(DEVICE)
    model.train()
    
    losses = Losses.Handler([
            #[Losses.VGGPerceptual().to(DEVICE), 0.0],
            [torch.nn.MSELoss().to(DEVICE), 1.0],
            #[torch.nn.L1Loss().to(DEVICE),  1.0]
            ])
    
    Laser._rays.requires_grad = True
    sigma.requires_grad = True

    optim = torch.optim.SGD([
        {'params': model.parameters(),  'lr': config.lr_model}, 
        {'params': Laser._rays,         'lr': config.lr_laser * 0.2},
        {'params': sigma,               'lr': config.lr_sigma * 1000000.0}
        ])
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, total_iters = config.iterations, power=0.99)

    upsampling = [global_scene.sensors()[0].film().size() // 2**i 
                  for i in range(config.n_upsamples - 1, -1, -1)]
    upsampling_step = 0


    reduction_steps = config.iterations - config.sigma_reduce_at
    sigma_step = (config.sigma - config.sigma_end) / reduction_steps

    for i in (progress_bar := tqdm(range(config.iterations))):
        #if i % upsample_at_iter == 0 and upsampling_step + 1 != num_upsamples:
        #    global_params['PerspectiveCamera.film.size'] = upsampling[upsampling_step]
        #    upsampling_step += 1
        #    global_params.update()


        if i == 13:
            a = 1

        if i >= config.sigma_reduce_at:
            sigma = sigma - sigma_step

        firefly_scene.randomize()
        segmentation = depth.get_segmentation_from_camera(global_scene).float()
        optim.zero_grad()

        points = Laser.projectRaysToNDC()[:, 0:2]
        texture_init = rasterization.rasterize_points(points, sigma, tex_size)
        texture_init = rasterization.softor(texture_init)

        cv2.imshow("Tex", texture_init.detach().cpu().numpy())
        cv2.waitKey(1)

        cv2.imshow("Seg", segmentation.detach().cpu().numpy().astype(np.uint8)*255)
        cv2.waitKey(1)

        hitpoints = cast_laser(Laser.originPerRay(), Laser.rays())

        world_points = Laser.originPerRay() + hitpoints * Laser.rays()

        ndc_points = transforms.project_to_camera_space(global_params, world_points).squeeze()
        sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)


        # We should remove points, that do not fall into the object itself here.
        image_space_points = (ndc_points[:, 0:2] * 0.5 + 0.5) * sensor_size
        quantized_indices = image_space_points.floor().int()

        object_hits = segmentation[quantized_indices[:, 1], quantized_indices[:, 0]].nonzero().squeeze()
        filtered_ndc_points = ndc_points[object_hits]


        #sparse_depth = rasterization.rasterize_depth(filtered_ndc_points[:, 0:2], filtered_ndc_points[:, 2:3], config.sigma, sensor_size)
        multi_res_depth = subsampled_point_raster(filtered_ndc_points, 6, config.sigma, sensor_size)

        with torch.no_grad():
            for j, mult in enumerate(multi_res_depth):
                cv2.imshow("Res {}".format(j), mult.detach().cpu().numpy().squeeze())

                if i % 500 == 0 and i > 0:
                    cv2.imwrite("Res{}.png".format(j), (mult.detach().cpu().numpy().squeeze() * 255).astype(np.uint8))

                cv2.waitKey(1)
    



        sparse_depth = multi_res_depth[0]

        rendered_image = render(texture_init.unsqueeze(-1), spp=config.spp, seed=i)
        
        dense_depth = depth.from_camera_non_wrapped(global_scene, config.spp).torch()

        dense_depth = dense_depth.reshape(sensor_size[1], sensor_size[0], config.spp).mean(dim=-1)
        #with torch.no_grad():
        #    vis_schmexy_depth(filtered_ndc_points, dense_depth.clone())
        dense_depth = 1 - (dense_depth - dense_depth.min()) / (dense_depth.max() - dense_depth.min())



        # Use U-Net to interpolate
        model_input = [multi_res.unsqueeze(0) for multi_res in multi_res_depth]
        pred_depth = model(model_input)

        loss = losses(pred_depth.repeat(1, 3, 1, 1), dense_depth.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1))

        # Make sure that epipolar lines do not overlap too much
        lines = Laser.render_epipolar_lines(sigma, tex_size)
        epc_regularization = torch.nn.MSELoss()(rasterization.softor(lines), lines.sum(dim=0))
        loss += epc_regularization * config.epipolar_constraint_lambda

        # Projected points should also not overlap
        rasterized_points = rasterization.rasterize_points(ndc_points[:, 0:2], config.sigma, sensor_size)
        loss += torch.nn.MSELoss()(rasterization.softor(rasterized_points), rasterized_points.sum(dim=0)) * 0.0005

        # Lets go for segmentation to projection similarity here
        loss += torch.nn.MSELoss()(rasterization.softor(rasterized_points), segmentation) * config.perspective_segmentation_similarity_lambda

        loss.backward()
        optim.step()
        scheduler.step()

        

        progress_bar.set_description("Loss: {0:.4f}, Sigma: {1:.4f}".format(loss.item(), sigma.detach().cpu().numpy()[0]))
        with torch.no_grad():
            Laser.randomize_out_of_bounds()
            Laser.normalize_rays()

            if config.visualize:
                pred_depth_map = pred_depth[0, 0].unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                gt_depth_map = dense_depth.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                rendering = torch.clamp(rendered_image, 0, 1).detach().cpu().numpy()
                #rendering = torch.clamp(sparse_depth, 0, 1).sum(dim=0).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                texture = texture_init.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                epipolar_lines = rasterization.softor(lines).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()

                concat_im = np.hstack([rendering, texture, pred_depth_map, gt_depth_map])

                scale_percent = config.upscale # percent of original size
                width = int(concat_im.shape[1] * scale_percent / 100)
                height = int(concat_im.shape[0] * scale_percent / 100)
                dim = (width, height)

                concat_im = cv2.resize(concat_im, dim, interpolation=cv2.INTER_AREA)
                cv2.imshow("Predicted Depth Map", concat_im)
                cv2.waitKey(1)

                cv2.imwrite("ims/{0:05d}.png".format(i), (concat_im*255).astype(np.uint8))

    printer.Printer.OKG("Optimization done. Initiating post-processing.")
    


    Laser.save(os.path.join(args.scene_path, "laser.yml"))
    print("Finished everything.")


if __name__ == "__main__":
    main()