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
import Objects.Transformable as Transformable

import Models.GatedUNet as GatedUNet
import Models.UNet as UNet
import Models.UNetToFlame as UNetToFlame
import Models.LP_UNet_Flame as LP_UNet_Flame
import Models.LP_MLP_Flame as LP_MLP_Flame
import Models.PointNet as PointNet

import Graphics.rasterization as rasterization
import Metrics.Losses as Losses
import Utils.ConfigArgsParser as CAP
import Utils.Args as Args
import Utils.utils as utils
import trimesh
import pyrender



import Utils.printer as printer
from tqdm import tqdm
from torch import autograd

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
    
    flame_key = None
    for key, value in firefly_scene.meshes.items():
        if type(value) == Transformable.FlameShapeModel:
            flame_key = key

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

    # Gotta flip this in y direction, since apparently I can't program
    final_sampling_map = torch.fliplr(final_sampling_map)
    final_sampling_map = torch.flip(final_sampling_map, (0,))

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
    MODEL_CONFIG = {
        'in_channels': 1, 
        'out_channels': 1, 
        'num_beams': config.n_beams,
        #'features': [32, 64, 128, 256, 512], 
        'features': [1024, 512, 256, 128, 64, 32],
        'output_image_size': global_scene.sensors()[0].film().size(), 
        'shape_params': 100, 
        'expression_params': 50}
    
    model = PointNet.Model(config=MODEL_CONFIG, device=DEVICE).to(DEVICE)
    model.train()
    
    losses = Losses.Handler([
            #[Losses.VGGPerceptual().to(DEVICE), 0.0],
            [torch.nn.MSELoss().to(DEVICE), 1.0],
            #[torch.nn.L1Loss().to(DEVICE),  1.0]
            ])
    
    Laser._rays.requires_grad = True
    sigma.requires_grad = True

    optim = torch.optim.Adam([
        {'params': model.parameters(),  'lr': config.lr_model}, 
        {'params': Laser._rays,         'lr': config.lr_laser},
        {'params': sigma,               'lr': config.lr_sigma}
        ])
    #scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, total_iters = config.iterations, power=1.0)

    upsampling = [global_scene.sensors()[0].film().size() // 2**i 
                  for i in range(config.n_upsamples - 1, -1, -1)]
    upsampling_step = 0


    reduction_steps = config.sigma_reduce_end
    sigma_step = (config.sigma - config.sigma_end) / reduction_steps
    with autograd.detect_anomaly():
        for i in (progress_bar := tqdm(range(config.iterations))):
            #if i % upsample_at_iter == 0 and upsampling_step + 1 != num_upsamples:
            #    global_params['PerspectiveCamera.film.size'] = upsampling[upsampling_step]
            #    upsampling_step += 1
            #    global_params.update()

            if i < reduction_steps:
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
            sparse_depth = rasterization.rasterize_depth(ndc_points[:, 0:2], ndc_points[:, 2:3], config.sigma, sensor_size)
            sparse_depth = (sparse_depth - sparse_depth.min()) / (sparse_depth.max() - sparse_depth.min())

            rendered_image = render(texture_init.unsqueeze(-1), spp=config.spp, seed=i)

            # Use U-Net to interpolate
            final_input = torch.vstack([sparse_depth, rendered_image.moveaxis(-1, 0)])
            predicted_vertices, shape_estimates, expression_estimates = model(world_points.unsqueeze(0).transpose(1,2))
            #print(predicted_vertices[0])
            #loss = losses(pred_depth.repeat(1, 3, 1, 1), dense_depth.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1))
            
            loss = torch.zeros(1, device=DEVICE)
            loss = losses(shape_estimates, firefly_scene.meshes[flame_key].shapeParams())
            print(shape_estimates)
            loss = losses(expression_estimates, firefly_scene.meshes[flame_key].expressionParams())


            # Make sure that epipolar lines do not overlap too much
            lines = Laser.render_epipolar_lines(sigma, tex_size)
            #epc_regularization = torch.nn.MSELoss()(rasterization.softor(lines), lines.sum(dim=0))
            #loss += epc_regularization * config.epipolar_constraint_lambda

            # Projected points should also not overlap
            rasterized_points = rasterization.rasterize_points(ndc_points[:, 0:2], config.sigma, sensor_size)
            #loss += torch.nn.MSELoss()(rasterization.softor(rasterized_points), rasterized_points.sum(dim=0)) * 0.0005

            # Lets go for segmentation to projection similarity here
            loss += torch.nn.MSELoss()(rasterization.softor(rasterized_points), segmentation) * config.perspective_segmentation_similarity_lambda




            loss.backward()
            optim.step()
            #scheduler.step()

            progress_bar.set_description("Loss: {0:.4f}, Sigma: {1:.4f}".format(loss.item(), sigma.detach().cpu().numpy()[0]))
            with torch.no_grad():
                Laser.randomize_out_of_bounds()
                Laser.normalize_rays()

                if config.visualize:
                    rendering = torch.clamp(rendered_image, 0, 1).detach().cpu().numpy()
                    #rendering = torch.clamp(sparse_depth, 0, 1).sum(dim=0).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                    texture = texture_init.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                    #epipolar_lines = rasterization.softor(lines).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()

                    concat_im = np.hstack([rendering, texture])

                    scale_percent = config.upscale # percent of original size
                    width = int(concat_im.shape[1] * scale_percent / 100)
                    height = int(concat_im.shape[0] * scale_percent / 100)
                    dim = (width, height)

                    concat_im = cv2.resize(concat_im, dim, interpolation=cv2.INTER_AREA)
                    cv2.imshow("Predicted Depth Map", concat_im)
                    cv2.waitKey(1)
                    if config.save_images:
                        cv2.imwrite("ims/{0:05d}.png".format(i), (concat_im*255).astype(np.uint8))


                    if i % 50 == 0:
                        radian = np.pi / 180.0
                        yaw_mat   = math.getYawTransform(90.0*radian, DEVICE)
                        pitch_mat = math.getPitchTransform(0.0, DEVICE)
                        roll_mat  = math.getRollTransform(0.0, DEVICE)

                        transform = transforms.toMat4x4(yaw_mat @ pitch_mat @ roll_mat)
                        transform_a = transform.clone()
                        transform_b = transform.clone()
                        transform_a[1, 3] = -0.15
                        transform_b[1, 3] = 0.15


                        # Visualize Landmarks
                        # This visualises the static landmarks and the pose dependent dynamic landmarks used for RingNet project
                        vis_vertices = predicted_vertices
                        #print(vis_vertices[0])
                        vertex_colors = np.ones([vis_vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 1.0]

                        pred_tri_mesh = trimesh.Trimesh(firefly_scene.meshes[flame_key].getVertexData()[0].detach().cpu().numpy(), model._flame.faces, vertex_colors=vertex_colors)
                        pred_tri_mesh.apply_transform(transform_a.detach().cpu().numpy() )
                        pred_mesh = pyrender.Mesh.from_trimesh(pred_tri_mesh)
                        
                        tri_mesh = trimesh.Trimesh(vis_vertices.detach().cpu().numpy().squeeze(), model._flame.faces, vertex_colors=vertex_colors)
                        tri_mesh.apply_transform(transform_b.detach().cpu().numpy() @ transforms.toMat4x4(math.getXTransform(np.pi*0.5, DEVICE)).detach().cpu().numpy())
                        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

                        scene = pyrender.Scene()
                        scene.add(mesh)
                        scene.add(pred_mesh)

                        pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=[1920, 1080])

    printer.Printer.OKG("Optimization done. Initiating post-processing.")
    


    Laser.save(os.path.join(args.scene_path, "laser.yml"))
    print("Finished everything.")


if __name__ == "__main__":
    main()