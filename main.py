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
import Utils.bridson

import Utils.printer as printer

from tqdm import tqdm

import kornia
import imageio
import time
import shutil

global_scene = None
global_params = None
global_key = None


@dr.wrap_ad(source="torch", target="drjit")
def render(texture, spp=256, seed=1):
    global_params[global_key] = texture
    global_params.update()
    return mi.render(
        global_scene, global_params, spp=spp, seed=seed, seed_grad=seed + 1
    )


@dr.wrap_ad(source="torch", target="drjit")
def cast_laser(origin, direction):
    origin_point = mi.Point3f(
        origin[:, 0].array, origin[:, 1].array, origin[:, 2].array
    )
    rays_vector = mi.Vector3f(
        direction[:, 0].array, direction[:, 1].array, direction[:, 2].array
    )
    surface_interaction = global_scene.ray_intersect(
        mi.Ray3f(origin_point, rays_vector)
    )
    result = surface_interaction.t
    result[~surface_interaction.is_valid()] = 0
    return mi.TensorXf(result, shape=(len(result), 1))


def subsampled_point_raster(ndc_points, num_subsamples, sigma, sensor_size):
    subsampled_rastered_depth = []
    for i in range(num_subsamples):
        rastered_depth = rasterization.rasterize_depth(
            ndc_points[:, 0:2], ndc_points[:, 2:3], sigma, sensor_size // 2**i
        )
        rastered_depth = rasterization.softor(rastered_depth, keepdim=True)
        # rastered_depth = (rastered_depth - rastered_depth.min()) / (
        #    rastered_depth.max() - rastered_depth.min()
        # )
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


def main(resume: bool = False, save_path: str = None):
    global global_scene
    global global_params
    global global_key

    parser = Args.GlobalArgumentParser()
    args = parser.parse_args()

    config_path = (
        os.path.join(args.scene_path, "config.yml")
        if not resume
        else os.path.join(save_path, "config.yml")
    )
    config = CAP.ConfigArgsParser(utils.read_config_yaml(config_path), args)
    config.printFormatted()
    config = config.asNamespace()

    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(os.path.join(args.scene_path, "scene.xml"))
    global_params = mi.traverse(global_scene)

    if not resume:
        save_base = os.path.join(args.scene_path, "optim")
        save_path = os.path.join(
            save_base,
            f'{time.strftime("%Y-%m-%d-%H:%M:%S")}_{config.pattern_initialization}_{config.iterations}',
        )
        save_render_path = os.path.join(save_path, "render")

        try:
            os.mkdir(save_base)
        except:
            pass

        try:
            os.mkdir(save_path)
        except:
            printer.Printer.Warning(
                f"Folder {save_path} does already exist. Please restart."
            )
            exit()

        if config.save_images:
            try:
                os.mkdir(save_render_path)
            except:
                printer.Printer.Warning(
                    f"Folder {save_render_path} does already exist. Please restart."
                )

        shutil.copy(
            os.path.join(args.scene_path, "config.yml"),
            os.path.join(save_path, "config.yml"),
        )

    if hasattr(config, "downscale_factor"):
        global_params["PerspectiveCamera.film.size"] = (
            global_params["PerspectiveCamera.film.size"] // config.downscale_factor
        )
        global_params["PerspectiveCamera_1.film.size"] = (
            global_params["PerspectiveCamera_1.film.size"] // config.downscale_factor
        )

    global_params["Projector.to_world"] = global_params["PerspectiveCamera_1.to_world"]
    global_params.update()
    global_key = "tex.data"

    firefly_scene = Firefly.Scene(
        global_params,
        args.scene_path,
        sequential_animation=config.sequential,
        steps_per_frame=config.steps_per_anim,
        device=DEVICE,
    )
    firefly_scene.randomize()

    camera_sensor = global_scene.sensors()[0]
    camera_x_fov = global_params["PerspectiveCamera.x_fov"]
    camera_near_clip = global_params["PerspectiveCamera.near_clip"]
    camera_far_clip = global_params["PerspectiveCamera.far_clip"]

    projector_sensor = global_scene.sensors()[0]
    projector_x_fov = global_params["PerspectiveCamera_1.x_fov"]
    projector_near_clip = global_params["PerspectiveCamera_1.near_clip"]
    projector_far_clip = global_params["PerspectiveCamera_1.far_clip"]

    K_CAMERA = mi.perspective_projection(
        camera_sensor.film().size(),
        camera_sensor.film().crop_size(),
        camera_sensor.film().crop_offset(),
        camera_x_fov,
        camera_near_clip,
        camera_far_clip,
    ).matrix.torch()[0]
    K_PROJECTOR = mi.perspective_projection(
        projector_sensor.film().size(),
        projector_sensor.film().crop_size(),
        projector_sensor.film().crop_offset(),
        projector_x_fov,
        projector_near_clip,
        projector_far_clip,
    ).matrix.torch()[0]

    # Build laser from Projector constraints
    tex_size = torch.tensor(global_scene.sensors()[1].film().size(), device=DEVICE)

    Laser = LaserEstimation.initialize_laser(
        global_scene,
        global_params,
        firefly_scene,
        config,
        config.pattern_initialization,
        DEVICE,
    )

    losses = Losses.Handler(
        [
            # [Losses.VGGPerceptual().to(DEVICE), 0.0],
            [torch.nn.MSELoss().to(DEVICE), 1.0],
            # [torch.nn.L1Loss().to(DEVICE), 1.0]
        ]
    )
    loss_values = []

    # Init U-Net and params
    UNET_CONFIG = {
        "in_channels": 1,
        "out_channels": 1,
        "features": [32, 64, 128],
    }
    model = UNetWithMultiInput.Model(config=UNET_CONFIG, device=DEVICE).to(DEVICE)

    Laser._rays.requires_grad = True
    sigma.requires_grad = True

    optim = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": config.lr_model},
            {"params": Laser._rays, "lr": config.lr_laser},
            # {"params": sigma, "lr": config.lr_sigma},
        ]
    )

    start_iter = 0
    if resume:
        state_dict = torch.load(os.path.join(save_path, "model.pth.tar"))
        model.load_from_dict(state_dict)
        optim.load_state_dict(state_dict["optimizer"])
        loss_values = state_dict["losses"]
        start_iter = state_dict["iteration"]

        # laser_config = utils.read_config_yaml(os.path.join(save_path, "laser.yml"))
        # np_rays = np.array(laser_config["rays"])
        Laser._rays = state_dict["laser_rays"]
        resume = False

    model.train()

    with torch.autograd.set_detect_anomaly(True):
        for i in (progress_bar := tqdm(range(start_iter, config.iterations))):
            firefly_scene.randomize()
            # segmentation = depth.get_segmentation_from_camera(global_scene).float()

            points = Laser.projectRaysToNDC()[:, 0:2]
            texture_init = rasterization.rasterize_points(points, sigma, tex_size)
            texture_init = rasterization.softor(texture_init)
            texture_init = kornia.filters.gaussian_blur2d(
                texture_init.unsqueeze(0).unsqueeze(0), (5, 5), (3, 3)
            ).squeeze()

            if i == 16:
                aoidsao = 1

            cv2.imshow("Tex", texture_init.detach().cpu().numpy())
            cv2.waitKey(1)

            # cv2.imshow("Seg", segmentation.detach().cpu().numpy().astype(np.uint8) * 255)

            hitpoints = cast_laser(Laser.originPerRay(), Laser.rays())

            world_points = Laser.originPerRay() + hitpoints * Laser.rays()
            CAMERA_WORLD = global_params["PerspectiveCamera.to_world"].matrix.torch()[0]
            world_points_hat = transforms.transform_points(
                world_points, CAMERA_WORLD.inverse()
            ).squeeze()
            ndc_points = transforms.transform_points(
                world_points_hat, K_CAMERA
            ).squeeze()
            sensor_size = torch.tensor(
                global_scene.sensors()[0].film().size(), device=DEVICE
            )

            # We should remove points, that do not fall into the object itself here.
            image_space_points = ndc_points[:, 0:2] * sensor_size
            # quantized_indices = image_space_points.floor().int()

            # object_hits = (
            #    segmentation[quantized_indices[:, 1], quantized_indices[:, 0]]
            #    .nonzero()
            #    .squeeze()
            # )
            # filtered_ndc_points = ndc_points[object_hits]
            filtered_ndc_points = ndc_points

            # sparse_depth = rasterization.rasterize_depth(filtered_ndc_points[:, 0:2], filtered_ndc_points[:, 2:3], config.sigma, sensor_size)
            multi_res_depth = subsampled_point_raster(
                filtered_ndc_points, 4, config.sigma, sensor_size
            )

            # with torch.no_grad():
            #    for j, mult in enumerate(multi_res_depth):
            #        cv2.imshow("Res {}".format(j), mult.detach().cpu().numpy().squeeze())
            #       cv2.waitKey(1)

            # sparse_depth = multi_res_depth[0]

            rendered_image = render(texture_init.unsqueeze(-1), spp=1, seed=i)

            dense_depth = depth.from_camera_non_wrapped(
                global_scene, config.spp
            ).torch()

            dense_depth = dense_depth.reshape(
                sensor_size[1], sensor_size[0], config.spp
            ).mean(dim=-1)
            # with torch.no_grad():
            #    vis_schmexy_depth(filtered_ndc_points, dense_depth.clone())
            # dense_depth = 1 - (dense_depth - dense_depth.min()) / (
            #    dense_depth.max() - dense_depth.min()
            # )

            # Use U-Net to interpolate
            # input_image = rendered_image.movedim(-1, 0).unsqueeze(0)
            model_input = [multi_res.unsqueeze(0) for multi_res in multi_res_depth]
            # model_input[0] = torch.concat([model_input[0], input_image], dim=1)
            pred_depth = model(model_input)

            loss = losses(
                pred_depth.repeat(1, 3, 1, 1),
                dense_depth.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
            )

            # Make sure that epipolar lines do not overlap too much
            """
            lines = Laser.render_epipolar_lines(sigma, tex_size)
            epc_regularization = torch.nn.MSELoss()(
                rasterization.softor(lines), lines.sum(dim=0)
            )
            loss += epc_regularization * config.epipolar_constraint_lambda

            # Projected points should also not overlap
            rasterized_points = rasterization.rasterize_points(
                ndc_points[:, 0:2], config.sigma, sensor_size
            )
            loss += (
                torch.nn.MSELoss()(
                    rasterization.softor(rasterized_points), rasterized_points.sum(dim=0)
                )
                * 0.0005
            )
            
            # Lets go for segmentation to projection similarity here
            loss += (
                torch.nn.MSELoss()(rasterization.softor(rasterized_points), segmentation)
                * config.perspective_segmentation_similarity_lambda
            )
            """

            loss = loss / config.gradient_accumulation_steps
            loss_values.append(loss.item())
            loss.backward()

            if i > 0 and i % config.gradient_accumulation_steps == 0:
                optim.step()
                optim.zero_grad()
                with torch.no_grad():

                    progress_bar.set_description(
                        "Loss: {0:.4f}, Sigma: {1:.4f}".format(
                            loss.item(), sigma.detach().cpu().numpy()[0]
                        )
                    )

                    Laser.clamp_to_fov(clamp_val=0.99)
                    Laser.normalize_rays()

                    if config.visualize:
                        dense_depth = 1 - (dense_depth - dense_depth.min()) / (
                            dense_depth.max() - dense_depth.min()
                        )
                        pred_depth = 1 - (pred_depth - pred_depth.min()) / (
                            pred_depth.max() - pred_depth.min()
                        )

                        pred_depth_map = (
                            pred_depth[0, 0]
                            .unsqueeze(-1)
                            .repeat(1, 1, 3)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        gt_depth_map = (
                            dense_depth.unsqueeze(-1)
                            .repeat(1, 1, 3)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        rendering = (
                            torch.clamp(rendered_image, 0, 1).detach().cpu().numpy()
                        )
                        # rendering = torch.clamp(sparse_depth, 0, 1).sum(dim=0).unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
                        texture = (
                            texture_init.unsqueeze(-1)
                            .repeat(1, 1, 3)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        """
                        epipolar_lines = (
                            rasterization.softor(lines)
                            .unsqueeze(-1)
                            .repeat(1, 1, 3)
                            .detach()
                            .cpu()
                            .numpy()
                        )"""

                        concat_im = np.hstack(
                            [rendering, texture, pred_depth_map, gt_depth_map]
                        )

                        scale_percent = config.upscale  # percent of original size
                        width = int(concat_im.shape[1] * scale_percent / 100)
                        height = int(concat_im.shape[0] * scale_percent / 100)
                        dim = (width, height)

                        concat_im = cv2.resize(
                            concat_im, dim, interpolation=cv2.INTER_AREA
                        )
                        cv2.imshow("Predicted Depth Map", concat_im)
                        cv2.waitKey(1)

                        if config.save_images:
                            cv2.imwrite(
                                os.path.join(save_render_path, f"{i:05d}.png"),
                                (concat_im * 255).astype(np.uint8),
                            )

            # if i == 10:
        #    resume = True
        #    print("Restarting.")
        #    break

    printer.Printer.OKG("Saving")
    checkpoint = {
        "optimizer": optim.state_dict(),
        "laser_rays": Laser._rays,
        "iteration": i + 1,
        "losses": loss_values,
    }
    checkpoint.update(model.get_statedict())

    torch.save(checkpoint, os.path.join(save_path, "model.pth.tar"))

    return resume, save_path


if __name__ == "__main__":
    resume = False
    save_path = None

    resume, save_path = main(resume, save_path)
    while resume:
        resume, save_path = main(resume, save_path)
