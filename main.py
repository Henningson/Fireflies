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
import Utils.math_helper as math_helper
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
from Metrics.evaluation import RSME, MAE, EvaluationCriterion

import Utils.printer as printer

from tqdm import tqdm

import kornia
import imageio
import time
import shutil

firefly_scene = None
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


def main(config, args):
    global firefly_scene
    global global_scene
    global global_params
    global global_key

    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(os.path.join(args.scene_path, "scene.xml"))
    global_params = mi.traverse(global_scene)

    save_base = os.path.join(args.scene_path, "optim")
    save_path = os.path.join(
        save_base,
        f'{time.strftime("%Y-%m-%d-%H:%M:%S")}_{config.pattern_initialization}_{config.iterations}_{config.n_beams}_{config.lr_laser}',
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

    config_args.save(os.path.join(save_path, "config.yml"))

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
    firefly_scene.train()
    firefly_scene.randomize()

    camera_sensor = global_scene.sensors()[0]
    camera_x_fov = global_params["PerspectiveCamera.x_fov"]
    camera_near_clip = global_params["PerspectiveCamera.near_clip"]
    camera_far_clip = global_params["PerspectiveCamera.far_clip"]

    K_CAMERA = mi.perspective_projection(
        camera_sensor.film().size(),
        camera_sensor.film().crop_size(),
        camera_sensor.film().crop_offset(),
        camera_x_fov,
        camera_near_clip,
        camera_far_clip,
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
            # [torch.nn.L1Loss().to(DEVICE), 1.0],
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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=config.scheduler_step_at, gamma=0.5
    )

    metrics = [EvaluationCriterion(RSME), EvaluationCriterion(MAE)]

    model.train()
    for i in (progress_bar := tqdm(range(config.iterations))):
        loss = train(
            optim,
            scheduler,
            losses,
            model,
            config,
            sigma,
            tex_size,
            Laser,
            K_CAMERA,
            loss_values,
        )

        progress_bar.set_description(f"Loss: {loss:.5f}")
        with torch.no_grad():
            if i > 0 and (i % config.eval_at_every == 0 or i == config.iterations - 1):
                model.eval()
                firefly_scene.eval()

                eval_iterations = (
                    config.eval_iter_interim
                    if i != config.iterations - 1
                    else config.eval_iter_final
                )
                eval(
                    model,
                    config,
                    Laser,
                    sigma,
                    tex_size,
                    K_CAMERA,
                    metrics,
                    eval_iterations,
                )

                for metric in metrics:
                    metric.save(save_path, i)

                render, pred, gt, tex = get_visualization(
                    model, config, tex_size, Laser, K_CAMERA
                )
                if config.visualize:
                    visualize(render, pred, gt, tex)

                if config.save_images:
                    save_images(render, pred, gt, tex, save_path, i)

                model.train()
                firefly_scene.train()

            if not config.save_images and i == config.iterations - 1:
                render, pred, gt, tex = get_visualization(
                    model, config, tex_size, Laser, K_CAMERA
                )
                save_images(render, pred, gt, tex, save_path, i)

    save_checkpoint(model, optim, Laser, i, losses, save_path)


def evaluate(config, args):
    global firefly_scene
    global global_scene
    global global_params
    global global_key

    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(os.path.join(args.scene_path, "scene.xml"))
    global_params = mi.traverse(global_scene)

    render_path = os.path.join(args.checkpoint_path, "render")
    gt_render_path = os.path.join(render_path, "gt")
    pred_render_path = os.path.join(render_path, "pred")

    if config.save_images:
        try:
            os.mkdir(render_path)
        except:
            printer.Printer.Warning(f"Folder {render_path} does already exist.")

        try:
            os.mkdir(gt_render_path)
        except:
            printer.Printer.Warning(f"Folder {gt_render_path} does already exist.")

        try:
            os.mkdir(pred_render_path)
        except:
            printer.Printer.Warning(f"Folder {pred_render_path} does already exist.")

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
    firefly_scene.eval()
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

    # Init U-Net and params
    UNET_CONFIG = {
        "in_channels": 1,
        "out_channels": 1,
        "features": [32, 64, 128],
    }
    model = UNetWithMultiInput.Model(config=UNET_CONFIG, device=DEVICE).to(DEVICE)

    state_dict = torch.load(os.path.join(args.checkpoint_path, "model_02499.pth.tar"))
    model.load_from_dict(state_dict)
    Laser._rays = state_dict["laser_rays"]
    num_beams = Laser._rays.shape[0]

    rmse = EvaluationCriterion(RSME)
    mae = EvaluationCriterion(MAE)
    metrics = [rmse, mae]

    printer.Printer.Header(f"Beginning evaluation of scene {args.scene_path}")
    printer.Printer.Header(f"Checkpoint: {args.checkpoint_path}")
    printer.Printer.OKB(f"Number of laser beams: {num_beams}")

    for i in tqdm(range(config.eval_iter_final)):
        firefly_scene.randomize()
        pred_depth, gt_depth, _ = inference(model, sigma, tex_size, Laser, K_CAMERA)

        for metric in metrics:
            metric.eval(pred_depth.squeeze(), gt_depth)

        render, pred, gt, tex = get_visualization(
            model, config, tex_size, Laser, K_CAMERA
        )

        if config.visualize:
            visualize(render, pred, gt, tex, waitKey=1)

        if config.save_images:
            save_image(gt, gt_render_path, i)
            save_image(pred, pred_render_path, i)

    for metric in metrics:
        print(metric)


def save_checkpoint(model, optim, Laser, iter, losses, save_path):
    checkpoint = {
        "optimizer": optim.state_dict(),
        "laser_rays": Laser._rays,
        "iteration": iter + 1,
        "losses": losses,
    }
    checkpoint.update(model.get_statedict())
    torch.save(checkpoint, os.path.join(save_path, f"model_{iter:05d}.pth.tar"))


def inference(model, sigma, tex_size, Laser, intrinsic):
    points = Laser.projectRaysToNDC()[:, 0:2]
    texture_init = rasterization.rasterize_points(points, sigma, tex_size)
    texture_init = rasterization.softor(texture_init)
    texture_init = kornia.filters.gaussian_blur2d(
        texture_init.unsqueeze(0).unsqueeze(0), (5, 5), (3, 3)
    ).squeeze()

    hitpoints = cast_laser(Laser.originPerRay(), Laser.rays())

    world_points = Laser.originPerRay() + hitpoints * Laser.rays()
    CAMERA_WORLD = global_params["PerspectiveCamera.to_world"].matrix.torch()[0]
    world_points_hat = transforms.transform_points(
        world_points, CAMERA_WORLD.inverse()
    ).squeeze()
    ndc_points = transforms.transform_points(world_points_hat, intrinsic).squeeze()
    sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)

    filtered_ndc_points = ndc_points
    multi_res_depth = subsampled_point_raster(
        filtered_ndc_points, 4, sigma, sensor_size
    )

    dense_depth = depth.from_camera_non_wrapped(global_scene, 1).torch()

    dense_depth = dense_depth.reshape(sensor_size[1], sensor_size[0], 1).mean(dim=-1)

    model_input = [multi_res.unsqueeze(0) for multi_res in multi_res_depth]
    pred_depth = model(model_input)

    return pred_depth, dense_depth, texture_init


def get_visualization(model, config, tex_size, Laser, camera_intrinsic):
    pred_depth, gt_depth, texture = inference(
        model, config.sigma, tex_size, Laser, camera_intrinsic
    )

    rendered_image = render(texture.unsqueeze(-1), spp=config.spp, seed=0)
    rendering = torch.clamp(rendered_image, 0, 1).detach().cpu().numpy()
    rendering = (rendering * 255).astype(np.uint8)

    gt_depth = 1 - utils.normalize(gt_depth)
    gt_depth = gt_depth.detach().cpu().numpy()
    gt_depth = (gt_depth * 255).astype(np.uint8)
    gt_depth = cv2.applyColorMap(gt_depth, cv2.COLORMAP_INFERNO)

    pred_depth = pred_depth.squeeze()
    pred_depth = 1 - utils.normalize(pred_depth)
    pred_depth = pred_depth.detach().cpu().numpy()
    pred_depth = (pred_depth * 255).astype(np.uint8)
    pred_depth = cv2.applyColorMap(pred_depth, cv2.COLORMAP_INFERNO)

    texture = texture.detach().cpu().numpy()
    texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    texture = (texture * 255).astype(np.uint8)

    return rendering, pred_depth, gt_depth, texture


def save_image(image, save_path, iter):
    imwrite_path = os.path.join(save_path, f"{iter:05d}.png")
    cv2.imwrite(imwrite_path, image)


def save_images(render, pred_depth, gt_depth, texture, save_path, iter):
    render_path = os.path.join(save_path, f"render_{iter:05d}.png")
    pred_path = os.path.join(save_path, f"pred_{iter:05d}.png")
    gt_path = os.path.join(save_path, f"gt_{iter:05d}.png")
    texture_path = os.path.join(save_path, f"tex_{iter:05d}.png")

    cv2.imwrite(render_path, render)
    cv2.imwrite(pred_path, pred_depth)
    cv2.imwrite(gt_path, gt_depth)
    cv2.imwrite(texture_path, texture)


def visualize(render, pred_depth, gt_depth, texture, waitKey=1):

    concat_im = np.hstack([render, texture, pred_depth, gt_depth])

    cv2.imshow("Predicted Depth Map", concat_im)
    cv2.waitKey(waitKey)


def train(
    optim,
    scheduler,
    loss_func,
    model,
    config,
    sigma,
    tex_size,
    Laser,
    camera_intrinsic,
    loss_values,
):
    for i in range(config.gradient_accumulation_steps):
        firefly_scene.randomize()
        pred_depth, dense_depth, _ = inference(
            model, sigma, tex_size, Laser, camera_intrinsic
        )

        loss = loss_func(
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

        loss.backward()

    loss = loss / config.gradient_accumulation_steps
    loss_values.append(loss.item())
    optim.step()
    optim.zero_grad()
    scheduler.step()

    with torch.no_grad():
        Laser.clamp_to_fov(clamp_val=0.99)
        Laser.normalize_rays()

    return loss.item()


def eval(model, config, Laser, sigma, tex_size, camera_intrinsic, metrics, iters):
    for _ in tqdm(range(iters)):
        firefly_scene.randomize()
        pred_depth, dense_depth, _ = inference(
            model, sigma, tex_size, Laser, camera_intrinsic
        )

        for metric in metrics:
            metric.eval(pred_depth.squeeze(), dense_depth)

    for metric in metrics:
        print(metric)


if __name__ == "__main__":
    parser = Args.GlobalArgumentParser()
    args = parser.parse_args()

    config_path = os.path.join(args.scene_path, "config.yml")
    config_args = CAP.ConfigArgsParser(utils.read_config_yaml(config_path), args)
    config_args.printFormatted()
    config = config_args.asNamespace()

    if args.eval:
        evaluate(config, args)
    else:
        main(config, args)
