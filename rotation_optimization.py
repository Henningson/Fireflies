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
import Models.RotationEncoder as RotationEncoder
import Models.UNet as UNet
import Models.UNetWithMultiInput as UNetWithMultiInput
import Graphics.rasterization as rasterization
import Metrics.Losses as Losses
import Utils.ConfigArgsParser as CAP
import Utils.Args as Args
import Utils.utils as utils
import Utils.bridson
import time
import Utils.printer as printer
from tqdm import tqdm

import pytorch3d.ops
import imageio
import matplotlib.pyplot as plt
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
        rastered_depth = (rastered_depth - rastered_depth.min()) / (
            rastered_depth.max() - rastered_depth.min()
        )
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
    config = CAP.ConfigArgsParser(
        utils.read_config_yaml(os.path.join(args.scene_path, "config.yml")), args
    )
    config.printFormatted()
    config = config.asNamespace()

    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(os.path.join(args.scene_path, "scene.xml"))
    global_params = mi.traverse(global_scene)

    save_path = os.path.join(
        args.scene_path,
        f'{time.strftime("%Y-%m-%d-%H:%M:%S")}_{config.pattern_initialization}_{config.iterations}',
    )

    try:
        os.mkdir(save_path)
    except:
        printer.Printer.Warning(
            f"Folder {save_path} does already exist. Please restart."
        )
        exit()
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

    K_CAMERA = mi.perspective_projection(
        camera_sensor.film().size(),
        camera_sensor.film().crop_size(),
        camera_sensor.film().crop_offset(),
        camera_x_fov,
        camera_near_clip,
        camera_far_clip,
    ).matrix.torch()[0]
    # K_PROJECTOR = mi.perspective_projection(projector_sensor.film().size(), projector_sensor.film().crop_size(), projector_sensor.film().crop_offset(), projector_x_fov, projector_near_clip, projector_far_clip).matrix.torch()[0]

    Laser = LaserEstimation.initialize_laser(
        global_scene,
        global_params,
        firefly_scene,
        config,
        config.pattern_initialization,
        DEVICE,
    )

    KNN_FEATURES = config.knn_features

    # Init U-Net and params
    MODEL_CONFIG = {"in_channels": KNN_FEATURES, "features": [32, 64, 128, 256]}
    model = RotationEncoder.PeakFinder(config=MODEL_CONFIG).to(DEVICE)
    model.train()

    loss_funcs = Losses.Handler([[torch.nn.MSELoss().to(DEVICE), 1.0]])
    loss_values = []

    Laser._rays.requires_grad = True
    sigma.requires_grad = True

    optim = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": config.lr_model},
            {"params": Laser._rays, "lr": 0.0},
            # {'params': sigma,               'lr': config.lr_sigma * 1000000.0}
        ]
    )

    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, total_iters = config.iterations, power=0.99)
    sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)
    tex_size = torch.tensor(global_scene.sensors()[1].film().size(), device=DEVICE)

    zero_gradient_counter = torch.zeros(
        Laser._rays.shape[0], dtype=torch.int32, device=DEVICE
    )
    zero_grad_threshold = config.zero_grad_threshold
    ray_mask = torch.ones(Laser._rays.shape[0], dtype=torch.int32, device=DEVICE)

    for i in (progress_bar := tqdm(range(config.iterations))):
        if i == config.warmup_iterations:
            optim.param_groups[1]["lr"] = config.lr_laser

        firefly_scene.randomize()

        # Generate depth image and look for peak of our Gaussian Blob
        depth_image = (
            depth.from_camera(global_scene, spp=1)
            .torch()
            .reshape(sensor_size[0], sensor_size[1])
        )
        depth_image = utils.normalize(depth_image)
        cv2.imshow("Depth", depth_image.detach().cpu().numpy())

        # Extract the 2D coordinate of the mean
        mean_coordinate = (
            (depth_image == torch.min(depth_image)).nonzero()[0].float().flip(dims=[0])
        )

        points = Laser.projectRaysToNDC()[:, 0:2]
        points = points[ray_mask.nonzero().flatten()]
        texture_init = rasterization.rasterize_points(points, sigma, tex_size)
        texture_init = rasterization.softor(texture_init)

        hitpoints = cast_laser(Laser.originPerRay(), Laser.rays())

        world_points = Laser.originPerRay() + hitpoints * Laser.rays()
        world_points = world_points[ray_mask.nonzero().flatten()]
        CAMERA_WORLD = global_params["PerspectiveCamera.to_world"].matrix.torch()[0]

        world_points_hat = transforms.transform_points(
            world_points, CAMERA_WORLD.inverse()
        ).squeeze()
        ndc_points = transforms.transform_points(world_points_hat, K_CAMERA).squeeze()
        sensor_size = torch.tensor(
            global_scene.sensors()[0].film().size(), device=DEVICE
        )

        point_coordinates = ndc_points[:, 0:2]
        point_coordinates = point_coordinates * sensor_size

        ndc_depth = ndc_points[:, 2:3]
        nearest_point_in_z = point_coordinates[ndc_depth.argmin()]

        dists, idx, _ = pytorch3d.ops.knn_points(
            nearest_point_in_z.unsqueeze(0).unsqueeze(0),
            point_coordinates.unsqueeze(0),
            K=KNN_FEATURES,
        )

        neighbours_at_peak = torch.concat(
            [point_coordinates[idx.flatten()], ndc_depth[idx.flatten()]], dim=1
        )

        pred_peak = model(neighbours_at_peak.flatten().unsqueeze(0))

        per_point_depth = rasterization.rasterize_depth(
            ndc_points[:, 0:2], ndc_depth, sigma**2, sensor_size
        )
        per_point_depth = rasterization.softor(per_point_depth)
        loss = loss_funcs(pred_peak, mean_coordinate)

        # Projected points should also not overlap
        rasterized_points = rasterization.rasterize_points(
            ndc_points[:, 0:2], config.sigma * 3, sensor_size
        )
        loss += (
            torch.nn.L1Loss()(
                rasterization.softor(rasterized_points), rasterized_points.sum(dim=0)
            )
            * config.vicinity_penalty_lambda
        )

        loss = loss / config.gradient_accumulation_steps
        """
        # Make sure that epipolar lines do not overlap too much
        lines = Laser.render_epipolar_lines(sigma, tex_size)
        epc_regularization = torch.nn.MSELoss()(rasterization.softor(lines), lines.sum(dim=0))
        loss += epc_regularization * config.epipolar_constraint_lambda

        # Projected points should also not overlap
        rasterized_points = rasterization.rasterize_points(ndc_points[:, 0:2], config.sigma, sensor_size)
        loss += torch.nn.MSELoss()(rasterization.softor(rasterized_points), rasterized_points.sum(dim=0)) * 0.0005

        # Lets go for segmentation to projection similarity here
        loss += torch.nn.MSELoss()(rasterization.softor(rasterized_points), segmentation) * config.perspective_segmentation_similarity_lambda
        """

        loss_values.append(loss.item())
        loss.backward()

        if i > 0 and i % config.gradient_accumulation_steps == 0:

            # Check which Laser rays do not have any gradient.
            has_zero_grad = (Laser._rays.grad == 0).any(dim=1) * 1

            # Increase count of zero grad rays by 1 and set everything else back to 0
            zero_gradient_counter = (
                zero_gradient_counter + has_zero_grad
            ) * has_zero_grad

            ray_mask = ray_mask * ~(zero_gradient_counter >= zero_grad_threshold)

            optim.step()
            optim.zero_grad()

            progress_bar.set_description(
                "Loss: {0:.4f}, Sigma: {1:.4f}".format(
                    loss.item(), sigma.detach().cpu().numpy()[0]
                )
            )

            if config.visualize:
                with torch.no_grad():
                    render_im = render(texture_init.unsqueeze(-1))
                    render_im = (
                        torch.clamp(render_im, 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
                    )
                    render_circle = cv2.circle(
                        render_im.clone(),
                        pred_peak.detach().cpu().numpy().flatten().astype(np.uint8),
                        3,
                        (0, 255, 0),
                        -1,
                    )
                    render_circle = cv2.circle(
                        render_circle,
                        mean_coordinate.detach()
                        .cpu()
                        .numpy()
                        .flatten()
                        .astype(np.uint8),
                        3,
                        (0, 0, 255),
                        -1,
                    )
                    render_circle = cv2.circle(
                        render_circle,
                        point_coordinates[idx.flatten()][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .flatten()
                        .astype(np.uint8),
                        3,
                        (255, 255, 0),
                        -1,
                    )
                    render_circle = cv2.circle(
                        render_circle,
                        point_coordinates[idx.flatten()][1]
                        .detach()
                        .cpu()
                        .numpy()
                        .flatten()
                        .astype(np.uint8),
                        3,
                        (255, 255, 0),
                        -1,
                    )

                    cv2.imshow("Render", render_circle)
                    # cv2.waitKey(1)

                    Laser.clamp_to_fov(clamp_val=0.99)
                    Laser.normalize_rays()
                    sparse_depth = torch.clamp(per_point_depth, 0, 1)
                    sparse_depth = sparse_depth.detach().cpu().numpy() * 255
                    sparse_depth = sparse_depth.astype(np.uint8)
                    print(point_coordinates[idx.flatten()])
                    cv2.imshow("Points", sparse_depth)
                    cv2.waitKey(1)

                    if config.save_images:
                        optim_path = os.path.join(save_path, optim_path)
                        os.mkdir(optim_path)
                        image_string = "{:05d}.png".format(
                            i // config.gradient_accumulation_steps
                        )
                        cv2.imwrite(
                            os.path.join(optim_path, image_string),
                            (render_im * 255).astype(np.uint8),
                        )

    printer.Printer.OKG("Optimization done. Saving.")
    checkpoint = {"optimizer": optim.state_dict()}
    checkpoint.update(model.get_statedict())

    torch.save(checkpoint, os.path.join(save_path, "model.pth.tar"))
    Laser.save(os.path.join(save_path, "laser.yml"))

    print("Finished everything.")

    plt.plot(loss_values)
    plt.show()


if __name__ == "__main__":
    main()
