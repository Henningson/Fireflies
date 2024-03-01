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
import Models.UNetWithMultiInput as UNetWithMultiInput
import Graphics.rasterization as rasterization
import Metrics.Losses as Losses
import Utils.ConfigArgsParser as CAP
import Utils.Args as Args
import Utils.utils as utils
import Utils.printer as printer
from Metrics.evaluation import RSME, MAE, EvaluationCriterion

from tqdm import tqdm

import kornia

global_scene = None
global_params = None
global_key = None


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


def main(scene_path: str = None, checkpoint_path: str = None):
    global global_scene
    global global_params
    global global_key

    parser = Args.GlobalArgumentParser()
    args = parser.parse_args()

    if args.scene_path is None and scene_path is None:
        printer.Printer.Error("Please specify a path to the scene")
        exit()

    if args.checkpoint_path is None and checkpoint_path is None:
        printer.Printer.Error("Please specify a path to the checkpoint folder.")
        exit()

    scene_path = args.scene_path if args.scene_path else scene_path
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else checkpoint_path

    config_path = os.path.join(checkpoint_path, "config.yml")
    config_args = CAP.ConfigArgsParser(utils.read_config_yaml(config_path), args)
    config_args.printFormatted()
    config = config_args.asNamespace()

    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(os.path.join(scene_path, "scene.xml"))
    global_params = mi.traverse(global_scene)

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
        scene_path,
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

    # Init U-Net and params
    UNET_CONFIG = {
        "in_channels": 1,
        "out_channels": 1,
        "features": [32, 64, 128],
    }
    model = UNetWithMultiInput.Model(config=UNET_CONFIG, device=DEVICE).to(DEVICE)

    state_dict = torch.load(os.path.join(checkpoint_path, "model.pth.tar"))
    model.load_from_dict(state_dict)
    Laser._rays = state_dict["laser_rays"]

    rmse = EvaluationCriterion(RSME)
    mae = EvaluationCriterion(MAE)

    printer.Printer.Header(f"Beginning evaluation of scene {scene_path}")
    printer.Printer.Header(f"Checkpoint: {checkpoint_path}")
    printer.Printer.OKB(f"Number of laser beams: {Laser._rays.shape[0]}")

    model.eval()
    for _ in tqdm(range(config.eval_steps)):
        firefly_scene.randomize()
        # segmentation = depth.get_segmentation_from_camera(global_scene).float()

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
        ndc_points = transforms.transform_points(world_points_hat, K_CAMERA).squeeze()
        sensor_size = torch.tensor(
            global_scene.sensors()[0].film().size(), device=DEVICE
        )

        # )
        # filtered_ndc_points = ndc_points[object_hits]
        filtered_ndc_points = ndc_points

        # sparse_depth = rasterization.rasterize_depth(filtered_ndc_points[:, 0:2], filtered_ndc_points[:, 2:3], config.sigma, sensor_size)
        multi_res_depth = subsampled_point_raster(
            filtered_ndc_points, 4, config.sigma, sensor_size
        )

        dense_depth = depth.from_camera_non_wrapped(global_scene, config.spp).torch()

        dense_depth = dense_depth.reshape(
            sensor_size[1], sensor_size[0], config.spp
        ).mean(dim=-1)

        model_input = [multi_res.unsqueeze(0) for multi_res in multi_res_depth]
        pred_depth = model(model_input)

        rmse.eval(pred_depth.squeeze(), dense_depth)
        mae.eval(pred_depth.squeeze(), dense_depth)

    print(rmse)
    print(mae)


if __name__ == "__main__":

    main()
    exit()

    scene_path = "scenes/Vocalfold/"

    eval_pathes = [
        "2024-02-29-18:56:25_POISSON_15000_10",
        "2024-02-29-19:08:59_POISSON_15000_20",
        "2024-02-29-19:20:45_POISSON_15000_30",
        "2024-02-29-19:31:42_POISSON_15000_40",
        "2024-02-29-19:42:25_POISSON_15000_50",
        "2024-02-29-19:53:15_POISSON_15000_60",
        "2024-02-29-20:04:03_POISSON_15000_70",
        "2024-02-29-20:15:01_POISSON_15000_80",
        "2024-02-29-20:26:04_POISSON_15000_90",
        "2024-02-29-20:37:05_POISSON_15000_100",
        "2024-02-29-20:48:13_POISSON_15000_125",
        "2024-02-29-20:59:24_POISSON_15000_150",
        "2024-02-29-21:10:44_POISSON_15000_175",
        "2024-02-29-21:22:28_POISSON_15000_200",
    ]

    for path in eval_pathes:
        checkpoint_path = os.path.join(scene_path, "optim", path)
        main(scene_path, checkpoint_path)
