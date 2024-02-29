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
from Metrics.evaluation import EvaluationCriterion, RSME, MAE
import torchmetrics.regression

global_scene = None
global_params = None
global_key = None
DEBUG = False


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


def eval():
    global global_scene
    global global_params
    global global_key

    parser = Args.GlobalArgumentParser()
    args = parser.parse_args()

    config_path = os.path.join(args.checkpoint_path, "config.yml")
    config = CAP.ConfigArgsParser(utils.read_config_yaml(config_path), args)
    config.printFormatted()
    config = config.asNamespace()

    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(os.path.join(args.scene_path, "scene.xml"))
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

    Laser = LaserEstimation.initialize_laser(
        global_scene,
        global_params,
        firefly_scene,
        config,
        "GRID",
        DEVICE,
    )

    # Init U-Net and params
    KNN_FEATURES = config.knn_features
    MODEL_CONFIG = {"in_channels": KNN_FEATURES, "features": [32, 64, 128, 256]}
    model = RotationEncoder.PeakFinder(config=MODEL_CONFIG).to(DEVICE)

    state_dict = torch.load(os.path.join(args.checkpoint_path, "model.pth.tar"))
    model.load_from_dict(state_dict)

    Laser._rays = state_dict["laser_rays"]
    ray_mask = state_dict["ray_mask"]
    Laser._rays = Laser._rays[ray_mask.nonzero().flatten()]

    model.eval()

    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, total_iters = config.iterations, power=0.99)
    sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)

    rmse = EvaluationCriterion(RSME)
    mae = EvaluationCriterion(MAE)

    printer.Printer.Header(f"Beginning evaluation of scene {args.scene_path}")
    printer.Printer.OKB(f"Number of laser beams: {Laser._rays.shape[0]}")
    for i in (progress_bar := tqdm(range(config.eval_steps))):
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
        ).unsqueeze(0)

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

        point_coordinates = ndc_points[:, 0:2]
        point_coordinates = point_coordinates * sensor_size

        ndc_depth = ndc_points[:, 2:3]
        nearest_point_in_z = point_coordinates[ndc_depth.argmin()]

        _, idx, _ = pytorch3d.ops.knn_points(
            nearest_point_in_z.unsqueeze(0).unsqueeze(0),
            point_coordinates.unsqueeze(0),
            K=KNN_FEATURES,
        )

        neighbours_at_peak = torch.concat(
            [point_coordinates[idx.flatten()], ndc_depth[idx.flatten()]], dim=1
        )

        pred_peak = model(neighbours_at_peak.flatten().unsqueeze(0))

        rmse.eval(pred_peak, mean_coordinate)
        mae.eval(pred_peak, mean_coordinate)

    print(rmse)
    print(mae)


if __name__ == "__main__":
    eval()
