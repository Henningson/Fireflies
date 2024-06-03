import os
import cv2
import torch
import torch.autograd

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
import Objects.Transformable as Transformable

import Models.GatedUNet as GatedUNet
import Models.UNet as UNet
import Models.UNetToFlame as UNetToFlame
import Models.LP_UNet_Flame as LP_UNet_Flame
import Models.LP_MLP_Flame as LP_MLP_Flame
import Models.PointNet as PointNet
import Models.FLAME_3DCNN as FLAME_3DCNN

import Graphics.rasterization as rasterization
import Metrics.Losses as Losses
import Utils.ConfigArgsParser as CAP
import Utils.Args as Args
import Utils.utils as utils

import trimesh
import pyrender
import matplotlib.pyplot as plt

import Utils.printer as printer

from tqdm import tqdm
from torch import autograd
from pytorch3d.structures import Meshes, Pointclouds, Volumes
from pytorch3d.ops import (
    sample_points_from_meshes,
    points_to_volumes,
    add_pointclouds_to_volumes,
)
from pytorch3d.loss import chamfer_distance


import numpy as np
import pyrender
import torch
import trimesh
import math
import matplotlib.pyplot as plt

from pytorch3d.structures import Meshes, Pointclouds, Volumes
from pytorch3d.ops import (
    sample_points_from_meshes,
    points_to_volumes,
    add_pointclouds_to_volumes,
)
from pytorch3d.loss import chamfer_distance
from Metrics.evaluation import RSME, MAE, EvaluationCriterion


import os
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    Textures,
)

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
)


from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes, save_obj
import time

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


def get_cameras_and_renderer(
    num_views, image_size, sigma, faces_per_pixel, DEVICE
):  # Get a batch of viewing angles.
    elev = torch.linspace(170, 170, num_views)
    azim = torch.linspace(90, 270, num_views)

    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    lights = PointLights(device=DEVICE, location=[[0.0, 0.0, 3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = look_at_view_transform(dist=3.0, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=DEVICE, R=R, T=T)
    # We arbitrarily choose one particular view that will be used to visualize
    # results
    camera = FoVPerspectiveCameras(device=DEVICE, R=R[None, 1, ...], T=T[None, 1, ...])

    # Define the settings for rasterization and shading. Here we set the output
    # image to be of size 128X128. As we are rendering images for visualization
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
    # rasterize_meshes.py for explanations of these parameters.  We also leave
    # bin_size and max_faces_per_bin to their default values of None, which sets
    # their values using heuristics and ensures that the faster coarse-to-fine
    # rasterization method is used.  Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftPhongShader(device=DEVICE, cameras=camera, lights=lights),
    )

    raster_settings_soft = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=False,
    )

    # Differentiable soft renderer using per vertex RGB colors for texture
    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings_soft),
        shader=SoftPhongShader(device=DEVICE, cameras=camera, lights=lights),
    )

    return renderer, renderer_textured, cameras, lights


def visualize(mesh_a, mesh_b, device):
    radian = np.pi / 180.0

    mesh = mesh_a

    for i in range(8):
        yaw_mat = getYawTransform(90 * radian, device)
        pitch_mat = getPitchTransform(0.0, device)
        roll_mat = getRollTransform(45 * radian, device)

        rot = toMat4x4(yaw_mat @ pitch_mat @ roll_mat)

        faces = mesh.faces_padded()
        vertices = mesh.vertices_padded()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 1.0]

        tri_mesh = trimesh.Trimesh(
            vertices, faces[i].detach().cpu().numpy(), vertex_colors=vertex_colors
        )
        tri_mesh.apply_transform(rot.detach().cpu().numpy())
        pyrend_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(pyrend_mesh)

        pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=[1920, 1080])


def pointcloud_to_density(points, num_voxels, DEVICE):
    feature_dim = 1
    point_features = torch.ones(
        points.shape[0], points.shape[1], feature_dim, device=DEVICE
    )
    pointclouds = Pointclouds(points=points, features=point_features)

    volume_features = torch.zeros(
        points.shape[0], feature_dim, num_voxels, num_voxels, num_voxels, device=DEVICE
    )
    volume_densities = torch.zeros(
        points.shape[0], 1, num_voxels, num_voxels, num_voxels, device=DEVICE
    )
    initial_volumes = Volumes(
        features=volume_features, densities=volume_densities, voxel_size=2 / num_voxels
    )

    updated_volumes = add_pointclouds_to_volumes(
        pointclouds=pointclouds,
        initial_volumes=initial_volumes,
        mode="trilinear",
    )

    return updated_volumes.densities()


def normalizeDensities(volume: torch.tensor) -> torch.tensor:
    _min = volume.amin(dim=(2, 3, 4), keepdims=True)
    _max = volume.amax(dim=(2, 3, 4), keepdims=True)
    _norm = (volume - _min) / (_max - _min)
    return _norm


def toUnitCube(points: torch.tensor) -> torch.tensor:
    _min = points.min(dim=1, keepdims=True)[0]
    _max = points.max(dim=1, keepdims=True)[0]
    _norm = (points - _min) / (_max - _min)
    return _norm


# Transform [0, 1] to [-1, 1]
def toMinusOneOne(points: torch.tensor) -> torch.tensor:
    return (points - 0.5) * 2.0


def show_rendered_mesh(
    flamelayer,
    renderer,
    cameras,
    lights,
    num_views,
    textures,
    device,
    batch_size=1,
    num_shape_params=100,
    num_expression_params=50,
):

    pose_params = torch.zeros(batch_size, 6).to(device)

    # Creating a batch of mean shapes
    shape_params = ((torch.rand(batch_size, num_shape_params) - 0.5) * 2.0).to(
        device
    )  # Shape to [-1, 1]
    shape_params *= 2.0
    # shape_params += 0.0

    # Cerating a batch of neutral expressions
    expression_params = torch.zeros(batch_size, num_expression_params, device=device)
    expression_params *= 0.0

    vertices, _ = flamelayer(shape_params, expression_params, pose_params)
    vertices = toMinusOneOne(toUnitCube(vertices))

    mesh = Meshes(
        vertices,
        torch.from_numpy(flamelayer.faces.astype(int))
        .to(device)
        .unsqueeze(0)
        .repeat(vertices.shape[0], 1, 1),
        textures,
    )
    # mesh = mesh.extend(num_views)

    target_images = renderer(mesh, cameras=cameras, lights=lights)
    target_rgb = [
        target_images[i, ..., :3].detach().cpu().numpy() for i in range(num_views)
    ]

    image_grid(target_rgb, rows=2, cols=4)
    plt.show()


def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


# Show a visualization comparing the rendered predicted mesh to the ground truth
# mesh
def visualize_prediction(predicted_images, target_image, title="", silhouette=False):
    inds = 3 if silhouette else range(3)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")


def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l["values"], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")


def pointcloud_to_volume(points: torch.tensor, num_voxels: int = 32):
    sampled_points = toMinusOneOne(toUnitCube(points))
    sample_volume = pointcloud_to_density(sampled_points, num_voxels, points.device)
    return normalizeDensities(sample_volume)


# Return shape params in range (-stddev, stddev)
def get_rand_params(batch_size: int, num_params: int, stddev: float = 2.0):
    return ((torch.rand(batch_size, num_params) - 0.5) * 2.0) * stddev


def main(config_args, args):
    global global_scene
    global global_params
    global global_key

    config = config_args.asNamespace()

    config.batch_size = config.num_views
    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(
        os.path.join(args.scene_path, "scene.xml"), parallel=False
    )
    global_params = mi.traverse(global_scene)

    save_base = os.path.join(args.scene_path, "optim")
    save_path = os.path.join(
        save_base,
        f'{time.strftime("%Y-%m-%d-%H:%M:%S")}_{config.pattern_initialization}_{config.iterations}_{config.n_beams}_{config.lr_laser}',
    )
    save_render_path = os.path.join(save_path, "render")
    save_tex_path = os.path.join(save_path, "tex")

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

    if config.save_images:
        try:
            os.mkdir(save_tex_path)
        except:
            printer.Printer.Warning(
                f"Folder {save_tex_path} does already exist. Please restart."
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
    firefly_scene.randomize()

    shapemodel_key = "PLYMesh"

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

    projector_sensor = global_scene.sensors()[1]
    near_clip = global_scene.sensors()[1].near_clip()
    far_clip = global_scene.sensors()[1].far_clip()
    laser_fov = float(global_params["PerspectiveCamera_1.x_fov"][0])
    near_clip = global_scene.sensors()[1].near_clip()

    radians = math.pi / 180.0

    LASER_K = mi.perspective_projection(
        projector_sensor.film().size(),
        projector_sensor.film().crop_size(),
        projector_sensor.film().crop_offset(),
        laser_fov,
        near_clip,
        far_clip,
    ).matrix.torch()[0]

    Laser = LaserEstimation.initialize_laser(
        global_scene,
        global_params,
        firefly_scene,
        config,
        config.pattern_initialization,
        DEVICE,
    )

    model = UNetToFlame.Model().to(DEVICE)
    model.train()

    loss_funcs = Losses.Handler(
        [
            # [Losses.VGGPerceptual().to(DEVICE), 0.0],
            [torch.nn.MSELoss().to(DEVICE), 1.0],
            # [torch.nn.L1Loss().to(DEVICE),  1.0]
        ]
    )

    gt_renderer, diff_renderer, cameras, lights = get_cameras_and_renderer(
        config.num_views, 128, sigma, 10, DEVICE
    )

    Laser._rays.requires_grad = True
    sigma.requires_grad = True
    printer.Printer.OKG(f"Number of Rays: {Laser._rays.shape[0]}")

    optim = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": config.lr_model},
            {"params": Laser._rays, "lr": config.lr_laser},
            {"params": sigma, "lr": config.lr_sigma},
        ]
    )

    losses = {
        "rgb": {"weight": 1.0, "values": []},
        "silhouette": {"weight": 1.0, "values": []},
        "chamf": {"weight": 0.5, "values": []},
        "shape_params": {"weight": 0.1, "values": []},
        "scene_seg": {"weight": 0.2, "values": []},
        "vicinity": {"weight": config.vicinity_penalty_lambda, "values": []},
    }

    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optim,
        power=0.99,
        total_iters=config.iterations // config.gradient_accumulation_steps,
    )
    shapemodel = firefly_scene.meshes[shapemodel_key]
    # Hacky magic number here. But we want a better L2 Loss
    vertex_color = torch.randn(1, 5023, 3).to(DEVICE)
    textures = Textures(verts_rgb=vertex_color)

    amount_of_hits = torch.zeros(Laser._rays.shape[0], device=DEVICE, dtype=torch.int32)
    max_hit_dist = 0.0
    min_hit_dist = 1000000000000000000000.0

    min_dist = 0.926
    max_dist = 2.366

    for i in (progress_bar := tqdm(range(config.iterations))):
        firefly_scene.randomize()
        optim.zero_grad()

        with torch.no_grad():
            vertices, _ = shapemodel.getVertexData()
            vertices = vertices.unsqueeze(0)
            mesh = Meshes(
                toMinusOneOne(toUnitCube(vertices)),
                shapemodel._shape_layer.faces_tensor.unsqueeze(0),
                textures,
            )

            target_images = gt_renderer(
                mesh.extend(config.num_views), cameras=cameras, lights=lights
            )
            target_rgb = target_images[..., :3]
            target_silhouette = target_images[..., 3]

        # Initialize optimizer
        segmentation = depth.get_segmentation_from_camera(global_scene).float()

        if config.visualize:
            cv2.imshow(
                "Seg", segmentation.detach().cpu().numpy().astype(np.uint8) * 255
            )
            cv2.waitKey(1)

        sensor_size = torch.tensor(
            global_scene.sensors()[0].film().size(), device=DEVICE
        )
        hit_d = cast_laser(Laser.originPerRay(), Laser.rays())

        if max_hit_dist < hit_d.max():
            max_hit_dist = hit_d.max()

        if hit_d.min() < min_hit_dist:
            min_hit_dist = hit_d.min()

        world_points = Laser.originPerRay() + Laser.rays() * hit_d

        CAMERA_WORLD = global_params["PerspectiveCamera.to_world"].matrix.torch()[0]
        world_points_hat = transforms.transform_points(
            world_points, CAMERA_WORLD.inverse()
        ).squeeze()
        ndc_points = transforms.transform_points(world_points_hat, K_CAMERA).squeeze()

        visualization_points = ndc_points.clone()
        # We should remove points, that do not fall into the object itself here.
        image_space_points = ndc_points[:, 0:2] * sensor_size
        quantized_indices = image_space_points.floor().int()
        object_hits = (
            segmentation[quantized_indices[:, 1], quantized_indices[:, 0]]
            .nonzero()
            .squeeze()
        )

        object_nonhits = (
            (segmentation[quantized_indices[:, 1], quantized_indices[:, 0]] == 0)
            .nonzero()
            .squeeze()
        )
        object_misses_ndc = ndc_points[object_nonhits]

        object_hits_ndc = ndc_points[object_hits]

        hits_raster = rasterization.rasterize_depth(
            object_hits_ndc[:, 0:2], object_hits_ndc[:, 2:], config.sigma, sensor_size
        )
        hits_raster = rasterization.softor(hits_raster)

        # hit_textures = rasterization.subsampled_point_raster(
        #    object_hits_ndc, 3, config.sigma, sensor_size
        # )
        # hits_tex = rasterization.softor(hits_raster)

        misses_raster = rasterization.rasterize_points(
            object_misses_ndc[:, 0:2], config.sigma, sensor_size
        )
        misses_tex = rasterization.softor(misses_raster)

        estimated_shape_params = model(hits_raster.unsqueeze(0).unsqueeze(0))
        pred_vertices, _ = shapemodel._shape_layer(
            estimated_shape_params,
            shapemodel.expressionParams(),
            shapemodel.poseParams(),
        )

        pred_vertices = toMinusOneOne(toUnitCube(pred_vertices))
        pred_mesh = Meshes(
            pred_vertices, shapemodel._shape_layer.faces_tensor.unsqueeze(0), textures
        ).extend(config.num_views)

        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=DEVICE) for k in losses}
        images_predicted = diff_renderer(pred_mesh, cameras=cameras, lights=lights)

        loss_silhouette = loss_funcs(images_predicted[..., 3], target_silhouette)
        loss["silhouette"] += loss_silhouette / config.num_views

        loss_rgb = loss_funcs(images_predicted[..., :3], target_rgb)
        loss["rgb"] += loss_rgb / config.num_views

        loss_seg = loss_funcs(segmentation, misses_tex)
        loss["scene_seg"] += loss_seg

        loss_verts = chamfer_distance(mesh.verts_padded(), pred_vertices)[0]
        loss["chamf"] += loss_verts

        exponential_shape_decay = torch.arange(0, 20, 1, device=DEVICE)
        exponential_shape_decay = torch.exp(-0.5 * exponential_shape_decay)
        loss_shape = torch.nn.L1Loss()(
            estimated_shape_params[:, :20] * exponential_shape_decay,
            shapemodel.shapeParams()[:, :20] * exponential_shape_decay,
        )
        loss["shape_params"] += loss_shape

        laser_tex = Laser.generateTexture(config.sigma, sensor_size)
        softor_laser_tex = rasterization.softor(laser_tex)
        sum_laser_tex = laser_tex.sum(dim=0)

        loss_vicinity = torch.nn.L1Loss()(softor_laser_tex, sum_laser_tex)
        loss["vicinity"] += loss_vicinity

        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=DEVICE)
        sum_loss.requires_grad = True

        for k, l in loss.items():
            sum_loss = sum_loss + l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))

        # Print the losses
        progress_bar.set_description("total_loss = %.6f" % sum_loss)

        sum_loss = sum_loss / config.gradient_accumulation_steps
        sum_loss.backward()

        # Gradient Accumulation
        if i > 0 and i % config.gradient_accumulation_steps == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad()

        with torch.no_grad():
            Laser.clamp_to_fov()
            Laser.normalize_rays()

            amount_of_hits[object_hits] += 1

            sensor_size = torch.tensor(
                global_scene.sensors()[0].film().size(), device=DEVICE
            )
            laser_texture = Laser.generateTexture(math.sqrt(config.sigma), sensor_size)
            laser_texture = rasterization.softor(laser_texture)
            laser_texture = laser_texture.detach().cpu().numpy() * 255
            laser_texture = laser_texture.astype(np.uint8)

            if config.visualize and i % config.gradient_accumulation_steps == 0:
                cv2.imshow("Points", laser_texture)
                cv2.waitKey(1)

            if config.save_images and i % config.gradient_accumulation_steps == 0:
                cv2.imwrite(
                    os.path.join(
                        save_tex_path,
                        f"{i//config.gradient_accumulation_steps:05d}.png",
                    ),
                    laser_texture,
                )

            if config.save_images and i == config.iterations - 1:
                object_hits = (
                    segmentation[quantized_indices[:, 1], quantized_indices[:, 0]]
                    .nonzero()
                    .squeeze()
                )
                points = Laser.projectRaysToNDC()[:, 0:2]
                points = points[amount_of_hits > 0]
                print(f"FINAL REMAINING POINTS: {points.shape[0]}")
                raster_points = rasterization.rasterize_points(
                    points, math.sqrt(sigma), sensor_size
                )
                raster_points = rasterization.softor(raster_points)
                raster_points = raster_points.detach().cpu().numpy() * 255
                raster_points = raster_points.astype(np.uint8)
                cv2.imwrite(
                    os.path.join(
                        save_path,
                        f"final_postprocess_texture.png",
                    ),
                    raster_points,
                )
                print(f"MAX DIST: {max_hit_dist}")
                print(f"MIN DIST: {min_hit_dist}")

    save_checkpoint(model, optim, Laser, i + 1, losses, save_path)

    # plot_losses(losses)
    # plt.show()


def save_checkpoint(model, optim, Laser, iter, losses, save_path):
    checkpoint = {
        "optimizer": optim.state_dict(),
        "laser_rays": Laser._rays,
        "iteration": iter + 1,
        "losses": losses,
    }
    checkpoint.update(model.get_statedict())
    torch.save(checkpoint, os.path.join(save_path, f"model_{iter:05d}.pth.tar"))


def evaluate(config_args, args):
    global global_scene
    global global_params
    global global_key

    config = config_args.asNamespace()

    config.batch_size = config.num_views
    sigma = torch.tensor([config.sigma], device=DEVICE)
    global_scene = mi.load_file(
        os.path.join(args.scene_path, "scene.xml"), parallel=False
    )
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

    shapemodel_key = "PLYMesh"

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
        global_scene, global_params, firefly_scene, config, "GRID", DEVICE
    )

    model = UNetToFlame.Model().to(DEVICE)

    state_dict = torch.load(os.path.join(args.checkpoint_path, "model_02500.pth.tar"))
    model.load_from_dict(state_dict)
    Laser._rays = state_dict["laser_rays"]
    num_beams = Laser._rays.shape[0]

    model.eval()

    gt_renderer, diff_renderer, cameras, lights = get_cameras_and_renderer(
        config.num_views, 128, sigma, 10, DEVICE
    )

    shapemodel = firefly_scene.meshes[shapemodel_key]
    # Hacky magic number here. But we want a better L2 Loss
    vertex_color = torch.ones(1, 5023, 3).to(DEVICE) * 0.5
    textures = Textures(verts_rgb=vertex_color)

    sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)

    rmse = EvaluationCriterion(RSME)
    mae = EvaluationCriterion(MAE)
    metrics = [rmse, mae]

    for i in (progress_bar := tqdm(range(config.eval_iter_final))):
        with torch.no_grad():
            vertices, _ = shapemodel.getVertexData()
            vertices = vertices.unsqueeze(0)
            mesh = Meshes(
                toMinusOneOne(toUnitCube(vertices)),
                shapemodel._shape_layer.faces_tensor.unsqueeze(0),
                textures,
            )

        # Initialize optimizer
        segmentation = depth.get_segmentation_from_camera(global_scene).float()
        firefly_scene.randomize()

        hit_d = cast_laser(Laser.originPerRay(), Laser.rays())
        world_points = Laser.originPerRay() + Laser.rays() * hit_d

        CAMERA_WORLD = global_params["PerspectiveCamera.to_world"].matrix.torch()[0]
        world_points_hat = transforms.transform_points(
            world_points, CAMERA_WORLD.inverse()
        ).squeeze()
        ndc_points = transforms.transform_points(world_points_hat, K_CAMERA).squeeze()

        image_space_points = ndc_points[:, 0:2] * sensor_size
        quantized_indices = image_space_points.floor().int()
        object_hits = (
            segmentation[quantized_indices[:, 1], quantized_indices[:, 0]]
            .nonzero()
            .squeeze()
        )

        object_nonhits = (
            (segmentation[quantized_indices[:, 1], quantized_indices[:, 0]] == 0)
            .nonzero()
            .squeeze()
        )
        object_hits_ndc = ndc_points[object_hits]

        hits_raster = rasterization.rasterize_depth(
            object_hits_ndc[:, 0:2], object_hits_ndc[:, 2:], config.sigma, sensor_size
        )
        hits_raster = rasterization.softor(hits_raster)

        estimated_shape_params = model(hits_raster.unsqueeze(0).unsqueeze(0))
        pred_vertices, _ = shapemodel._shape_layer(
            estimated_shape_params,
            shapemodel.expressionParams(),
            shapemodel.poseParams(),
        )

        pred_vertices = toMinusOneOne(toUnitCube(pred_vertices))

        for metric in metrics:
            metric.eval(estimated_shape_params, shapemodel.shapeParams())

    for metric in metrics:
        print(metric)
        metric.save(args.checkpoint_path, 2500)


if __name__ == "__main__":
    parser = Args.GlobalArgumentParser()
    args = parser.parse_args()

    if args.checkpoint_path is not None and args.eval_dir:
        temp = args.checkpoint_path
        BASES = [
            "GRID",
            "GRID_LR",
            "POISSON",
            "POISSON_LR",
            "RANDOM",
            "RANDOM_LR",
            "SMARTY",
            "SMARTY_LR",
        ]

        for base in BASES:
            args.checkpoint_path = os.path.join(temp, base)
            checkpoint_base = args.checkpoint_path
            folders = [
                folder
                for folder in os.listdir(args.checkpoint_path)
                if os.path.isdir(os.path.join(checkpoint_base, folder))
            ]
            folders.sort()

            for folder in folders:
                args.checkpoint_path = os.path.join(checkpoint_base, folder)
                config_path = os.path.join(args.checkpoint_path, "config.yml")
                config_args = CAP.ConfigArgsParser(
                    utils.read_config_yaml(config_path), args
                )
                # config_args.printFormatted()
                printer.Printer.OKC(folder)
                evaluate(config_args, args)
                printer.Printer.OKG("Done evaluating")

    exit()

    config_path = os.path.join(args.scene_path, "config.yml")
    config_args = CAP.ConfigArgsParser(utils.read_config_yaml(config_path), args)
    config_args.printFormatted()
    if args.eval:
        evaluate(config_args, args)
    else:
        main(config_args, args)