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
import Utils.math as math
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
from pytorch3d.ops import sample_points_from_meshes, points_to_volumes, add_pointclouds_to_volumes
from pytorch3d.loss import chamfer_distance


import numpy as np
import pyrender
import torch
import trimesh
import math
import matplotlib.pyplot as plt

from pytorch3d.structures import Meshes, Pointclouds, Volumes
from pytorch3d.ops import sample_points_from_meshes, points_to_volumes, add_pointclouds_to_volumes
from pytorch3d.loss import chamfer_distance


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
    Textures
)

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
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


def get_cameras_and_renderer(num_views, image_size, sigma, faces_per_pixel, DEVICE):    # Get a batch of viewing angles. 
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
    camera = FoVPerspectiveCameras(device=DEVICE, R=R[None, 1, ...], 
                                    T=T[None, 1, ...]) 


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
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=DEVICE, 
            cameras=camera,
            lights=lights
        )
    )


    raster_settings_soft = RasterizationSettings(
        image_size=image_size, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=faces_per_pixel, 
        perspective_correct=False, 
    )

    # Differentiable soft renderer using per vertex RGB colors for texture
    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(device=DEVICE, 
            cameras=camera,
            lights=lights)
    )

    return renderer, renderer_textured, cameras, lights


def visualize(mesh_a, mesh_b, device):    
    radian = np.pi / 180.0

    mesh = mesh_a

    for i in range(8):
        yaw_mat   = getYawTransform(90*radian, device)
        pitch_mat = getPitchTransform(0.0, device)
        roll_mat  = getRollTransform(45*radian, device)

        rot = toMat4x4(yaw_mat @ pitch_mat @ roll_mat)

        faces = mesh.faces_padded()
        vertices = mesh.vertices_padded()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 1.0]

        tri_mesh = trimesh.Trimesh(vertices, faces[i].detach().cpu().numpy(), vertex_colors=vertex_colors)
        tri_mesh.apply_transform(rot.detach().cpu().numpy())
        pyrend_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        scene.add(pyrend_mesh)

        pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=[1920,1080])


def pointcloud_to_density(points, num_voxels, DEVICE):
    feature_dim = 1
    point_features = torch.ones(points.shape[0], points.shape[1], feature_dim, device=DEVICE)
    pointclouds = Pointclouds(points=points, features=point_features)

    volume_features = torch.zeros(points.shape[0], feature_dim, num_voxels, num_voxels, num_voxels, device=DEVICE)
    volume_densities = torch.zeros(points.shape[0], 1, num_voxels, num_voxels, num_voxels, device=DEVICE)
    initial_volumes = Volumes(
        features = volume_features,
        densities = volume_densities,
        voxel_size = 2 / num_voxels
    )

    updated_volumes = add_pointclouds_to_volumes(
        pointclouds=pointclouds,
        initial_volumes=initial_volumes,
        mode="trilinear",
    )

    return updated_volumes.densities()


def normalizeDensities(volume: torch.tensor) -> torch.tensor:
    _min = volume.amin(dim=(2,3,4), keepdims=True)
    _max = volume.amax(dim=(2,3,4), keepdims=True)
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


def show_rendered_mesh(flamelayer, renderer, cameras, lights, num_views, textures, device, batch_size=1, num_shape_params=100, num_expression_params=50):

    pose_params = torch.zeros(batch_size, 6).to(device)

    # Creating a batch of mean shapes
    shape_params = ((torch.rand(batch_size, num_shape_params) - 0.5) * 2.0).to(device) # Shape to [-1, 1]
    shape_params *= 5.0
    #shape_params += 0.0

    # Cerating a batch of neutral expressions
    expression_params = ((torch.rand(batch_size, num_expression_params) - 0.5) * 2.0).to(device) # Expression to [-1, 1]
    expression_params *= 0.0


    vertices, _ = flamelayer(shape_params, expression_params, pose_params)
    vertices = toMinusOneOne(toUnitCube(vertices))

    mesh = Meshes(vertices, torch.from_numpy(flamelayer.faces.astype(int)).to(device).unsqueeze(0).repeat(vertices.shape[0], 1, 1), textures)
    #mesh = mesh.extend(num_views)

    target_images = renderer(mesh, cameras=cameras, lights=lights)
    target_rgb = [target_images[i, ..., :3].detach().cpu().numpy() for i in range(num_views)]

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
def visualize_prediction(predicted_images, 
                         target_image, title='', 
                         silhouette=False):
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
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")


def pointcloud_to_volume(points: torch.tensor, num_voxels: int = 32):
    sampled_points = toMinusOneOne(toUnitCube(points))
    sample_volume = pointcloud_to_density(sampled_points, num_voxels, points.device)
    return normalizeDensities(sample_volume)


# Return shape params in range (-stddev, stddev)
def get_rand_params(batch_size: int, num_params:int , stddev: float = 2.0):
    return ((torch.rand(batch_size, num_params) - 0.5) * 2.0) * stddev

def main():
    global global_scene
    global global_params
    global global_key

    parser = Args.GlobalArgumentParser()
    args = parser.parse_args()
    config = CAP.ConfigArgsParser(utils.read_config_yaml(os.path.join(args.scene_path, "config.yml")), args)
    config.printFormatted()
    config = config.asNamespace()

    NUM_SHAPE_PARAMS = 100
    config.batch_size = config.num_views_per_iteration
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
    
    shapemodel_key = None
    for key, value in firefly_scene.meshes.items():
        if isinstance(value, Transformable.ShapeModel):
            shapemodel_key = key


    camera_sensor = global_scene.sensors()[0]
    camera_x_fov = global_params['PerspectiveCamera.x_fov']
    camera_near_clip = global_params['PerspectiveCamera.near_clip']
    camera_far_clip = global_params['PerspectiveCamera.far_clip']


    projector_sensor = global_scene.sensors()[1]
    projector_x_fov = global_params['PerspectiveCamera_1.x_fov']
    projector_near_clip = global_params['PerspectiveCamera_1.near_clip']
    projector_far_clip = global_params['PerspectiveCamera_1.far_clip']

    
    K_CAMERA = mi.perspective_projection(camera_sensor.film().size(), camera_sensor.film().crop_size(), camera_sensor.film().crop_offset(), camera_x_fov, camera_near_clip, camera_far_clip).matrix.torch()[0]
    #K_PROJECTOR = mi.perspective_projection(projector_sensor.film().size(), projector_sensor.film().crop_size(), projector_sensor.film().crop_offset(), projector_x_fov, projector_near_clip, projector_far_clip).matrix.torch()[0]
    



    Laser = LaserEstimation.initialize_laser(global_scene, global_params, firefly_scene, config, "GRID", DEVICE)
    
    model = FLAME_3DCNN.Shape(in_channels=1, num_classes=1, shape_params=NUM_SHAPE_PARAMS).to(DEVICE)
    model.train()
    
    loss_funcs = Losses.Handler([
            #[Losses.VGGPerceptual().to(DEVICE), 0.0],
            [torch.nn.MSELoss().to(DEVICE), 1.0],
            #[torch.nn.L1Loss().to(DEVICE),  1.0]
            ])
    

    gt_renderer, diff_renderer, cameras, lights = get_cameras_and_renderer(config.num_views, 128, sigma, 10, DEVICE)
    
    Laser._rays.requires_grad = True
    sigma.requires_grad = True

    optim = torch.optim.Adam([
        {'params': model.parameters(),  'lr': config.lr_model}, 
        {'params': Laser._rays,         'lr': 0.0},
        {'params': sigma,               'lr': config.lr_sigma}
        ])
    
    losses = {"rgb": {"weight": 1.0, "values": []},
            "silhouette": {"weight": 1.0, "values": []},
            "chamf": {"weight": 0.5, "values": []},
            "shape_params": {"weight": 0.0, "values": []},
            "scene_seg": {"weight": 0.2, "values": []},
            }

    #scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, power=0.99, total_iters=config.iterations)
    shapemodel = firefly_scene.meshes[shapemodel_key]
    # Hacky magic number here. But we want a better L2 Loss
    vertex_color = torch.randn(1, 5023, 3).to(DEVICE)
    textures = Textures(verts_rgb=vertex_color)


    reduction_steps = config.sigma_reduce_end
    sigma_step = (config.sigma - config.sigma_end) / reduction_steps
    for i in (progress_bar := tqdm(range(config.iterations))):
        if i == config.warmup_iterations:
            optim.param_groups[1]['lr'] = config.lr_laser

        if i < reduction_steps:
            sigma = sigma - sigma_step

        with torch.no_grad():
            vertices, _ = shapemodel.getVertexData()
            vertices = vertices.unsqueeze(0)
            mesh = Meshes(toMinusOneOne(toUnitCube(vertices)), 
                        shapemodel._shape_layer.faces_tensor.unsqueeze(0), 
                        textures)

            target_images = gt_renderer(mesh.extend(config.num_views), cameras=cameras, lights=lights)
            target_rgb = target_images[..., :3]
            target_silhouette = target_images[..., 3]

        # Initialize optimizer
        segmentation = depth.get_segmentation_from_camera(global_scene).float()

        cv2.imshow("Seg", segmentation.detach().cpu().numpy().astype(np.uint8)*255)
        cv2.waitKey(1)

        firefly_scene.randomize()
        optim.zero_grad()

        sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)
        hit_d = cast_laser(Laser.originPerRay(), Laser.rays())
        world_points = Laser.originPerRay() + Laser.rays() * hit_d
        
        
        CAMERA_WORLD = global_params["PerspectiveCamera.to_world"].matrix.torch()[0]
        world_points_hat = transforms.transform_points(world_points, CAMERA_WORLD.inverse()).squeeze()
        ndc_points = transforms.transform_points(world_points_hat, K_CAMERA).squeeze()

        # We should remove points, that do not fall into the object itself here.
        image_space_points = ndc_points[:, 0:2] * sensor_size
        quantized_indices = image_space_points.floor().int()
        object_hits = segmentation[quantized_indices[:, 1], quantized_indices[:, 0]].nonzero().squeeze()
        
        object_nonhits = (segmentation[quantized_indices[:, 1], quantized_indices[:, 0]] == 0).nonzero().squeeze()
        non_hits_ndc = ndc_points[object_nonhits]

        world_points = world_points[object_hits]

        ndc_raster = rasterization.rasterize_points(ndc_points[:, 0:2], config.sigma, sensor_size)
        ndc_tex = rasterization.softor(ndc_raster)
        sample_volume = pointcloud_to_volume(world_points.unsqueeze(0), num_voxels=config.num_voxels)

        estimated_shape_params = model(sample_volume)
        pred_vertices, _ = shapemodel._shape_layer(estimated_shape_params, shapemodel.expressionParams(), shapemodel.poseParams())
    

        pred_vertices = toMinusOneOne(toUnitCube(pred_vertices))
        pred_mesh = Meshes(pred_vertices, shapemodel._shape_layer.faces_tensor.unsqueeze(0), textures).extend(config.num_views)
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=DEVICE) for k in losses}
        images_predicted = diff_renderer(pred_mesh, cameras=cameras, lights=lights)

        loss_silhouette = loss_funcs(images_predicted[..., 3], target_silhouette)
        loss["silhouette"] += loss_silhouette / config.num_views_per_iteration
        
        loss_rgb = loss_funcs(images_predicted[..., :3], target_rgb)
        loss["rgb"] += loss_rgb / config.num_views_per_iteration

        loss_seg = loss_funcs(segmentation, ndc_tex)
        loss["scene_seg"] += loss_seg

        loss_verts = chamfer_distance(mesh.verts_padded(), pred_vertices)[0]
        loss["chamf"] += loss_verts

        loss_shape = torch.nn.L1Loss()(estimated_shape_params, shapemodel.shapeParams()) / NUM_SHAPE_PARAMS
        loss["shape_params"] += loss_shape

        
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
            #scheduler.step()
            optim.zero_grad()

        with torch.no_grad():
            Laser.clamp_to_fov()
            Laser.normalize_rays()

            sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)
            sparse_depth = rasterization.rasterize_points(ndc_points[:, 0:2], config.sigma, sensor_size)
            sparse_depth = rasterization.softor(sparse_depth)
            sparse_depth = sparse_depth.detach().cpu().numpy() * 255
            sparse_depth = sparse_depth.astype(np.uint8)

            cv2.imshow("Points", (ndc_tex.detach().cpu().numpy()*255).astype(np.uint8))
            cv2.waitKey(1)
    
    plot_losses(losses)
    plt.show()            

    printer.Printer.OKG("Optimization done. Initiating post-processing.")
    


    Laser.save(os.path.join(args.scene_path, "laser.yml"))
    print("Finished everything.")


if __name__ == "__main__":
    main()