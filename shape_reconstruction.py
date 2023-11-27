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


def get_cameras_and_renderer(num_views, image_size, device):    # Get a batch of viewing angles. 
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
        image_size=im_size, 
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
        image_size=im_size, 
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

    return gt_renderer, diff_renderer, cameras


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

    num_init_points = 200
    num_shape_params = 100
    num_expression_params = 50
    num_voxels = 32
    learning_rate = 1.0
    num_views = 8
    num_views_per_iteration = num_views

    shape_interval = 2.0
    expression_interval = 0.0
    pose_interval = 0.0

    plot_period = 100
    Niter = 2000
    faces_per_pixel = 10
    sigma = 1e-4
    config.batch_size = num_views_per_iteration
    batch_size = config.batch_size


    
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
    
    model = FLAME_3DCNN.Shape(in_channels=1, num_classes=1, shape_params=num_shape_params).to(DEVICE)
    model.train()
    
    losses = Losses.Handler([
            #[Losses.VGGPerceptual().to(DEVICE), 0.0],
            [torch.nn.MSELoss().to(DEVICE), 1.0],
            #[torch.nn.L1Loss().to(DEVICE),  1.0]
            ])
    
    gt_renderer, diff_renderer, cameras = get_cameras_and_renderer(num_views, global_scene.sensors()[0].film().size(), DEVICE)
    
    Laser._rays.requires_grad = True
    sigma.requires_grad = True

    optim = torch.optim.Adam([
        {'params': model.parameters(),  'lr': config.lr_model}, 
        {'params': Laser._rays,         'lr': config.lr_laser},
        {'params': sigma,               'lr': config.lr_sigma}
        ])
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, power=0.99, total_iters=Niter)


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

            with torch.no_grad():
                pose_params = get_rand_params(batch_size, 6, pose_interval).to(DEVICE)
                shape_params = get_rand_params(batch_size, num_shape_params, shape_interval).to(DEVICE)
                expression_params = get_rand_params(batch_size, num_expression_params, expression_interval).to(DEVICE)
                vertices, _ = flamelayer(shape_params, expression_params, pose_params)

                mesh = Meshes(toMinusOneOne(toUnitCube(vertices)), 
                            torch.from_numpy(flamelayer.faces.astype(int)).to(DEVICE).unsqueeze(0).repeat(vertices.shape[0], 1, 1), 
                            textures)
                
                sample_volume = pointcloud_to_volume(
                    sample_points_from_meshes(mesh, num_init_points), 
                    num_voxels=num_voxels)

                target_images = gt_renderer(mesh, cameras=cameras, lights=lights)
                target_rgb = [target_images[i, ..., :3] for i in range(num_views)]

                target_cameras = cameras   
                target_silhouette = [target_images[i, ..., 3] for i in range(num_views)]

            # Initialize optimizer
            optim.zero_grad()
            estimated_shape_params = model(sample_volume)
            pred_vertices, _ = flamelayer(estimated_shape_params, expression_params, pose_params)
            pred_vertices = toMinusOneOne(toUnitCube(pred_vertices))
            pred_mesh = Meshes(pred_vertices, torch.from_numpy(flamelayer.faces.astype(int)).to(DEVICE).unsqueeze(0).repeat(pred_vertices.shape[0], 1, 1), textures)
            
            # Losses to smooth /regularize the mesh shape
            loss = {k: torch.tensor(0.0, device=DEVICE) for k in losses}

            # Randomly select two views to optimize over in this iteration.  Compared
            # to using just one view, this helps resolve ambiguities between updating
            # mesh shape vs. updating mesh texture
            for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
                
                images_predicted = renderer_textured(pred_mesh, cameras=target_cameras[j], lights=lights)
                #silhouette_predicted = renderer_silhouette(pred_mesh, cameras=target_cameras[j], lights=lights)
                
                # Squared L2 distance between the predicted silhouette and the target 
                # silhouette from our dataset
                predicted_silhouette = images_predicted[..., 3]
                loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
                loss["silhouette"] += loss_silhouette / num_views_per_iteration
                
                # Squared L2 distance between the predicted RGB image and the target 
                # image from our dataset
                predicted_rgb = images_predicted[..., :3]
                loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
                loss["rgb"] += loss_rgb / num_views_per_iteration

                loss_verts = chamfer_distance(mesh.verts_padded(), pred_vertices)[0]
                loss["chamf"] += loss_verts

                loss_shape = torch.nn.L1Loss()(estimated_shape_params, shape_params) / num_shape_params
                loss["shape_params"] += loss_shape

            
            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=DEVICE)
            sum_loss.requires_grad = True

            for k, l in loss.items():
                sum_loss = sum_loss + l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
            
            # Print the losses
            loop.set_description("total_loss = %.6f" % sum_loss)


            sum_loss.backward()
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