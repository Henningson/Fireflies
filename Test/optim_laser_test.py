import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import Objects.laser as laser
import Utils.transforms as transforms
import Graphics.rasterization as rasterization
import matplotlib.pyplot as plt
import torch
import numpy as np

torch.autograd.set_detect_anomaly(True)
import imageio
import cv2

global_scene = None
global_params = None
global_key = None


def generateReference(laser, sigma, texture_size):
    points = laser.projectRaysToNDC()[:, 0:2]

    texture = rasterization.rasterize_points(points, sigma, texture_size)
    scene_init = mi.load_file("scenes/proj_cbox.xml", spp=256)
    params = mi.traverse(scene_init)

    params["tex.data"] = mi.TensorXf(texture.unsqueeze(-1).repeat(1, 1, 3))
    params.update()

    render_init = mi.render(scene_init, spp=256)

    return render_init


@dr.wrap_ad(source="torch", target="drjit")
def render(texture, spp=256, seed=1):
    global_params[global_key] = texture
    global_params.update()
    return mi.render(
        global_scene, global_params, spp=spp, seed=seed, seed_grad=seed + 1
    )


def render_for_vis(texture, spp=256, seed=1):
    global_params[global_key] = mi.TensorXf(texture.unsqueeze(-1).repeat(1, 1, 3))
    global_params.update()
    return mi.render(
        global_scene, global_params, spp=256, seed=seed, seed_grad=seed + 1
    )


@dr.wrap_ad(source="torch", target="drjit")
def render_depth(scene, spp=64):
    sensor = scene.sensors()[0]
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    total_samples = dr.prod(film_size) * spp

    if sampler.wavefront_size() != total_samples:
        sampler.seed(0, total_samples)

    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    pos = dr.arange(mi.UInt32, total_samples)

    pos //= spp
    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(
        mi.Float(pos % int(film_size[0])), mi.Float(pos // int(film_size[0]))
    )

    pos += sampler.next_2d()

    # Sample rays starting from the camera sensor
    rays, weights = sensor.sample_ray_differential(
        time=0, sample1=sampler.next_1d(), sample2=pos * scale, sample3=0
    )

    # Intersect rays with the scene geometry
    surface_interaction = scene.ray_intersect(rays)

    # Given intersection, compute the final pixel values as the depth t
    # of the sampled surface interaction
    result = surface_interaction.t

    # Set to zero if no intersection was found
    result[~surface_interaction.is_valid()] = 0

    depth = (
        torch.tensor(result)
        .reshape(sensor.film().size()[0], sensor.film().size()[1], -1)
        .mean(dim=-1)
    )
    depth = depth / depth.max()

    return depth


def main():
    global global_scene, global_params, global_key
    device = torch.device("cuda")
    laser_reference = laser.DeprecatedLaser(
        20, 20, 0.5, torch.eye(4), torch.tensor([0.0, 0.0, 0.0]), max_fov=12
    )
    laser_reference.initRandomRays()

    laser_init = laser.DeprecatedLaser(
        20, 20, 0.5, torch.eye(4), torch.tensor([0.0, 0.0, 0.0]), max_fov=12
    )
    sigma = 0.001
    texture_size = torch.tensor([512, 512], device=device)

    reference_image = generateReference(laser_reference, sigma, texture_size)
    ref_save = np.array(mi.util.convert_to_bitmap(reference_image))[:, :, [2, 1, 0]]
    cv2.imwrite("Ref.png", ref_save)

    init_image = generateReference(laser_init, sigma, texture_size)
    init_save = np.array(mi.util.convert_to_bitmap(init_image))[:, :, [2, 1, 0]]
    cv2.imwrite("Init.png", init_save)
    init_gray = init_save.mean(axis=-1)

    points = laser_init.projectRaysToNDC()[:, 0:2]
    texture_init = rasterization.rasterize_points(points, sigma, texture_size)
    scene_init = mi.load_file("scenes/proj_cbox.xml", spp=1024)
    params = mi.traverse(scene_init)

    global_scene = scene_init
    global_params = params
    global_key = "tex.data"

    laser_init._rays.requires_grad = True
    optimizer = torch.optim.Adam([laser_init._rays], lr=0.0019)
    loss_fn = torch.nn.L1Loss()

    # Optimization hyper-parameters
    iteration_count = 1000
    spp = 8

    writer = imageio.get_writer("opt_scene.mp4", fps=25)
    writer_error = imageio.get_writer("opt_error.mp4", fps=25)

    reference_image = utils_torch.normalize_channelwise(reference_image.torch())
    for iter in range(iteration_count):
        optimizer.zero_grad()
        points = laser_init.projectRaysToNDC()[:, 0:2]
        texture_init = rasterization.rasterize_points(points, sigma, texture_size)

        rendered_img = render(texture_init.unsqueeze(-1).repeat(1, 1, 3), spp=spp)
        grayscale_render = rendered_img.mean(dim=-1)
        grayscale_render = grayscale_render / grayscale_render.max()

        rendered_depth = render_depth(global_scene)

        # rendered_img = utils_torch.normalize_channelwise(rendered_img)

        thresh_img = grayscale_render * (grayscale_render > 0.05)

        cv2.imshow(
            "Rendered Image", rendered_img.detach().cpu().numpy()[:, :, [2, 1, 0]]
        )
        cv2.imshow(
            "Thresholded Image",
            (thresh_img.detach().cpu().numpy() * 255).astype(np.uint8),
        )
        cv2.imshow("Depth", rendered_depth.detach().cpu().numpy())
        cv2.waitKey(1)

        loss = loss_fn(rendered_img, torch.randn(reference_image.shape, device=device))

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            laser_init.randomize_out_of_bounds()
            laser_init.normalize_rays()

            # Visualization stuff
            bitmap = mi.util.convert_to_bitmap(render_for_vis(texture_init))
            bitmap = np.array(bitmap)
            bitmap_gray = bitmap.mean(axis=-1)

            error_map = utils_np.normalize(
                np.abs(
                    bitmap_gray - reference_image.detach().cpu().numpy().mean(axis=-1)
                )
            )
            error_map = cv2.applyColorMap(
                (error_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
            )

            writer.append_data(bitmap)
            writer_error.append_data(error_map)
            print("Loss {0}: {1}".format(iter, loss.item()))
    writer.close()
    writer_error.close()

    # print("Init | GT | Depth")
    # plt.axis("off")
    # plt.title("GT")
    # plt.imshow(image_init)
    # plt.show(block=True)


if __name__ == "__main__":
    main()
