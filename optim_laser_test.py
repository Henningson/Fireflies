import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import laser_torch
import transforms_torch
import utils_torch
import rasterization
import rasterization
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


def render_for_vis(texture, spp=256, seed=1):
    global_params[global_key] = mi.TensorXf(texture.unsqueeze(-1).repeat(1, 1, 3))
    global_params.update()
    return mi.render(global_scene, global_params, spp=256, seed=seed, seed_grad=seed+1)


@dr.wrap_ad(source='torch', target='drjit')
def render(texture, spp=256, seed=1):
    global_params[global_key] = texture
    global_params.update()
    return mi.render(global_scene, global_params, spp=spp, seed=seed, seed_grad=seed+1)


def main():
    global global_scene, global_params, global_key
    device = torch.device("cuda")
    laser_reference = laser_torch.Laser(20, 20, 0.5, torch.eye(4), torch.tensor([0.0, 0.0, 0.0]), max_fov=12)
    laser_reference.initRandomRays()

    laser_init = laser_torch.Laser(20, 20, 0.5, torch.eye(4), torch.tensor([0.0, 0.0, 0.0]), max_fov=12)
    sigma = 0.001
    texture_size = torch.tensor([512, 512], device=device)

    reference_image = generateReference(laser_reference, sigma, texture_size)
    ref_save = np.array(mi.util.convert_to_bitmap(reference_image))[:, :, [2, 1, 0]]
    cv2.imwrite("Ref.png", ref_save)

    init_image = generateReference(laser_init, sigma, texture_size)
    init_save = np.array(mi.util.convert_to_bitmap(init_image))[:, :, [2, 1, 0]]
    cv2.imwrite("Init.png", init_save)
    
    
    points = laser_init.projectRaysToNDC()[:, 0:2]
    texture_init = rasterization.rasterize_points(points, sigma, texture_size)
    scene_init = mi.load_file("scenes/proj_cbox.xml", spp=1024)
    params = mi.traverse(scene_init)

    global_scene = scene_init
    global_params = params
    global_key = "tex.data"

    laser_init._rays.requires_grad = True
    optimizer = torch.optim.Adam([laser_init._rays], lr=0.002)
    loss_fn = torch.nn.L1Loss()

    # Optimization hyper-parameters
    iteration_count = 200
    spp = 8


    writer = imageio.get_writer('test.mp4', fps=25)
    for iter in range(iteration_count):
        optimizer.zero_grad()
        points = laser_init.projectRaysToNDC()[:, 0:2]
        texture_init = rasterization.rasterize_points(points, sigma, texture_size)
        rendered_img = render(texture_init.unsqueeze(-1).repeat(1, 1, 3), "tex.data", spp=spp)
        loss = loss_fn(rendered_img, reference_image.torch())

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            laser_init.randomize_out_of_bounds()

            bitmap = mi.util.convert_to_bitmap(render_for_vis(texture_init))
            bitmap = np.array(bitmap)
            writer.append_data(bitmap)
            print("Loss {0}: {1}".format(iter, loss.item()))
    writer.close()



    #print("Init | GT | Depth")
    #plt.axis("off")
    #plt.title("GT")
    #plt.imshow(image_init)
    #plt.show(block=True)



if __name__ == "__main__":
    main()