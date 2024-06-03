import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "Utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Objects"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Graphics"))

from bs4 import BeautifulSoup
from pathlib import Path
import os
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import utils
import transforms
import torch
import Transformable
import numpy as np
import math_helper
import LaserEstimation
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import rasterization
import Firefly
import kornia

from argparse import Namespace

from argparse import Namespace


if __name__ == "__main__":
    base_path = "scenes/scenes_for_visualization/RealColon/"

    config = utils.read_config_yaml(os.path.join(base_path, "config.yml"))
    config = Namespace(**config)

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    mitsuba_params["PerspectiveCamera.film.size"] //= config.downscale_factor
    mitsuba_params["PerspectiveCamera_1.film.size"] //= config.downscale_factor

    mitsuba_params["PerspectiveCamera_1.to_world"] = mitsuba_params[
        "PerspectiveCamera.to_world"
    ]

    mitsuba_params["Projector.to_world"] = mitsuba_params[
        "PerspectiveCamera_1.to_world"
    ]

    mitsuba_params.update()

    sigma = torch.tensor([config.sigma], device="cuda")
    texture_size = torch.tensor(mitsuba_scene.sensors()[1].film().size(), device="cuda")

    firefly_scene = Firefly.Scene(mitsuba_params, base_path, sequential_animation=True)
    firefly_scene.eval()

    laser_init = LaserEstimation.initialize_laser(
        mitsuba_scene,
        mitsuba_params,
        firefly_scene,
        config,
        config.pattern_initialization,
        device="cuda",
    )
    points = laser_init.projectRaysToNDC()[:, 0:2]

    """
    colors = [(0.0, 0.1921, 0.4156), (0, 0.69, 0.314)]  # R -> G -> B
    fig = plt.figure(frameon=False)
    fig.set_size_inches(16 / 16 * 10, 9 / 16 * 10)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.scatter(
        points.detach().cpu().numpy()[:, 0],
        points.detach().cpu().numpy()[:, 1],
        s=60.0 * 2.5,
        color=colors[0],
    )
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    cv2.imshow("Sexy Tex", img_plot[:, :, [2, 1, 0]])
    """
    # for i in range(1, 1000):

    camera_sensor = mitsuba_scene.sensors()[0]
    camera_x_fov = mitsuba_params["PerspectiveCamera.x_fov"]
    camera_near_clip = mitsuba_params["PerspectiveCamera.near_clip"]
    camera_far_clip = mitsuba_params["PerspectiveCamera.far_clip"]

    K_CAMERA = mi.perspective_projection(
        camera_sensor.film().size(),
        camera_sensor.film().crop_size(),
        camera_sensor.film().crop_offset(),
        camera_x_fov,
        camera_near_clip,
        camera_far_clip,
    ).matrix.torch()[0]

    texture_init = rasterization.rasterize_points(points, config.sigma, texture_size)
    texture_init = rasterization.softor(texture_init)

    '''
    laser_init._rays.requires_grad = True

    optim = torch.optim.Adam([{"params": laser_init._rays, "lr": 0.001}])

    for i in range(1000):
        firefly_scene.randomize()
        optim.zero_grad()

        laser_s = laser_init.originPerRay() + 1.0 * laser_init.rays()
        laser_e = laser_init.originPerRay() + 1.1 * laser_init.rays()

        CAMERA_WORLD = mitsuba_params["PerspectiveCamera.to_world"].matrix.torch()[0]
        laser_s = transforms.transform_points(laser_s, CAMERA_WORLD.inverse()).squeeze()
        laser_s = transforms.transform_points(laser_s, K_CAMERA).squeeze()[:, :2]

        laser_e = transforms.transform_points(laser_e, CAMERA_WORLD.inverse()).squeeze()
        laser_e = transforms.transform_points(laser_e, K_CAMERA).squeeze()[:, :2]

        lines = torch.concat(
            [laser_s.unsqueeze(-1), laser_e.unsqueeze(-1)], dim=-1
        ).transpose(1, 2)
        lines_copy = lines.clone()
        line_render = rasterization.rasterize_lines(lines, config.sigma, texture_size)
        line_softor = rasterization.softor(line_render)

        loss = torch.nn.L1Loss()(line_softor, line_render.sum(dim=0))
        loss.backward()
        optim.step()

        with torch.no_grad():
            line_vis = line_softor.detach().cpu().numpy() * 255
            line_vis = line_vis.astype(np.uint8).transpose()
            cv2.imshow("Line Render", line_vis)

            """
            colors = [(0.0, 0.1921, 0.4156), (0, 0.69, 0.314)]  # R -> G -> B
            fig = plt.figure(frameon=False)
            fig.set_size_inches(10, 10)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_axis_off()
            ax.set_aspect(aspect="equal")
            fig.add_axes(ax)

            lines_copy = lines_copy.transpose(1, 2).detach().cpu().numpy()
            for j in range(lines_copy.shape[0]):
                ax.plot(
                    lines_copy[j, 0, :],
                    lines_copy[j, 1, :],
                    c=colors[0],
                    linewidth=9.5,
                    solid_capstyle="round",
                )  # c=colors[0], linewidth=60)

            fig.canvas.draw()
            img_plot = np.array(fig.canvas.renderer.buffer_rgba())
            img_plot = np.transpose(
                cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR), [1, 0, 2]
            )
            img_plot = cv2.flip(img_plot, 1)
            cv2.imshow("Testrender", img_plot)
            cv2.imwrite(f"EpipolarLinesOptimization/{i:05d}.png", img_plot)
            plt.close()
            """
            laser_tex = rasterization.softor(
                laser_init.generateTexture(config.sigma, texture_size)
            )
            laser_tex = laser_tex.detach().cpu().numpy()
            cv2.imshow("LASERTEX", laser_tex)
            cv2.waitKey(1)
            laser_init.clamp_to_fov(clamp_val=0.99)
            laser_init.normalize_rays()

    exit()
    '''

    # sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)

    # texture_init = torch.ones(texture_init.shape, device=texture_init.device)
    # texture_init = torch.flipud(texture_init)
    # print(i)
    # cv2.imshow("Wat", texture_init.detach().cpu().numpy())
    # cv2.waitKey(1)

    texture_init /= 1.25
    texture_init = kornia.filters.gaussian_blur2d(
        texture_init.unsqueeze(0).unsqueeze(0), (5, 5), (5.0, 5.0)
    ).squeeze()

    texture_init = torch.stack(
        [torch.zeros_like(texture_init), texture_init, torch.zeros_like(texture_init)],
        dim=-1,
    )
    texture_init += 1
    texture_init /= texture_init

    mitsuba_params["tex.data"] = texture_init
    firefly_scene.eval()

    # firefly_scene.randomize()
    # render_im = mi.render(mitsuba_scene)

    for i in tqdm(range(300)):
        firefly_scene.randomize()

        render_im = mi.render(mitsuba_scene, spp=500)
        render_im = torch.clamp(render_im.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
        render_im *= 255
        render_im = render_im.astype(np.uint8)
        cv2.imshow("a", render_im)
        cv2.waitKey(1)

        cv2.imwrite("COLON_VIS/{:05d}.png".format(i), render_im)
