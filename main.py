from tqdm import tqdm
import os
import cv2
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")

import torch
import fireflies


def render_to_opencv(render):
    render = torch.clamp(render.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
    return (render * 255).astype(np.uint8)


if __name__ == "__main__":

    base_path = "Old/scenes/Vocalfold"

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    fireflies_scene = fireflies.Scene(mitsuba_params)

    fireflies_scene.mesh_at(0).scale_y(0.0, 2.0)

    for i in range(300):
        fireflies_scene.randomize()

        render = mi.render(mitsuba_scene, spp=10)
        cv2.imshow("a", render_to_opencv(render))
        cv2.waitKey(10)
