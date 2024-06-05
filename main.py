from tqdm import tqdm
import os
import cv2
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")

import torch
import fireflies

if __name__ == "__main__":

    base_path = "Old/scenes/Vocalfold"

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    fireflies_scene = fireflies.Scene(mitsuba_params)

    mesh = fireflies_scene.mesh_at(0)
    mesh.rotate_x(-np.pi, np.pi)

    for i in tqdm(range(300)):

        render_im = mi.render(mitsuba_scene, spp=10)
        render_im = torch.clamp(render_im.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
        render_im *= 255
        render_im = render_im.astype(np.uint8)
        cv2.imshow("a", render_im)
        cv2.waitKey(10)

        fireflies_scene.randomize()
