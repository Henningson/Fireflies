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
    path = "examples/scenes/hello_world/hello_world.xml"

    mitsuba_scene = mi.load_file(path)
    mitsuba_params = mi.traverse(mitsuba_scene)
    fireflies_scene = fireflies.Scene(mitsuba_params)

    fireflies_scene.mesh_at(0).rotate_z(-np.pi, np.pi)

    fireflies_scene.train()
    for i in range(100):
        fireflies_scene.randomize()

        render = mi.render(mitsuba_scene, spp=10)

        cv2.imshow("a", render_to_opencv(render))
        cv2.imwrite(f"im/{i:05d}.png", render_to_opencv(render))
        cv2.waitKey(1)
