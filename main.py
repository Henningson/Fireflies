import cv2
import numpy as np
import torch

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import fireflies


def render_to_opencv(render):
    render = torch.clamp(render.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
    return (render * 255).astype(np.uint8)


if __name__ == "__main__":
    path = "examples/scenes/parent_child/parent_child.xml"

    mi_scene = mi.load_file(path)
    mi_params = mi.traverse(mi_scene)
    ff_scene = fireflies.Scene(mi_params)

    cone = ff_scene.mesh("mesh-Cone")
    sphere = ff_scene.mesh("mesh-Sphere")
    light = ff_scene.light("emit-Light")

    # Add sphere as the cones parent
    # Also let the cone be randomizable, since it wouldn't be randomized if this is not set.
    cone.setParent(sphere)
    cone.set_randomizable(True)

    min_intensity = torch.tensor([150, 0, 0], device=ff_scene._device)
    max_intensity = torch.tensor([150, 150, 150], device=ff_scene._device)
    light.add_vec3_key("intensity.value", min_intensity, max_intensity)

    # Rotate everything around the z-axis.
    sphere.rotate_z(-np.pi, np.pi)

    ff_scene.eval()
    for i in range(100):
        ff_scene.randomize()

        render = mi.render(mi_scene, spp=10)

        cv2.imshow("a", render_to_opencv(render))
        cv2.imwrite(f"im/{i:05d}.png", render_to_opencv(render))
        cv2.waitKey(10)
