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

    mi_scene = mi.load_file(path)
    mi_params = mi.traverse(mi_scene)
    ff_scene = fireflies.Scene(mi_params)

    print(ff_scene.material_at(1)._vec3_attributes)

    # Lets randomize the color of the cube.
    min_color = torch.tensor([0.2, 0.3, 0.2], device=ff_scene._device)
    max_color = torch.tensor([0.8, 1.0, 0.8], device=ff_scene._device)

    material = ff_scene.material("mat-Material")
    material.add_vec3_key("brdf_0.base_color.value", min_color, max_color)

    ff_scene.train()
    for i in range(100):
        ff_scene.randomize()

        render = mi.render(mi_scene, spp=10)

        cv2.imshow("a", render_to_opencv(render))
        cv2.imwrite(f"im/{i:05d}.png", render_to_opencv(render))
        cv2.waitKey(10)
