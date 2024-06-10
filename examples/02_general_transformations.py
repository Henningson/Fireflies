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
    path = "scenes/hello_world/hello_world.xml"

    mi_scene = mi.load_file(path)
    mi_params = mi.traverse(mi_scene)
    ff_scene = fireflies.Scene(mi_params)

    mesh = ff_scene.mesh_at(0)

    # Rotations
    mesh.rotate_x(-0.5, 0.5)
    mesh.rotate_y(-0.5, 0.5)
    mesh.rotate_z(-0.5, 0.5)
    mesh.rotate(
        torch.tensor([-0.5, -0.5, -0.5], device=ff_scene._device),
        torch.tensor([0.5, 0.5, 0.5], device=ff_scene._device),
    )

    # Translations
    mesh.translate_x(-0.5, 0.5)
    mesh.translate_y(-0.5, 0.5)
    mesh.translate_z(-0.5, 0.5)
    mesh.translate(
        torch.tensor([-0.5, -0.5, -0.5], device=ff_scene._device),
        torch.tensor([0.5, 0.5, 0.5], device=ff_scene._device),
    )

    # Scale
    mesh.scale_x(-0.5, 0.5)
    mesh.scale_y(-0.5, 0.5)
    mesh.scale_z(-0.5, 0.5)
    mesh.scale(
        torch.tensor([-0.5, -0.5, -0.5], device=ff_scene._device),
        torch.tensor([0.5, 0.5, 0.5], device=ff_scene._device),
    )

    # There's more in later examples :)

    ff_scene.train()
    for i in range(100):
        ff_scene.randomize()

        render = mi.render(mi_scene, spp=10)

        cv2.imshow("a", render_to_opencv(render))
        cv2.imwrite(f"im/{i:05d}.png", render_to_opencv(render))
        cv2.waitKey(10)
