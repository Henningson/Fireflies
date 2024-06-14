import cv2
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")

import torch
import fireflies
import fireflies.sampling


def render_to_opencv(render):
    render = torch.clamp(render.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
    return (render * 255).astype(np.uint8)


# Let's define an animation function that will visualize a sine wave.
# You can define completely arbitrary functions.
def animation_function(vertices: torch.tensor, time: float) -> torch.tensor:
    # Let's not change the incoming vertices in place.
    vertices_clone = vertices.clone()

    # Change z coordinate of plane via the sin(x_coordinate + time)
    wave_direction = 0

    vertices_clone[:, 1] = (
        vertices_clone[:, 1]
        + torch.sin(vertices_clone[:, 2] * 10.0 + time * 20.0) / 10.0
    )

    return vertices_clone


if __name__ == "__main__":
    path = "examples/scenes/animation/animation.xml"

    mitsuba_scene = mi.load_file(path)
    mitsuba_params = mi.traverse(mitsuba_scene)
    ff_scene = fireflies.Scene(mitsuba_params)

    mesh = ff_scene.mesh("mesh-Animation")
    mesh.add_animation_func(
        animation_function,
        torch.tensor([0.0]).to(ff_scene.device()),
        torch.tensor([2 * np.pi]).to(ff_scene.device()),
    )

    normal_distribution_sampler = fireflies.sampling.GaussianSampler(
        min=torch.ones(3, device=ff_scene.device())*0.5,
        max=torch.ones(3, device=ff_scene.device())*1.5, 
        mean=torch.ones(3, device=ff_scene.device())*1.0, 
        std=torch.ones(3, device=ff_scene.device())*0.5, 
        eval_step_size=0.01,
        )
    mesh.set_scale_sampler(normal_distribution_sampler)

    ff_scene.train()
    for i in range(1000):
        ff_scene.randomize()
        render = mi.render(mitsuba_scene, spp=12)
        cv2.imwrite("a.png", render_to_opencv(render))
        cv2.waitKey(25)