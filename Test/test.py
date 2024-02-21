import os
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import drjit as dr
import Objects.entity as entity
import torch
import Graphics.Firefly as Firefly
import Graphics.LaserEstimation as LaserEstimation
import matplotlib.pyplot as plt
import Graphics.depth as depth


def test_render(firefly_scene, mitsuba_scene, mitsuba_params, num=5):
    for i in range(num):
        firefly_scene.randomize()
        mitsuba_params.update()
        image = mi.render(mitsuba_scene, spp=16)
        plt.axis("off")
        plt.imshow(image)
        plt.show()


def test_epipolar_constraint_map(mitsuba_scene, mitsuba_params, device):
    constraint_map = LaserEstimation.generate_epipolar_constraints(
        mitsuba_scene, mitsuba_params, device
    )
    plt.axis("off")
    plt.imshow(constraint_map.detach().cpu().numpy())
    plt.show()


def test_depth_maps(firefly_scene, mitsuba_scene):
    depth_maps = depth.random_depth_maps(firefly_scene, mitsuba_scene, num_maps=5)

    for i in range(depth_maps.shape[0]):
        plt.axis("off")
        plt.imshow(depth_maps[i].detach().cpu().numpy())
        plt.show()


if __name__ == "__main__":
    base_path = "scenes/EasyCube/"
    sequential = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    mitsuba_params["Projector.to_world"] = mitsuba_params[
        "PerspectiveCamera_1.to_world"
    ]
    mitsuba_params.update()

    firefly_scene = Firefly.Scene(
        mitsuba_params, base_path, sequential_animation=sequential
    )

    # test_render(firefly_scene, mitsuba_scene, mitsuba_params)
    test_epipolar_constraint_map(mitsuba_scene, mitsuba_params, DEVICE)
    test_depth_maps(firefly_scene, mitsuba_scene)
