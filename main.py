import cv2
import mitsuba as mi
import numpy as np
import kornia

mi.set_variant("cuda_ad_rgb")

import torch
import fireflies
import fireflies.sampling
import fireflies.projection.laser
import fireflies.postprocessing
import fireflies.utils.math
import fireflies.graphics.depth
import os

from tqdm import tqdm


def render_to_numpy(render):
    render = torch.clamp(render.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
    return render


if __name__ == "__main__":
    path = "examples/scenes/realistic_vf/vocalfold.xml"
    dataset_path = "../LearningFromFireflies/fireflies_dataset_v4/"

    mitsuba_scene = mi.load_file(path, parallel=False)
    mitsuba_params = mi.traverse(mitsuba_scene)
    ff_scene = fireflies.Scene(mitsuba_params)
    ff_scene._camera._name = "Camera"

    projector_sensor = mitsuba_scene.sensors()[1]
    x_fov = mitsuba_params["PerspectiveCamera_1.x_fov"]
    near_clip = mitsuba_params["PerspectiveCamera_1.near_clip"]
    far_clip = mitsuba_params["PerspectiveCamera_1.far_clip"]

    K_PROJECTOR = mi.perspective_projection(
        projector_sensor.film().size(),
        projector_sensor.film().crop_size(),
        projector_sensor.film().crop_offset(),
        x_fov,
        near_clip,
        far_clip,
    ).matrix.torch()[0]

    # laser_rays = fireflies.projection.Laser.generate_uniform_rays(
    #    0.0275, 18, 18, device=ff_scene.device()
    # )
    laser_rays = fireflies.projection.Laser.generate_uniform_rays(
        0.0275, 18, 18, device=ff_scene.device()
    )

    laser = fireflies.projection.Laser(
        ff_scene._projector,
        laser_rays,
        K_PROJECTOR,
        x_fov,
        near_clip,
        far_clip,
        device=ff_scene.device(),
    )
    texture = laser.generateTexture(
        10.0, torch.tensor([500, 500], device=ff_scene.device())
    )
    texture = texture.sum(dim=0)

    texture = kornia.filters.gaussian_blur2d(
        texture.unsqueeze(0).unsqueeze(0), (5, 5), (3, 3)
    ).squeeze()
    texture = torch.stack(
        [torch.zeros_like(texture), texture, torch.zeros_like(texture)]
    )
    texture = torch.movedim(texture, 0, -1)

    mitsuba_params["tex.data"] = mi.TensorXf(texture.cpu().numpy())

    vocalfold_mesh = ff_scene.mesh("mesh-Vocalfold")
    vocalfold_mesh.scale_x(1.0, 3.0)
    vocalfold_mesh.scale_z(1.0, 3.0)
    vocalfold_mesh.rotate_y(-0.2, 0.2)
    vocalfold_mesh.translate_y(-0.05, -0.05)
    vocalfold_mesh.add_train_animation_from_obj("examples/scenes/vocalfold_new/train/")
    vocalfold_mesh.add_eval_animation_from_obj("examples/scenes/vocalfold_new/test/")

    a = vocalfold_mesh._anim_data_train
    scale_mat = torch.eye(4, device=ff_scene.device()) * 0.054
    scale_mat[3, 3] = 1.0
    for i in range(a.shape[0]):
        a[i] = fireflies.utils.math.transform_points(a[i], scale_mat)

    larynx_mesh = ff_scene.mesh("mesh-Larynx")
    larynx_mesh.scale_x(1.0, 4.0)
    larynx_mesh.scale_z(1.0, 2.0)

    # Randomization of Mucosa material
    material = ff_scene.material("mat-Mucosa")
    material.add_float_key("brdf_0.clearcoat.value", 0.0, 1.0)
    material.add_float_key("brdf_0.clearcoat_gloss.value", 0.0, 1.0)
    material.add_float_key("brdf_0.metallic.value", 0.0, 0.5)
    material.add_float_key("brdf_0.specular", 0.0, 1.0)
    material.add_float_key("brdf_0.roughness.value", 0.0, 1.0)
    material.add_float_key("brdf_0.anisotropic.value", 0.0, 1.0)
    material.add_float_key("brdf_0.sheen.value", 0.0, 0.5)
    material.add_float_key("brdf_0.spec_trans.value", 0.0, 0.4)
    material.add_float_key("brdf_0.flatness.value", 0.0, 1.0)

    # Camera Randomization
    ff_scene._camera.translate_x(-0.15, 0.15)
    ff_scene._camera.translate_y(-0.15, 0.0)
    ff_scene._camera.translate_z(-0.15, 0.15)
    ff_scene._camera.rotate_x(-0.2, 0.2)
    ff_scene._camera.rotate_y(-0.5, 0.5)
    ff_scene._camera.rotate_z((-np.pi / 2.0) - 0.5, (-np.pi / 2.0) + 0.5)
    ff_scene._camera.add_float_key("x_fov", 70.0, 130.0)

    # Light Randomization
    scalar_to_vec3_sampler = fireflies.sampling.UniformScalarToVec3Sampler(
        0.1, 10.0, device=ff_scene.device()
    )
    light = ff_scene.light("emit-Spot")
    light.add_vec3_sampler("intensity.value", scalar_to_vec3_sampler)

    texture = (
        mitsuba_params["mat-Mucosa.brdf_0.base_color.data"]
        .torch()
        .moveaxis(-1, 0)
        .shape
    )

    lerp_sampler = fireflies.sampling.NoiseTextureLerpSampler(
        color_a=torch.tensor([0.0, 0.0, 0.0], device=ff_scene.device()),
        color_b=torch.tensor([1.0, 1.0, 1.0], device=ff_scene.device()),
        texture_shape=(1024, 1024),
    )

    post_process_funcs = [
        fireflies.postprocessing.GaussianBlur((3, 3), (5, 5), 0.5),
        fireflies.postprocessing.ApplySilhouette(),
        fireflies.postprocessing.WhiteNoise(0.0, 0.05, 0.5),
    ]
    post_processor = fireflies.postprocessing.PostProcessor(post_process_funcs)
    spp_sampler = fireflies.sampling.AnimationSampler(1, 100, 1, 100)
    ff_scene.train()
    count = 0
    while count != 10000:
        lerp_sampler._color_a = torch.rand((3), device=ff_scene.device())
        lerp_sampler._color_b = torch.rand((3), device=ff_scene.device())
        mucosa_texture = lerp_sampler.sample()
        mitsuba_params["mat-Mucosa.brdf_0.base_color.data"] = mi.TensorXf(
            mucosa_texture.moveaxis(0, -1).cpu().numpy()
        )

        ff_scene.randomize()
        render = mi.render(mitsuba_scene, spp=spp_sampler.sample())
        render = render_to_numpy(render)
        render = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
        render = post_processor.post_process(render)

        segmentation = (
            fireflies.graphics.depth.get_segmentation_from_camera(mitsuba_scene)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if segmentation.max() == 0:
            continue

        seg_test = segmentation.copy()
        seg_test[seg_test == 1] = 0
        seg_test[seg_test == 2] = 1
        seg_test = 1 - seg_test
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_test.astype(np.uint8)
        )

        if n_labels > 3:
            continue

        segmentation_map = segmentation / segmentation.max()
        final = cv2.hconcat([render, segmentation_map])

        cv2.imwrite(
            os.path.join(dataset_path, "train/images/{0:05d}.png".format(count)),
            (render * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(dataset_path, "train/segmentation/{0:05d}.png".format(count)),
            segmentation.astype(np.uint8),
        )
        count += 1

    count = 0
    while count != 500:
        lerp_sampler._color_a = torch.rand((3), device=ff_scene.device())
        lerp_sampler._color_b = torch.rand((3), device=ff_scene.device())
        mucosa_texture = lerp_sampler.sample()
        mitsuba_params["mat-Mucosa.brdf_0.base_color.data"] = mi.TensorXf(
            mucosa_texture.moveaxis(0, -1).cpu().numpy()
        )

        ff_scene.randomize()
        render = mi.render(mitsuba_scene, spp=spp_sampler.sample())
        render = render_to_numpy(render)
        render = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
        render = post_processor.post_process(render)

        segmentation = (
            fireflies.graphics.depth.get_segmentation_from_camera(mitsuba_scene)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if segmentation.max() == 0:
            continue

        seg_test = segmentation.copy()
        seg_test[seg_test == 1] = 0
        seg_test[seg_test == 2] = 1
        seg_test = 1 - seg_test
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_test.astype(np.uint8)
        )

        if n_labels > 3:
            continue

        segmentation_map = segmentation / segmentation.max()
        final = cv2.hconcat([render, segmentation_map])

        cv2.imwrite(
            os.path.join(dataset_path, "eval/images/{0:05d}.png".format(count)),
            (render * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(dataset_path, "eval/segmentation/{0:05d}.png".format(count)),
            segmentation.astype(np.uint8),
        )
        count += 1

'''
if __name__ == "__main__":
    path = "examples/scenes/vocalfold_new/vocalfold.xml"

    mitsuba_scene = mi.load_file(path, parallel=False)
    mitsuba_params = mi.traverse(mitsuba_scene)
    ff_scene = fireflies.Scene(mitsuba_params)
    ff_scene._camera._name = "Camera"

    projector_sensor = mitsuba_scene.sensors()[1]
    x_fov = mitsuba_params["PerspectiveCamera_1.x_fov"]
    near_clip = mitsuba_params["PerspectiveCamera_1.near_clip"]
    far_clip = mitsuba_params["PerspectiveCamera_1.far_clip"]

    dataset_path = "../LearningFromFireflies/fireflies_dataset_v3/"

    K_PROJECTOR = mi.perspective_projection(
        projector_sensor.film().size(),
        projector_sensor.film().crop_size(),
        projector_sensor.film().crop_offset(),
        x_fov,
        near_clip,
        far_clip,
    ).matrix.torch()[0]

    # laser_rays = fireflies.projection.Laser.generate_uniform_rays(
    #    0.0275, 18, 18, device=ff_scene.device()
    # )
    laser_rays = fireflies.projection.Laser.generate_uniform_rays(
        0.0275, 18, 18, device=ff_scene.device()
    )

    laser = fireflies.projection.Laser(
        ff_scene._projector,
        laser_rays,
        K_PROJECTOR,
        x_fov,
        near_clip,
        far_clip,
        device=ff_scene.device(),
    )
    texture = laser.generateTexture(
        10.0, torch.tensor([500, 500], device=ff_scene.device())
    )
    texture = texture.sum(dim=0)

    texture = kornia.filters.gaussian_blur2d(
        texture.unsqueeze(0).unsqueeze(0), (5, 5), (3, 3)
    ).squeeze()
    texture = torch.stack(
        [torch.zeros_like(texture), texture, torch.zeros_like(texture)]
    )
    texture = torch.movedim(texture, 0, -1)

    mitsuba_params["tex.data"] = mi.TensorXf(texture.cpu().numpy())

    vocalfold_mesh = ff_scene.mesh("mesh-VocalFold")
    vocalfold_mesh.scale_x(0.75, 3.0)
    vocalfold_mesh.scale_y(1.1, 2.0)
    vocalfold_mesh.rotate_y(-0.2, 0.2)
    vocalfold_mesh.add_train_animation_from_obj("examples/scenes/vocalfold_new/train/")
    vocalfold_mesh.add_eval_animation_from_obj("examples/scenes/vocalfold_new/test/")

    a = vocalfold_mesh._anim_data_train
    scale_mat = torch.eye(4, device=ff_scene.device()) * 0.05
    scale_mat[3, 3] = 1.0
    rot_mat = fireflies.utils.math.toMat4x4(
        fireflies.utils.math.getXTransform(np.pi / 2.0, ff_scene.device())
    )
    for i in range(a.shape[0]):
        a[i] = fireflies.utils.math.transform_points(a[i], scale_mat)
        a[i] = fireflies.utils.math.transform_points(a[i], rot_mat)

    larynx_mesh = ff_scene.mesh("mesh-Larynx")
    larynx_mesh.scale_x(0.3, 1.2)
    larynx_mesh.rotate_y(-0.5, 0.5)
    # larynx_mesh.scale_z(1.0, 2.5)

    material = ff_scene.material("mat-Default OBJ")
    material.add_vec3_key(
        "brdf_0.base_color.value",
        torch.tensor([0.3, 0.3, 0.33], device=ff_scene.device()),
        torch.tensor([0.85, 0.85, 0.85], device=ff_scene.device()),
    )

    for key in material.float_attributes():
        if "sampling_rate" in key:
            continue

        material.add_float_key(key, 0.01, 0.99)

    scalar_to_vec3_sampler = fireflies.sampling.UniformScalarToVec3Sampler(
        1.0, 80.0, device=ff_scene.device()
    )
    light = ff_scene.light("emit-Spot")
    light.add_vec3_sampler("intensity.value", scalar_to_vec3_sampler)

    post_process_funcs = [
        fireflies.postprocessing.GaussianBlur((3, 3), (5, 5), 0.5),
        fireflies.postprocessing.ApplySilhouette(),
        fireflies.postprocessing.WhiteNoise(0.0, 0.05, 0.5),
    ]
    post_processor = fireflies.postprocessing.PostProcessor(post_process_funcs)

    ff_scene._camera.translate_x(-0.5, 0.5)
    ff_scene._camera.translate_y(-0.5, 0.5)
    ff_scene._camera.translate_z(-0.5, 0.5)
    ff_scene._camera.add_float_key("x_fov", 20.0, 50.0)
    ff_scene.train()

    spp_sampler = fireflies.sampling.AnimationSampler(1, 100, 1, 100)
    spp_sampler.train()
    """
    count = 0
    while count != 10000:
        ff_scene.randomize()
        render = mi.render(mitsuba_scene, spp=spp_sampler.sample())
        render = render_to_numpy(render)
        render = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)

        render = post_processor.post_process(render)

        segmentation = (
            fireflies.graphics.depth.get_segmentation_from_camera(mitsuba_scene)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if segmentation.max() == 0:
            continue

        seg_test = segmentation.copy()
        seg_test[seg_test == 2] = 1
        seg_test = 1 - seg_test
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_test.astype(np.uint8)
        )

        if n_labels > 3:
            continue

        segmentation_map = segmentation / segmentation.max()
        final = cv2.hconcat([render, segmentation_map])

        cv2.imwrite(
            os.path.join(dataset_path, "train/images/{0:05d}.png".format(count)),
            (render * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(dataset_path, "train/segmentation/{0:05d}.png".format(count)),
            segmentation.astype(np.uint8),
        )
        count += 1
    """

    count = 0
    while count != 500:
        ff_scene.randomize()
        render = mi.render(mitsuba_scene, spp=100)
        render = render_to_numpy(render)
        render = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)

        render = post_processor.post_process(render)

        segmentation = (
            fireflies.graphics.depth.get_segmentation_from_camera(mitsuba_scene)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if segmentation.max() == 0:
            continue

        seg_test = segmentation.copy()
        seg_test[seg_test == 2] = 1
        seg_test = 1 - seg_test
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_test.astype(np.uint8)
        )

        if n_labels > 3:
            continue

        segmentation_map = segmentation / segmentation.max()
        final = cv2.hconcat([render, segmentation_map])

        cv2.imwrite(
            os.path.join(dataset_path, "eval/images/{0:05d}.png".format(count)),
            (render * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(dataset_path, "eval/segmentation/{0:05d}.png".format(count)),
            segmentation.astype(np.uint8),
        )
        count += 1
'''
