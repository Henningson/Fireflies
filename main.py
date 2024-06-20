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


def render_to_numpy(render):
    render = torch.clamp(render.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
    return render


if __name__ == "__main__":
    path = "examples/scenes/vocalfold_aov/vocalfold.xml"

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

    vocalfold_mesh = ff_scene.mesh("mesh-VocalFold")
    vocalfold_mesh.scale_x(0.75, 3.0)
    vocalfold_mesh.rotate_y(-0.25, 0.25)
    vocalfold_mesh.add_train_animation_from_obj("examples/scenes/vocalfold_aov/train/")
    vocalfold_mesh.add_eval_animation_from_obj("examples/scenes/vocalfold_aov/test/")

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
    larynx_mesh.scale_x(0.7, 1.2)
    larynx_mesh.rotate_y(-0.1, 0.1)

    material = ff_scene.material("mat-Default OBJ")
    material.add_vec3_key(
        "brdf_0.base_color.value",
        torch.tensor([0.8, 0.14, 0.34], device=ff_scene.device()),
        torch.tensor([0.85, 0.5, 0.44], device=ff_scene.device()),
    )
    material.add_float_key("brdf_0.specular", 0.0, 0.75)

    scalar_to_vec3_sampler = fireflies.sampling.UniformScalarToVec3Sampler(
        1.0, 20.0, device=ff_scene.device()
    )
    light = ff_scene.light("emit-Spot")
    light.add_vec3_sampler("intensity.value", scalar_to_vec3_sampler)

    ff_scene._camera.translate_x(-0.3, 0.3)
    ff_scene._camera.translate_z(0.0, 0.5)

    post_process_funcs = [
        fireflies.postprocessing.WhiteNoise(0.0, 0.05, 0.5),
        fireflies.postprocessing.GaussianBlur((3, 3), (5, 5), 0.5),
    ]
    post_processor = fireflies.postprocessing.PostProcessor(post_process_funcs)

    ff_scene.train()
    for i in range(1000):
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

        segmentation_map = segmentation / segmentation.max()
        final = cv2.hconcat([render, segmentation_map])
        cv2.imshow("a", final)
        cv2.waitKey(0)

        segmentation_plane = (segmentation == 1) * 1
        a = 1
