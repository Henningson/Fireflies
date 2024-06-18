import cv2
import mitsuba as mi
import numpy as np
import kornia

mi.set_variant("cuda_ad_rgb")

import torch
import fireflies
import fireflies.sampling
import fireflies.projection.laser


def render_to_opencv(render):
    render = torch.clamp(render.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
    return (render * 255).astype(np.uint8)


if __name__ == "__main__":
    path = "examples/scenes/vocalfold/vocalfold.xml"

    mitsuba_scene = mi.load_file(path)
    mitsuba_params = mi.traverse(mitsuba_scene)
    ff_scene = fireflies.Scene(mitsuba_params)

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
    laser_rays = fireflies.projection.Laser.generate_blue_noise_rays(
        500, 500, 18 * 18, K_PROJECTOR, device=ff_scene.device()
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
    larynx_mesh = ff_scene.mesh("mesh-Larynx")
    larynx_mesh.scale_x(0.8, 1.2)
    larynx_mesh.rotate_y(-0.1, 0.1)

    vocalfold_mesh.scale_x(0.5, 2.0)
    vocalfold_mesh.rotate_y(-0.25, 0.25)

    material = ff_scene.material("mat-Default OBJ")
    scalar_to_vec3_sampler = fireflies.sampling.UniformScalarToVec3Sampler(
        1.0, 20.0, device=ff_scene.device()
    )

    light = ff_scene.light("emit-Spot")
    light.add_vec3_sampler("intensity.value", scalar_to_vec3_sampler)

    material.add_vec3_key(
        "brdf_0.base_color.value",
        torch.tensor([0.8, 0.14, 0.34], device=ff_scene.device()),
        torch.tensor([0.85, 0.5, 0.44], device=ff_scene.device()),
    )
    material.add_float_key("brdf_0.specular", 0.0, 0.75)

    # ff_scene._camera.rotate_y(-0.25, 0.25)
    # ff_scene._projector.setParent(ff_scene._camera)
    # ff_scene._projector._randomizable = True
    # ff_scene._projector.rotate_x(3.141, 3.141)

    ff_scene.train()
    for i in range(1000):
        ff_scene.randomize()
        render = mi.render(mitsuba_scene, spp=100)
        render = render_to_opencv(render)

        if i % 2 == 0:
            render = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY).astype(int)
            noise = np.random.normal(np.zeros_like(render), np.ones_like(render) * 0.05)
            noise *= 255
            noise = noise.astype(np.int)
            render += noise
            render[render > 255] = 255
            render[render < 0] = 0
            render = render.astype(np.uint8)

        cv2.imwrite("vf_renderings/{0:05d}.png".format(i), render)
        # cv2.imshow("A", render)
        # cv2.waitKey(0)
