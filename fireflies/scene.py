import os

from bs4 import BeautifulSoup
from pathlib import Path
import os
import mitsuba as mi

import utils.math

import torch
import numpy as np

import graphics
import entity
import projection
import utils
import light


class scene:
    MESH_KEYS = ["mesh", "ply"]
    CAMERA_KEYS = ["camera", "perspective", "perspectivecamera"]
    PROJECTOR_KEYS = ["projector"]
    MAT_KEYS = ["mat, bdsf"]
    LIGHT_KEYS = ["light", "spot"]
    TEXTURE_KEYS = ["tex"]

    def __init__(
        self,
        mitsuba_params,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        # Here, only objects are saved, that have a "randomizable"-tag inside the yaml file.
        self._meshes = []
        self._projector = None
        self._camera = None
        self._lights = []
        self._curves = []

        self._transformables = []

        self._device = device

        self._num_updates = 0
        self._sequential_animation = False

        self.mitsuba_params = mitsuba_params

        self.initFromParams(self.mitsuba_params)

    def initFromParams(self, mitsuba_params):
        # Get all scene keys
        param_keys = [key.split(".")[0] for key in mitsuba_params.keys()]

        # Remove multiples
        param_keys = set(param_keys)
        param_keys = sorted(param_keys)

        for key in param_keys:
            # Check if its a mesh
            if any(
                key.lowercase() in MESH_KEY.lowercase() for MESH_KEY in self.MESH_KEYS
            ):
                self.load_mesh(key)
                continue
            elif any(
                key.lowercase() in CAMERA_KEY.lowercase()
                for CAMERA_KEY in self.CAMERA_KEYS
            ):
                self.load_camera(key)
                continue
            elif any(
                key.lowercase() in PROJECTOR_KEY.lowercase()
                for PROJECTOR_KEY in self.PROJECTOR_KEYS
            ):
                self.load_projector(key)
                continue
            elif any(
                key.lowercase() in LIGHT_KEY.lowercase()
                for LIGHT_KEY in self.LIGHT_KEYS
            ):
                self.load_light(key)
                continue
            elif any(
                key.lowercase() in MATERIAL_KEY.lowercase()
                for MATERIAL_KEY in self.MATERIAL_KEYS
            ):
                self.load_material(key)
                continue

    def load_mesh(self, base_key: str):
        # Gotta compute the centroid here, as mitsuba does not have a world transform for meshes
        vertices = torch.tensor(
            self.mitsuba_params[base_key + ".vertex_positions"], device=self._device
        ).reshape(-1, 3)
        centroid = torch.linalg.norm(vertices, dim=0, keepdims=True)

        aligned_vertices = vertices - centroid

        world = torch.eye(4)
        world[0, 0] = centroid.squeeze()[0]
        world[1, 1] = centroid.squeeze()[1]
        world[2, 2] = centroid.squeeze()[2]

        transformable_mesh = entity.Mesh(base_key, aligned_vertices, self._device)
        transformable_mesh.set_world(world)

        self.meshes.append(transformable_mesh)

    def load_camera(self, base_key: str) -> None:
        camera_world = torch.tensor(
            self.mitsuba_params[base_key + ".to_world"], device=self._device
        ).reshape(4, 4)
        transformable_camera = entity.Transformable(base_key, None, self._device)
        transformable_camera.set_world(camera_world)
        self._camera = transformable_camera

    def load_projector(self, base_key: str) -> None:
        camera_world = torch.tensor(
            self.mitsuba_params[base_key + ".to_world"], device=self._device
        ).reshape(4, 4)
        transformable_projector = entity.Transformable(base_key, None, self._device)
        transformable_projector.set_world(camera_world)
        self._projector = transformable_projector

    def load_light(self, base_key: str) -> None:
        new_light = light.Light(base_key, device=self._device)
        to_world = torch.tensor(
            self.mitsuba_params[base_key + ".to_world"], device=self._device
        ).reshape(4, 4)

        new_light.set_world(to_world)

        light_keys = [base_key in key for key in self.mitsuba_params.keys()]
        for key in light_keys:
            key_without_base = key.split(".")[1:].join()
            value = self.mitsuba_params[key].torch()

            if len(value) == 1:
                new_light.add_float_key(key_without_base, value, value)
            elif len(value) == 3:
                new_light.add_vec3_key(key_without_base, value, value)

        self._lights.append(new_light)

    def load_material(self, base_key: str) -> None:
        new_light = light.Light(base_key, device=self._device)
        to_world = torch.tensor(
            self.mitsuba_params[base_key + ".to_world"], device=self._device
        ).reshape(4, 4)

        new_light.set_world(to_world)

        light_keys = [base_key in key for key in self.mitsuba_params.keys()]
        for key in light_keys:
            key_without_base = key.split(".")[1:].join()
            value = self.mitsuba_params[key].torch()

            if len(value) == 1:
                new_light.add_float_key(key_without_base, value, value)
            elif len(value) == 3:
                new_light.add_vec3_key(key_without_base, value, value)

        self._lights.append(new_light)

    def train(self) -> None:
        # Set all objects to train mode
        for transformable in self._parent_transformables:
            transformable.train()

            iterator_child = transformable.child()
            while iterator_child is not None:
                iterator_child.train()
                iterator_child = iterator_child.child()

    def eval(self) -> None:
        # Set all objects to eval mode
        for transformable in self._parent_transformables:
            transformable.eval()

            iterator_child = transformable.child()
            while iterator_child is not None:
                iterator_child.eval()
                iterator_child = iterator_child.child()

    def getMitsubaXML(self, path):
        data = None
        with open(path, "r") as f:
            data = f.read()
        return BeautifulSoup(data, "xml")

    def loadProjector(self):
        sensor_name = "Projector"
        sensor_yaml_path = os.path.join(self.firefly_path, sensor_name + ".yaml")
        sensor_config = utils.read_config_yaml(sensor_yaml_path)

        self.projector = entity.base.Transformable(
            sensor_name, sensor_config, self._device
        )
        self._parent_transformables.append(self.projector)

    def loadCameras(self):
        sensor_name = "Camera"
        sensor_yaml_path = os.path.join(self.firefly_path, sensor_name + ".yaml")
        sensor_config = utils.read_config_yaml(sensor_yaml_path)

        self.camera = entity.base.Transformable(
            sensor_name, sensor_config, self._device
        )
        self._parent_transformables.append(self.camera)

    def loadLights(self):
        # TODO: Implement me
        a = 1
        pass

    def loadMeshes(self):
        meshes = self.mi_xml.find_all("shape")
        param_mesh = "PLYMesh"

        count = 0
        for mesh in meshes:
            if mesh.find("emitter") is not None:
                continue

            temp_param_mesh = param_mesh
            if count > 0:
                temp_param_mesh += "_{0}".format(count)

            mesh_name = self.getMeshName(mesh)
            mesh_yaml_path = os.path.join(self.firefly_path, mesh_name + ".yaml")
            mesh_config = utils.read_config_yaml(mesh_yaml_path)

            if not mesh_config["randomizable"]:
                continue

            # Object is randomizable => Create randomizable object, and connect it to the mitsuba parameter.
            if "is_flame" in mesh_config and mesh_config["is_flame"]:
                self.meshes[temp_param_mesh] = entity.flame.FlameShapeModel(
                    name=mesh_name,
                    config=mesh_config,
                    vertex_data=self.scene_params[
                        temp_param_mesh + ".vertex_positions"
                    ],
                    sequential_animation=self._sequential_animation,
                    base_path=self.firefly_path,
                    device=self._device,
                )
            else:
                self.meshes[temp_param_mesh] = entity.mesh.Mesh(
                    name=mesh_name,
                    config=mesh_config,
                    vertex_data=self.scene_params[
                        temp_param_mesh + ".vertex_positions"
                    ],
                    sequential_animation=self._sequential_animation,
                    base_path=self.firefly_path,
                    device=self._device,
                )

            self._parent_transformables.append(self.meshes[temp_param_mesh])
            count += 1

    def loadCurves(self):
        nurbs_files = [
            f
            for f in os.listdir(self.firefly_path)
            if os.path.isfile(os.path.join(self.firefly_path, f))
            and "path" in f.lower()
        ]

        for nurbs_path in nurbs_files:
            yaml_path = os.path.join(self.firefly_path, nurbs_path)
            config = utils.read_config_yaml(yaml_path)

            object_name = os.path.splitext(nurbs_path)[0]
            curve = utils.importBlenderNurbsObj(
                os.path.join(self.firefly_path, object_name, object_name + ".obj")
            )
            transformable_curve = entity.curve.Curve(
                object_name, curve, config, self._device
            )

            self.curves.append(transformable_curve)
            self._parent_transformables.append(transformable_curve)

    def connectParents(self):
        for a in self._parent_transformables:
            if not a.relative():
                continue

            for b in self._parent_transformables:
                if a == b:
                    continue

                # CHECK FOR RELATIVES HERE
                if a.parentName() == b.name():
                    a.setParent(b)

    def cleanParentTransformables(self) -> None:
        self._parent_transformables = [
            transformable
            for transformable in self._parent_transformables
            if transformable.parent() is None
        ]

    def initScene(self) -> None:
        self.loadMeshes()
        self.loadCameras()
        self.loadProjector()
        self.loadLights()
        self.loadCurves()

        # Connect relative objects
        self.connectParents()

        # Filter non-parent objects from transformable list
        self.cleanParentTransformables()

    def updateMeshes(self) -> None:
        for key, mesh in self.meshes.items():
            rand_verts, faces = mesh.getVertexData()
            self.scene_params[key + ".vertex_positions"] = mi.Float32(
                rand_verts.flatten()
            )

            if faces is not None:
                self.scene_params[key + ".faces"] = mi.UInt32(faces.flatten())

            if mesh.animated():
                mesh.next_anim_step()

    def updateCamera(self) -> None:
        if self.camera is None:
            return

        # TODO: Remove key
        key = "PerspectiveCamera"

        # Couldn't find a better way to get this torch tensor into mitsuba Transform4f
        worldMatrix = self.camera.world()
        worldMatrix[0:3, 0:3] = worldMatrix[0:3, 0:3] @ utils.math.getYTransform(
            np.pi, self._device
        )
        # worldMatrix[0:3, 0:3] = worldMatrix[0:3, 0:3]
        self.scene_params[key + ".to_world"] = mi.Transform4f(worldMatrix.tolist())

    def updateProjector(self) -> None:
        if self.projector is None:
            return

        # TODO: Remove key
        key = "Projector"
        worldMatrix = self.projector.world()
        worldMatrix[0:3, 0:3] = worldMatrix[0:3, 0:3] @ utils.math.getYTransform(
            np.pi, self._device
        )

        # TODO: Is there a better way here?
        # Couldn't find a better way to get this torch tensor into mitsuba Transform4f
        self.scene_params[key + ".to_world"] = mi.Transform4f(worldMatrix.tolist())
        self.scene_params["PerspectiveCamera_1.to_world"] = mi.Transform4f(
            worldMatrix.tolist()
        )

    def updateLights(self) -> None:
        # TODO: Implement me
        return None

    def randomize(self) -> None:
        # We first randomize all of our objects
        for transformable in self._parent_transformables:
            transformable.randomize()

            iterator_child = transformable.child()
            while iterator_child is not None:
                iterator_child.randomize()
                iterator_child = iterator_child.child()

        # And then copy the updates to the mitsuba parameters
        self.updateMeshes()
        self.updateCamera()
        self.updateProjector()
        self.updateLights()

        # We finally update the mitsuba scene graph itself
        self.scene_params.update()
        self._num_updates += 1

    def getMeshName(self, mesh) -> str:
        for child in mesh.find_all("string"):
            if child.has_attr("name") and child.attrs["name"] == "filename":
                return Path(child.attrs["value"]).stem

        return None


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import cv2
    import graphics.rasterization
    from argparse import Namespace

    base_path = "scenes/Vocalfold/"

    config = utils.read_config_yaml(os.path.join(base_path, "config.yml"))
    config = Namespace(**config)

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    mitsuba_params["PerspectiveCamera.film.size"] //= config.downscale_factor
    mitsuba_params["PerspectiveCamera_1.film.size"] //= config.downscale_factor

    mitsuba_params["PerspectiveCamera_1.to_world"] = mitsuba_params[
        "PerspectiveCamera.to_world"
    ]

    mitsuba_params["Projector.to_world"] = mitsuba_params[
        "PerspectiveCamera_1.to_world"
    ]

    mitsuba_params.update()

    sigma = torch.tensor([config.sigma], device="cuda")
    texture_size = torch.tensor(mitsuba_scene.sensors()[1].film().size(), device="cuda")

    firefly_scene = Scene(mitsuba_params, base_path, sequential_animation=True)
    firefly_scene.eval()

    laser_init = utils.laser_estimation.initialize_laser(
        mitsuba_scene,
        mitsuba_params,
        firefly_scene,
        config,
        config.pattern_initialization,
        device="cuda",
    )
    points = laser_init.projectRaysToNDC()[:, 0:2]

    """
    colors = [(0.0, 0.1921, 0.4156), (0, 0.69, 0.314)]  # R -> G -> B
    fig = plt.figure(frameon=False)
    fig.set_size_inches(16 / 16 * 10, 9 / 16 * 10)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.scatter(
        points.detach().cpu().numpy()[:, 0],
        points.detach().cpu().numpy()[:, 1],
        s=60.0 * 2.5,
        color=colors[0],
    )
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    cv2.imshow("Sexy Tex", img_plot[:, :, [2, 1, 0]])
    """
    # for i in range(1, 1000):

    camera_sensor = mitsuba_scene.sensors()[0]
    camera_x_fov = mitsuba_params["PerspectiveCamera.x_fov"]
    camera_near_clip = mitsuba_params["PerspectiveCamera.near_clip"]
    camera_far_clip = mitsuba_params["PerspectiveCamera.far_clip"]

    K_CAMERA = mi.perspective_projection(
        camera_sensor.film().size(),
        camera_sensor.film().crop_size(),
        camera_sensor.film().crop_offset(),
        camera_x_fov,
        camera_near_clip,
        camera_far_clip,
    ).matrix.torch()[0]

    texture_init = graphics.rasterization.rasterize_points(
        points, config.sigma, texture_size
    )
    texture_init = graphics.rasterization.softor(texture_init)

    '''
    laser_init._rays.requires_grad = True

    optim = torch.optim.Adam([{"params": laser_init._rays, "lr": 0.001}])

    for i in range(1000):
        firefly_scene.randomize()
        optim.zero_grad()

        laser_s = laser_init.originPerRay() + 1.0 * laser_init.rays()
        laser_e = laser_init.originPerRay() + 1.1 * laser_init.rays()

        CAMERA_WORLD = mitsuba_params["PerspectiveCamera.to_world"].matrix.torch()[0]
        laser_s = transforms.transform_points(laser_s, CAMERA_WORLD.inverse()).squeeze()
        laser_s = transforms.transform_points(laser_s, K_CAMERA).squeeze()[:, :2]

        laser_e = transforms.transform_points(laser_e, CAMERA_WORLD.inverse()).squeeze()
        laser_e = transforms.transform_points(laser_e, K_CAMERA).squeeze()[:, :2]

        lines = torch.concat(
            [laser_s.unsqueeze(-1), laser_e.unsqueeze(-1)], dim=-1
        ).transpose(1, 2)
        lines_copy = lines.clone()
        line_render = rasterization.rasterize_lines(lines, config.sigma, texture_size)
        line_softor = rasterization.softor(line_render)

        loss = torch.nn.L1Loss()(line_softor, line_render.sum(dim=0))
        loss.backward()
        optim.step()

        with torch.no_grad():
            line_vis = line_softor.detach().cpu().numpy() * 255
            line_vis = line_vis.astype(np.uint8).transpose()
            cv2.imshow("Line Render", line_vis)

            """
            colors = [(0.0, 0.1921, 0.4156), (0, 0.69, 0.314)]  # R -> G -> B
            fig = plt.figure(frameon=False)
            fig.set_size_inches(10, 10)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_axis_off()
            ax.set_aspect(aspect="equal")
            fig.add_axes(ax)

            lines_copy = lines_copy.transpose(1, 2).detach().cpu().numpy()
            for j in range(lines_copy.shape[0]):
                ax.plot(
                    lines_copy[j, 0, :],
                    lines_copy[j, 1, :],
                    c=colors[0],
                    linewidth=9.5,
                    solid_capstyle="round",
                )  # c=colors[0], linewidth=60)

            fig.canvas.draw()
            img_plot = np.array(fig.canvas.renderer.buffer_rgba())
            img_plot = np.transpose(
                cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR), [1, 0, 2]
            )
            img_plot = cv2.flip(img_plot, 1)
            cv2.imshow("Testrender", img_plot)
            cv2.imwrite(f"EpipolarLinesOptimization/{i:05d}.png", img_plot)
            plt.close()
            """
            laser_tex = rasterization.softor(
                laser_init.generateTexture(config.sigma, texture_size)
            )
            laser_tex = laser_tex.detach().cpu().numpy()
            cv2.imshow("LASERTEX", laser_tex)
            cv2.waitKey(1)
            laser_init.clamp_to_fov(clamp_val=0.99)
            laser_init.normalize_rays()

    exit()
    '''

    # sensor_size = torch.tensor(global_scene.sensors()[0].film().size(), device=DEVICE)

    # texture_init = torch.ones(texture_init.shape, device=texture_init.device)
    # texture_init = torch.flipud(texture_init)
    # print(i)
    # cv2.imshow("Wat", texture_init.detach().cpu().numpy())
    # cv2.waitKey(1)

    texture_init = torch.stack(
        [torch.zeros_like(texture_init), texture_init, torch.zeros_like(texture_init)],
        dim=-1,
    )
    mitsuba_params["tex.data"] = texture_init

    # firefly_scene.randomize()
    # render_im = mi.render(mitsuba_scene)

    for i in tqdm(range(300)):
        firefly_scene.randomize()

        render_im = mi.render(mitsuba_scene, spp=10)
        render_im = torch.clamp(render_im.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
        render_im *= 255
        render_im = render_im.astype(np.uint8)
        cv2.imshow("a", render_im)
        cv2.waitKey(10)
