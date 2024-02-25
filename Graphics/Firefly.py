from bs4 import BeautifulSoup
from pathlib import Path
import os
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import drjit as dr
import Objects.entity as entity
import Utils.utils as utils
import torch
import Objects.laser as laser
import Objects.Camera as Camera
import Objects.Transformable as Transformable
import Utils.transforms as transforms
import numpy as np
import Utils.math as utils_math
import Graphics.LaserEstimation as LaserEstimation
import math


class Scene:
    def __init__(
        self,
        scene_params,
        base_path: str,
        sequential_animation: bool = False,
        steps_per_frame: int = 1,
        device: torch.cuda.device = torch.device("cuda"),
    ):

        self.mi_xml = self.getMitsubaXML(os.path.join(base_path, "scene.xml"))
        self.firefly_path = os.path.join(base_path, "Firefly")
        self.scene_params = scene_params

        self.base_path = base_path

        # Here, only objects are saved, that have a "randomizable"-tag inside the yaml file.
        self.meshes = {}
        self.projector = None
        self.camera = None
        self.lights = {}
        self.curves = []

        self._parent_transformables = []

        self._device = device

        self._num_updates = 0
        self._sequential_animation = sequential_animation
        self._steps_per_frame = steps_per_frame

        self.initScene()

    def getMitsubaXML(self, path):
        data = None
        with open(path, "r") as f:
            data = f.read()
        return BeautifulSoup(data, "xml")

    def loadProjector(self):
        sensor_name = "Projector"
        sensor_yaml_path = os.path.join(self.firefly_path, sensor_name + ".yaml")
        sensor_config = utils.read_config_yaml(sensor_yaml_path)

        self.projector = Transformable.Transformable(
            sensor_name, sensor_config, self._device
        )
        self._parent_transformables.append(self.projector)

    def loadCameras(self):
        sensor_name = "Camera"
        sensor_yaml_path = os.path.join(self.firefly_path, sensor_name + ".yaml")
        sensor_config = utils.read_config_yaml(sensor_yaml_path)

        self.camera = Transformable.Transformable(
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
                self.meshes[temp_param_mesh] = Transformable.FlameShapeModel(
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
                self.meshes[temp_param_mesh] = Transformable.Mesh(
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
            transformable_curve = Transformable.Curve(
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
                if self._num_updates % self._steps_per_frame == 0:
                    mesh.next_anim_step()

    def updateCamera(self) -> None:
        if self.camera is None:
            return

        # TODO: Remove key
        key = "PerspectiveCamera"

        # Couldn't find a better way to get this torch tensor into mitsuba Transform4f
        worldMatrix = self.camera.world()
        worldMatrix[0:3, 0:3] = worldMatrix[0:3, 0:3] @ utils_math.getYTransform(
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
        worldMatrix[0:3, 0:3] = worldMatrix[0:3, 0:3] @ utils_math.getYTransform(
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


def generate_epipolar_shadow(scene):
    pass


@dr.wrap_ad(source="torch", target="drjit")
def render(texture, spp=256, seed=1):
    global_params.update()
    return mi.render(
        global_scene, global_params, spp=spp, seed=seed, seed_grad=seed + 1
    )


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import cv2
    import Graphics.rasterization as rasterization
    from argparse import Namespace

    base_path = "scenes/FLAME SHAPE/"

    config = utils.read_config_yaml(os.path.join(base_path, "config.yml"))
    config = Namespace(**config)

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    mitsuba_params["PerspectiveCamera.film.size"] //= 2
    mitsuba_params["PerspectiveCamera_1.film.size"] //= 2
    mitsuba_params["Projector.to_world"] = mitsuba_params[
        "PerspectiveCamera_1.to_world"
    ]
    mitsuba_params.update()

    sigma = torch.tensor([config.sigma], device="cuda")
    texture_size = torch.tensor(mitsuba_scene.sensors()[1].film().size(), device="cuda")

    firefly_scene = Scene(mitsuba_params, base_path, sequential_animation=True)
    firefly_scene.randomize()

    laser_init = LaserEstimation.initialize_laser(
        mitsuba_scene,
        mitsuba_params,
        firefly_scene,
        config,
        config.pattern_initialization,
        device="cuda",
    )
    points = laser_init.projectRaysToNDC()[:, 0:2]

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

    texture_init = rasterization.rasterize_points(points, sigma, texture_size)
    texture_init = rasterization.softor(texture_init)
    # texture_init = torch.flipud(texture_init)

    cv2.imshow("Wat", texture_init.detach().cpu().numpy())
    cv2.waitKey(1)
    mitsuba_params["tex.data"] = texture_init.unsqueeze(-1)

    # firefly_scene.randomize()
    # render_im = mi.render(mitsuba_scene)

    for i in tqdm(range(100000)):
        firefly_scene.randomize()

        render_im = mi.render(mitsuba_scene, spp=config.spp)
        render_im = torch.clamp(render_im.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
        cv2.imshow("Render", render_im)
        cv2.waitKey(1)
