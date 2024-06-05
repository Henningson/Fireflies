import os

from pathlib import Path
import os
import mitsuba as mi

import utils.math

import torch
import numpy as np

import graphics
import entity
import utils
import emitter
import material


class Scene:
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
        self._materials = []

        self._transformables = []

        self._device = device

        self._num_updates = 0
        self._sequential_animation = False

        self._mitsuba_params = mitsuba_params

        self.init_from_params(self._mitsuba_params)

    def init_from_params(self, mitsuba_params):
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
                for CAMERA_KEY in self._camera_KEYS
            ):
                self.load_camera(key)
                continue
            elif any(
                key.lowercase() in PROJECTOR_KEY.lowercase()
                for PROJECTOR_KEY in self._projector_KEYS
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
            self._mitsuba_params[base_key + ".vertex_positions"], device=self._device
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
            self._mitsuba_params[base_key + ".to_world"], device=self._device
        ).reshape(4, 4)
        transformable_camera = entity.Transformable(base_key, None, self._device)
        transformable_camera.set_world(camera_world)
        self._camera = transformable_camera

    def load_projector(self, base_key: str) -> None:
        camera_world = torch.tensor(
            self._mitsuba_params[base_key + ".to_world"], device=self._device
        ).reshape(4, 4)
        transformable_projector = entity.Transformable(base_key, None, self._device)
        transformable_projector.set_world(camera_world)
        self._projector = transformable_projector

    def load_light(self, base_key: str) -> None:
        new_light = emitter.Light(base_key, device=self._device)
        to_world = torch.tensor(
            self._mitsuba_params[base_key + ".to_world"], device=self._device
        ).reshape(4, 4)

        new_light.set_world(to_world)

        light_keys = [base_key in key for key in self._mitsuba_params.keys()]
        for key in light_keys:
            key_without_base = key.split(".")[1:].join()
            value = self._mitsuba_params[key].torch()

            if len(value) == 1:
                new_light.add_float_key(key_without_base, value, value)
            elif len(value) == 3:
                new_light.add_vec3_key(key_without_base, value, value)

        self._lights.append(new_light)

    def load_material(self, base_key: str) -> None:
        new_material = material.Material(base_key, device=self._device)

        material_keys = [base_key in key for key in self.mitsuba_params.keys()]
        for key in material_keys:
            key_without_base = key.split(".")[1:].join()
            value = self.mitsuba_params[key].torch()

            if len(value) == 1:
                new_material.add_float_key(key_without_base, value, value)
            elif len(value) == 3:
                new_material.add_vec3_key(key_without_base, value, value)

        self._materials.append(new_material)

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

    def load_curve(self, path: str, name: str = "Curve"):
        curve = utils.importBlenderNurbsObj(path)
        transformable_curve = entity.Curve(name, curve, self._device)

        self.curves.append(transformable_curve)

    def update_meshes(self) -> None:
        for mesh in self.meshes:
            if not mesh.randomizable():
                continue

            vertex_data = mesh.get_randomized_vertices()
            self._mitsuba_params[mesh.name() + ".vertex_positions"] = mi.Float32(
                vertex_data.flatten()
            )

    def update_camera(self) -> None:
        if not self._camera.randomizable():
            return

        self._mitsuba_params[self._camera.name() + ".to_world"] = mi.Transform4f(
            self._camera.world().tolist()
        )

    def update_projector(self) -> None:
        if not self._projector.randomizable():
            return

        self._mitsuba_params[self._projector.name() + ".to_world"] = mi.Transform4f(
            self._projector.world().tolist()
        )

    def update_lights(self) -> None:
        for light in self._lights:
            if not light.randomizable():
                continue

            self.scene_params[key + ".to_world"] = mi.Transform4f(
                light.world().tolist()
            )

            float_dict = light.get_randomized_float_attributes()
            vec3_dict = light.get_randomized_vec3_attributes()

            for key, value in float_dict.items():
                self._mitsuba_params[light.name() + "." + key] = value

            for key, value in vec3_dict.items():
                self._mitsuba_params[light.name() + "." + key] = value

    def update_materials(self) -> None:
        for material in self._materials:
            if not material.randomizable():
                continue

            float_dict = material.get_randomized_float_attributes()
            vec3_dict = material.get_randomized_vec3_attributes()

            for key, value in float_dict.items():
                self._mitsuba_params[material.name() + "." + key] = value

            for key, value in vec3_dict.items():
                self._mitsuba_params[material.name() + "." + key] = value

    def randomize(self) -> None:
        # We first randomize all of our objects
        for mesh in self._meshes:
            mesh.randomize()

        for light in self._lights:
            light.randomize()

        for material in self._materials:
            material.randomize()

        self._camera.randomize()
        self._projector.randomize()

        # And then copy the updates to the mitsuba parameters
        self.update_meshes()
        self.update_camera()
        self.update_projector()
        self.update_lights()
        self.update_materials()

        # We finally update the mitsuba scene graph itself
        self.scene_params.update()
        self._num_updates += 1


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import cv2
    import graphics.rasterization
    from argparse import Namespace

    base_path = "Old/scenes/Vocalfold"

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    fireflies_scene = Scene(mitsuba_params)

    for i in tqdm(range(300)):
        fireflies_scene.randomize()

        render_im = mi.render(mitsuba_scene, spp=10)
        render_im = torch.clamp(render_im.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
        render_im *= 255
        render_im = render_im.astype(np.uint8)
        cv2.imshow("a", render_im)
        cv2.waitKey(10)
