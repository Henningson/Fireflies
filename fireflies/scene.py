import mitsuba as mi

import torch
import fireflies.entity
import fireflies.utils
import fireflies.emitter
import fireflies.material

from typing import List


class Scene:
    MESH_KEYS = ["mesh", "ply"]
    CAM_KEYS = ["camera", "perspective", "perspectivecamera"]
    PROJ_KEYS = ["projector"]
    MAT_KEYS = ["mat", "bsdf"]
    LIGHT_KEYS = ["light", "spot"]
    TEX_KEYS = ["tex"]

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

        self._mitsuba_params = mitsuba_params

        self.init_from_params(self._mitsuba_params)

    def device(self) -> torch.cuda.device:
        return self._device

    def mesh_at(self, index: int) -> fireflies.entity.Transformable:
        return self._meshes[index]

    def meshes(self) -> fireflies.entity.Transformable:
        return self._meshes

    def get_mesh(self, name: str) -> fireflies.entity.Transformable:
        for mesh in self._meshes:
            if mesh.name() == name:
                return mesh

        return None

    def mesh(self, name: str) -> fireflies.entity.Transformable:
        return self.get_mesh(name)

    def light_at(self, index: int) -> fireflies.entity.Transformable:
        return self._lights[index]

    def lights(self) -> fireflies.entity.Transformable:
        return self._lights

    def get_light(self, name: str) -> fireflies.entity.Transformable:
        for light in self._lights:
            if light.name() == name:
                return light

        return None

    def light(self, name: str) -> fireflies.entity.Transformable:
        return self.get_light(name)

    def material_at(self, index: int) -> fireflies.entity.Transformable:
        return self._materials[index]

    def materials(self) -> fireflies.entity.Transformable:
        return self._materials

    def get_material(self, name: str) -> fireflies.entity.Transformable:
        for mesh in self._materials:
            if mesh.name() == name:
                return mesh

        return None

    def material(self, name: str) -> fireflies.entity.Transformable:
        return self.get_material(name)

    def init_from_params(self, mitsuba_params) -> None:
        # Get all scene keys
        param_keys = [key.split(".")[0] for key in mitsuba_params.keys()]

        # Remove multiples
        param_keys = set(param_keys)
        param_keys = sorted(param_keys)

        for key in param_keys:
            # Check if its a mesh
            if any(MESH_KEY.lower() in key.lower() for MESH_KEY in self.MESH_KEYS):
                self.load_mesh(key)
                continue
            elif any(CAMERA_KEY.lower() in key.lower() for CAMERA_KEY in self.CAM_KEYS):
                self.load_camera(key)
                continue
            elif any(PROJ_KEY.lower() in key.lower() for PROJ_KEY in self.PROJ_KEYS):
                self.load_projector(key)
                continue
            elif any(LIGHT_KEY.lower() in key.lower() for LIGHT_KEY in self.LIGHT_KEYS):
                self.load_light(key)
                continue
            elif any(MAT_KEY.lower() in key.lower() for MAT_KEY in self.MAT_KEYS):
                self.load_material(key)
                continue

    def load_mesh(self, base_key: str):
        # Gotta compute the centroid here, as mitsuba does not have a world transform for meshes
        vertices = torch.tensor(
            self._mitsuba_params[base_key + ".vertex_positions"], device=self._device
        ).reshape(-1, 3)
        centroid = vertices.sum(dim=0, keepdim=True) / vertices.shape[0]

        aligned_vertices = vertices - centroid

        transformable_mesh = fireflies.entity.Mesh(
            base_key, aligned_vertices, self._device
        )
        transformable_mesh.set_centroid(centroid)

        self._meshes.append(transformable_mesh)

    def load_camera(self, base_key: str) -> None:
        to_world = self._mitsuba_params[base_key + ".to_world"].matrix.torch()
        to_world = to_world.squeeze().to(self._device)
        transformable_camera = fireflies.entity.Transformable(base_key, self._device)
        transformable_camera.set_world(to_world)
        transformable_camera.set_randomizable(False)
        self._camera = transformable_camera

    def load_projector(self, base_key: str) -> None:
        to_world = self._mitsuba_params[base_key + ".to_world"].matrix.torch()
        to_world = to_world.squeeze().to(self._device)
        transformable_projector = fireflies.entity.Transformable(base_key, self._device)
        transformable_projector.set_world(to_world)
        transformable_projector.set_randomizable(False)
        self._projector = transformable_projector

    def load_light(self, base_key: str) -> None:
        new_light = fireflies.emitter.Light(base_key, device=self._device)

        if base_key + ".to_world" in self._mitsuba_params.keys():
            to_world = self._mitsuba_params[base_key + ".to_world"].matrix.torch()
            to_world = to_world.squeeze().to(self._device)
            new_light.set_world(to_world)

        light_keys = []
        for key in self._mitsuba_params.keys():
            if base_key in key:
                light_keys.append(key)

        for key in light_keys:
            key_without_base = ".".join(key.split(".")[1:])
            value = self._mitsuba_params[key]

            if type(value) == mi.Transform4f:
                continue

            if isinstance(value, mi.Float) or isinstance(value, float):
                new_light.add_float_key(key_without_base, value, value)
            elif len(value) == 3:
                value = value.torch().squeeze()
                new_light.add_vec3_key(key_without_base, value, value)

        new_light.set_randomizable(False)
        self._lights.append(new_light)

    def load_material(self, base_key: str) -> None:
        new_material = fireflies.material.Material(base_key, device=self._device)

        material_keys = []
        for key in self._mitsuba_params.keys():
            if base_key in key:
                material_keys.append(key)

        for key in material_keys:
            key_without_base = ".".join(key.split(".")[1:])
            value = self._mitsuba_params[key]

            if type(value) == mi.Transform4f:
                continue

            if isinstance(value, mi.Float) or isinstance(value, float):
                new_material.add_float_key(key_without_base, value, value)
            elif len(value) == 3:
                value = value.torch().squeeze()
                new_material.add_vec3_key(key_without_base, value, value)

        new_material.set_randomizable(False)
        self._materials.append(new_material)

    def train(self) -> None:
        # We first randomize all of our objects
        for mesh in self._meshes:
            mesh.train()

        for light in self._lights:
            light.train()

        for material in self._materials:
            material.train()

        if self._camera is not None:
            self._camera.train()

        if self._projector is not None:
            self._projector.train()

    def eval(self) -> None:
        # We first randomize all of our objects
        for mesh in self._meshes:
            mesh.eval()

        for light in self._lights:
            light.eval()

        for material in self._materials:
            material.eval()

        if self._camera is not None:
            self._camera.eval()

        if self._projector is not None:
            self._projector.eval()

    def load_curve(self, path: str, name: str = "Curve") -> None:
        curve = fireflies.utils.importBlenderNurbsObj(path)
        transformable_curve = fireflies.entity.Curve(name, curve, self._device)

        self.curves.append(transformable_curve)

    def update_meshes(self) -> None:
        for mesh in self._meshes:
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

            if light.name() + ".to_world" in self._mitsuba_params.keys():
                self._mitsuba_params[light.name() + ".to_world"] = mi.Transform4f(
                    light.world().tolist()
                )

            float_dict = light.get_randomized_float_attributes()
            vec3_dict = light.get_randomized_vec3_attributes()

            for key, value in float_dict.items():
                joined_key = light.name() + "." + key
                temp_type = type(self._mitsuba_params[joined_key])
                self._mitsuba_params[joined_key] = temp_type(value.item())

            for key, value in vec3_dict.items():
                joined_key = light.name() + "." + key
                temp_type = type(self._mitsuba_params[joined_key])
                self._mitsuba_params[light.name() + "." + key] = temp_type(
                    value.tolist()
                )

    def update_materials(self) -> None:
        for material in self._materials:
            if not material.randomizable():
                continue

            float_dict = material.get_randomized_float_attributes()
            vec3_dict = material.get_randomized_vec3_attributes()

            for key, value in float_dict.items():
                joined_key = material.name() + "." + key
                temp_type = type(self._mitsuba_params[joined_key])
                self._mitsuba_params[joined_key] = temp_type(value.item())

            for key, value in vec3_dict.items():
                joined_key = material.name() + "." + key
                temp_type = type(self._mitsuba_params[joined_key])
                self._mitsuba_params[material.name() + "." + key] = temp_type(
                    value.tolist()
                )

    def randomize_list(self, entity_list: List[fireflies.entity.Transformable]) -> None:
        # First find parent objects, i.e. child is none
        parent_objects = []
        for entity in entity_list:
            if entity.parent() is None:
                parent_objects.append(entity)

        # Now iterate through every parent object and iteratively call each child randomization function
        for entity in parent_objects:
            entity.randomize()

            iterator_child = entity.child()
            while iterator_child is not None:
                iterator_child.randomize()
                iterator_child = iterator_child.child()

    def randomize(self) -> None:
        # We first randomize all of our objects
        self.randomize_list(self._meshes)
        self.randomize_list(self._lights)
        self.randomize_list(self._materials)

        if self._camera is not None:
            self._camera.randomize()

        if self._projector is not None:
            self._projector.randomize()

        # And then copy the updates to the mitsuba parameters
        self.update_meshes()

        if self._camera is not None:
            self.update_camera()

        if self._projector is not None:
            self.update_projector()
        self.update_lights()
        self.update_materials()

        # We finally update the mitsuba scene graph itself
        self._mitsuba_params.update()
