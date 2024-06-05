import mitsuba as mi

import torch
import fireflies.entity
import fireflies.utils
import fireflies.emitter
import fireflies.material


class Scene:
    MESH_KEYS = ["mesh", "ply"]
    CAM_KEYS = ["camera", "perspective", "perspectivecamera"]
    PROJ_KEYS = ["projector"]
    MAT_KEYS = ["mat, bdsf"]
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

        self._num_updates = 0
        self._sequential_animation = False

        self._mitsuba_params = mitsuba_params

        self.init_from_params(self._mitsuba_params)

    def mesh_at(self, index: int):
        return self._meshes[index]

    def meshes(self):
        return self._meshes

    def get_mesh(self, name: str) -> fireflies.entity.Transformable:
        for mesh in self._meshes:
            if mesh.name() == name:
                return mesh

        return None

    def light_at(self, index: int):
        return self._lights[index]

    def lights(self):
        return self._lights

    def get_light(self, name: str) -> fireflies.entity.Transformable:
        for light in self._lights:
            if light.name() == name:
                return light

        return None

    def init_from_params(self, mitsuba_params):
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
        centroid = torch.linalg.norm(vertices, dim=0, keepdims=True)

        aligned_vertices = vertices - centroid

        world = torch.eye(4)
        world[0, 0] = centroid.squeeze()[0]
        world[1, 1] = centroid.squeeze()[1]
        world[2, 2] = centroid.squeeze()[2]

        transformable_mesh = fireflies.entity.Mesh(
            base_key, aligned_vertices, self._device
        )
        transformable_mesh.set_world(world)

        self._meshes.append(transformable_mesh)

    def load_camera(self, base_key: str) -> None:
        to_world = self._mitsuba_params[base_key + ".to_world"].matrix.torch()
        to_world = to_world.squeeze().to(self._device)
        transformable_camera = fireflies.entity.Transformable(base_key, self._device)
        transformable_camera.set_world(to_world)

        self._camera = transformable_camera

    def load_projector(self, base_key: str) -> None:
        to_world = self._mitsuba_params[base_key + ".to_world"].matrix.torch()
        to_world = to_world.squeeze().to(self._device)
        transformable_projector = fireflies.entity.Transformable(base_key, self._device)
        transformable_projector.set_world(to_world)
        self._projector = transformable_projector

    def load_light(self, base_key: str) -> None:
        new_light = fireflies.emitter.Light(base_key, device=self._device)
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

            if type(value) is float:
                new_light.add_float_key(key_without_base, value, value)
            elif len(value) == 3:
                value = value.torch().squeeze()
                new_light.add_vec3_key(key_without_base, value, value)

        self._lights.append(new_light)

    def load_material(self, base_key: str) -> None:
        new_material = fireflies.material.Material(base_key, device=self._device)

        material_keys = []
        for key in self._mitsuba_params.keys():
            if base_key in key:
                material_keys.append(key)

        for key in material_keys:
            key_without_base = ".".join(key.split(".")[1:])
            value = self.mitsuba_params[key].torch()

            if type(value) is float:
                new_material.add_float_key(key_without_base, value, value)
            elif len(value) == 3:
                value = value.torch().squeeze()
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
        self._mitsuba_params.update()
        self._num_updates += 1
