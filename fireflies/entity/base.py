import torch
from typing import List

import fireflies.utils.math
import fireflies.sampling


class Transformable:
    def __init__(
        self,
        name: str,
        device: torch.cuda.device = torch.device("cuda"),
    ):

        self._device: torch.cuda.device = device
        self._name: str = name

        self._randomizable: bool = False

        self._parent = None
        self._child = None

        self._train = True

        self._float_attributes = {}
        self._randomized_float_attributes = {}

        self._vec3_attributes = {}
        self._randomized_vec3_attributes = {}

        zeros = torch.zeros(3, device=self._device)
        self._rotation_sampler = fireflies.sampling.UniformSampler(
            zeros.clone(), zeros.clone()
        )

        self._translation_sampler = fireflies.sampling.UniformSampler(
            zeros.clone(), zeros.clone()
        )

        self._world = torch.eye(4, device=self._device)
        self._randomized_world = torch.eye(4, device=self._device)

        self._centroid_mat = torch.zeros((4, 4), device=self._device)

        self._eval_delta = 0.01
        self._num_updates = 0

    def randomizable(self) -> bool:
        return self._randomizable

    def set_centroid(self, centroid: torch.tensor) -> None:
        self._centroid_mat[0, 3] = centroid.squeeze()[0]
        self._centroid_mat[1, 3] = centroid.squeeze()[1]
        self._centroid_mat[2, 3] = centroid.squeeze()[2]

    def set_randomizable(self, randomizable: bool) -> None:
        self._randomizable = randomizable

    def get_randomized_vec3_attributes(self) -> dict:
        return self._randomized_vec3_attributes

    def get_randomized_float_attributes(self) -> dict:
        return self._randomized_float_attributes

    def vec3_attributes(self) -> dict:
        return self._vec3_attributes

    def float_attributes(self) -> dict:
        return self._float_attributes

    def add_float_sampler(self, key: str, sampler: fireflies.sampling.Sampler) -> None:
        self._randomizable = True
        self._float_attributes[key] = sampler

    def add_float_key(self, key: str, min: float, max: float) -> None:
        """Transforms float key into a Uniform Sampler"""
        self._randomizable = True
        self._float_attributes[key] = fireflies.sampling.UniformSampler(
            min, max, device=self._device
        )

    def add_vec3_key(self, key: str, min: torch.tensor, max: torch.tensor) -> None:
        """Transforms vec3 into Uniform Sampler"""
        self._randomizable = True
        self._vec3_attributes[key] = fireflies.sampling.UniformSampler(
            min, max, device=self._device
        )

    def add_vec3_sampler(self, key: str, sampler: fireflies.sampling.Sampler) -> None:
        self._randomizable = True
        self._vec3_attributes[key] = sampler

    def parent(self):
        return self._parent

    def child(self):
        return self._child

    def name(self):
        return self._name

    def train(self) -> None:
        self._train = True
        self._translation_sampler.train()
        self._rotation_sampler.train()

        for sampler in self._float_attributes.values():
            sampler.train()

        for sampler in self._vec3_attributes.values():
            sampler.train()

    def eval(self) -> None:
        self._train = False
        self._translation_sampler.eval()
        self._rotation_sampler.eval()

        for sampler in self._float_attributes.values():
            sampler.eval()

        for sampler in self._vec3_attributes.values():
            sampler.eval()

    def set_world(self, _origin: torch.tensor) -> None:
        self._world = _origin
        self._randomized_world = self._world.clone()

    def setParent(self, parent) -> None:
        self._parent = parent
        parent.setChild(self)

    def setChild(self, child) -> None:
        self._child = child

    def set_rotation_sampler(self, sampler: fireflies.sampling.Sampler) -> None:
        self._rotation_sampler = sampler

    def set_translation_sampler(self, sampler: fireflies.sampling.Sampler) -> None:
        self._translation_sampler = sampler

    def update_index_from_sampler(self, sampler, min, max, index) -> None:
        sampler_min = sampler.get_min()
        sampler_max = sampler.get_max()

        sampler_min[index] = min
        sampler_max[index] = max

    def rotate_x(self, min_rot: float, max_rot: float) -> None:
        """Convenience function for Uniform Sampler"""
        self._randomizable = True
        self.update_index_from_sampler(self._rotation_sampler, min_rot, max_rot, 0)

    def rotate_y(self, min_rot: float, max_rot: float) -> None:
        """Convenience function for Uniform Sampler"""
        self._randomizable = True
        self.update_index_from_sampler(self._rotation_sampler, min_rot, max_rot, 1)

    def rotate_z(self, min_rot: float, max_rot: float) -> None:
        """Convenience function for Uniform Sampler"""
        self._randomizable = True
        self.update_index_from_sampler(self._rotation_sampler, min_rot, max_rot, 2)

    def rotate(self, min: torch.tensor, max: torch.tensor) -> None:
        """Convenience function for Uniform Sampler"""
        self._randomizable = True
        self._rotation_sampler.set_sample_interval(
            min.to(self._device), max.to(self._device)
        )

    def translate_x(self, min_translation: float, max_translation: float) -> None:
        self._randomizable = True
        self.update_index_from_sampler(
            self._translation_sampler, min_translation, max_translation, 0
        )

    def translate_y(self, min_translation: float, max_translation: float) -> None:
        self._randomizable = True
        self.update_index_from_sampler(
            self._translation_sampler, min_translation, max_translation, 1
        )

    def translate_z(self, min_translation: float, max_translation: float) -> None:
        self._randomizable = True
        self.update_index_from_sampler(
            self._translation_sampler, min_translation, max_translation, 2
        )

    def translate(self, min: torch.tensor, max: torch.tensor) -> None:
        self._randomizable = True
        self._translation_sampler.set_sample_interval(
            min.to(self._device), max.to(self._device)
        )

    def sample_rotation(self) -> torch.tensor:
        self._sampled_rotation = self._rotation_sampler.sample()

        zMat = fireflies.utils.math.getPitchTransform(
            self._sampled_rotation[2], self._device
        )
        yMat = fireflies.utils.math.getYawTransform(
            self._sampled_rotation[1], self._device
        )
        xMat = fireflies.utils.math.getRollTransform(
            self._sampled_rotation[0], self._device
        )

        return fireflies.utils.math.toMat4x4(zMat @ yMat @ xMat)

    def sample_translation(self) -> torch.tensor:
        translation = torch.eye(4, device=self._device)

        self._random_translation = self._translation_sampler.sample()

        translation[0, 3] = self._random_translation[0]
        translation[1, 3] = self._random_translation[1]
        translation[2, 3] = self._random_translation[2]
        self._last_translation = translation
        return translation

    def randomize(self) -> None:
        if not self.randomizable():
            return

        self._randomized_world = (
            (self.sample_translation() + self._centroid_mat)
            @ self.sample_rotation()
            @ self._world
        )

        for key, sampler in self._float_attributes.items():
            self._randomized_float_attributes[key] = sampler.sample()

        for key, sampler in self._vec3_attributes.items():
            self._randomized_vec3_attributes[key] = sampler.sample()

    def relative(self) -> None:
        return self._parent is not None

    def world(self) -> torch.tensor:
        # If no parent exists, just return the current translation
        if self._parent is None:
            return self._randomized_world.clone()

        return self._parent.world() @ self._randomized_world

    def nonRandomizedWorld(self) -> torch.tensor:
        if self._parent is None:
            return self._world

        return self._parent.nonRandomizedWorld() @ self._world
