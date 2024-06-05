import torch
from typing import List

import fireflies.utils.math


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

        self._rotation_min = torch.zeros(3, dtype=float, device=self._device)
        self._rotation_max = torch.zeros(3, dtype=float, device=self._device)

        self._translation_min = torch.zeros(3, dtype=float, device=self._device)
        self._translation_max = torch.zeros(3, dtype=float, device=self._device)

        self._world = None
        self._randomized_world = None

    def randomizable(self) -> bool:
        return self._randomizable

    def get_randomized_vec3_attributes(self) -> dict:
        return self._randomized_vec3_attributes

    def get_randomized_float_attributes(self) -> dict:
        return self._randomized_float_attributes

    def add_float_key(self, key, value_min: float, value_max: float) -> None:
        self._randomizable = True
        self._float_attributes[key] = (value_min, value_max)

    def add_vec3_key(self, key, min_vec: torch.tensor, max_vec: torch.tensor) -> None:
        self._randomizable = True
        self._vec3_attributes[key] = (min_vec, max_vec)

    def parent(self):
        return self._parent

    def child(self):
        return self._child

    def name(self):
        return self._name

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False

    def set_world(self, _origin: torch.tensor) -> None:
        self._world = _origin
        self._randomized_world = self._world.clone()

    def setParent(self, parent) -> None:
        self._parent = parent
        parent.setChild(self)

    def setChild(self, child) -> None:
        self._child = child

    def rotate_x(self, min_rot: float, max_rot: float) -> None:
        self._randomizable = True
        self._rotation_min[0] = min_rot
        self._rotation_max[0] = max_rot

    def rotate_y(self, min_rot: float, max_rot: float) -> None:
        self._randomizable = True
        self._rotation_min[1] = min_rot
        self._rotation_max[1] = max_rot

    def rotate_z(self, min_rot: float, max_rot: float) -> None:
        self._randomizable = True
        self._rotation_min[2] = min_rot
        self._rotation_max[2] = max_rot

    def rotate(self, min: torch.tensor, max: torch.tensor) -> None:
        self._randomizable = True
        self._rotation_min = min.to(self._device)
        self._rotation_max = max.to(self._device)

    def translate_x(self, min_translation: float, max_translation: float) -> None:
        self._randomizable = True
        self._translation_min[0] = min_translation
        self._translation_max[0] = max_translation

    def translate_y(self, min_translation: float, max_translation: float) -> None:
        self._randomizable = True
        self._translation_min[1] = min_translation
        self._translation_max[1] = max_translation

    def translate_z(self, min_translation: float, max_translation: float) -> None:
        self._randomizable = True
        self._translation_min[2] = min_translation
        self._translation_max[2] = max_translation

    def translate(self, min: torch.tensor, max: torch.tensor) -> None:
        self._randomizable = True
        self._translation_min = min.to(self._device)
        self._translation_max = max.to(self._device)

    def sample_rotation(self) -> torch.tensor:
        self._sampled_rotation = fireflies.utils.math.randomBetweenTensors(
            self._rotation_min, self._rotation_max
        )

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
        self.random_translation = fireflies.utils.math.randomBetweenTensors(
            self._translation_min, self._translation_max
        )

        translation[0, 3] = self.random_translation[0]
        translation[1, 3] = self.random_translation[1]
        translation[2, 3] = self.random_translation[2]
        self._last_translation = translation
        return translation

    def sample_scale(self) -> torch.tensor:
        scale = torch.eye(4, device=self._device)
        self._random_scale = fireflies.utils.math.randomBetweenTensors(
            self._translation_min, self._translation_max
        )

        scale[0, 0] = self._random_scale[0]
        scale[1, 1] = self._random_scale[1]
        scale[2, 2] = self._random_scale[2]
        self._last_scale = scale
        return scale

    def randomize(self) -> None:
        self._randomized_world = (
            (self.sample_translation() * self.sample_scale())
            @ self.sample_rotation()
            @ self._world
        )

        for key, value in self._float_attributes.items():
            self._randomized_float_attributes[key] = (
                fireflies.utils.math.uniformBetweenValues(value[0], value[1])
            )

        for key, value in self._vec3_attributes.items():
            self._randomized_float_attributes[key] = (
                fireflies.utils.math.randomBetweenTensors(value[0], value[1])
            )

    def relative(self) -> None:
        return self._parent is not None

    def world(self) -> torch.tensor:
        # If no parent exists, just return the current translation
        if self._parent is None:
            temp = self._randomized_world.clone()
            return temp

        return self._parent.world() @ self._randomized_world

    def nonRandomizedWorld(self) -> torch.tensor:
        if self._parent is None:
            return self._world

        return self._parent.nonRandomizedWorld() @ self._world
