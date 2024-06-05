import torch
from typing import List

import fireflies.utils.math


class Material:
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

    def add_float_key(self, key, value_min: float, value_max: float) -> None:
        self._float_attributes[key] = (value_min, value_max)

    def add_vec3_key(self, key, min_vec: torch.tensor, max_vec: torch.tensor) -> None:
        self._vec3_attributes[key] = (min_vec, max_vec)

    def name(self):
        return self._name

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False

    def randomize(self) -> None:
        for key, value in self._float_attributes.items():
            self._randomized_float_attributes[key] = (
                fireflies.utils.math.uniformBetweenValues(value[0], value[1])
            )

        for key, value in self._vec3_attributes.items():
            self._randomized_float_attributes[key] = (
                fireflies.utils.math.randomBetweenTensors(value[0], value[1])
            )
