import torch
import numpy as np
from typing import List

import mesh

import fireflies.utils.math
import fireflies.utils.transforms


class ShapeModel(mesh.Mesh):
    def __init__(
        self,
        name: str,
        vertex_data: List[float],
        config: dict,
        device: torch.cuda.device = torch.device("cuda"),
        base_path: str = None,
        sequential_animation: bool = False,
    ):

        self._device = device
        self._name = name

        self.setTranslationBoundaries(config["translation"])
        self.setRotationBoundaries(config["rotation"])
        self.setWorld(config["to_world"])
        self._world = self._world @ fireflies.utils.transforms.toMat4x4(
            fireflies.utils.math.getXTransform(np.pi * 0.5, self._device)
        )
        self._randomized_world = self._world.clone()

        self._randomizable = bool(config["randomizable"])
        self._relative = bool(config["is_relative"])

        self._parent_name = config["parent_name"] if self._relative else None
        # Is loaded in a second step
        self._parent = None
        self._child = None

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])
        self._animated = bool(config["animated"])
        self._sequential_animation = sequential_animation

        self._animation_index = 0

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])
        self._stddev_range = config["stddev_range"]
        self._shape_layer = None
        self._model_params = {}
        self._train = True

    def loadAnimation(self):
        return None

    def modelParmas(self) -> dict:
        return self._model_params

    def setModelParams(self, dict: dict) -> None:
        assert NotImplementedError

    def getVertexData(self):
        assert NotImplementedError
