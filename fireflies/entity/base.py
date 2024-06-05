import torch
from typing import List

import fireflies.utils.math


class Transformable:
    def __init__(
        self,
        name: str,
        config: dict,
        device: torch.cuda.device = torch.device("cuda"),
    ):

        self._device = device
        self._name = name

        if config is not None:
            self.setTranslationBoundaries(config["translation"])
            self.setRotationBoundaries(config["rotation"])
            self.setWorld(config["to_world"])

        self._randomizable = (
            bool(config["randomizable"]) if config is not None else False
        )
        self._relative = bool(config["is_relative"]) if config is not None else False

        self._parent_name = config["parent_name"] if self._relative else None
        # Is loaded in a second step
        self._parent = None
        self._child = None
        self._train = True
        self.xRot = 0.0
        self.yRot = 0.0
        self.zRot = 0.0

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

    def parentName(self) -> str:
        return self._parent_name

    def setWorld(self, _origin: List[List[float]]) -> None:
        self._world = torch.tensor(_origin, device=self._device)
        self._randomized_world = self._world.clone()

    def setParent(self, parent) -> None:
        self._parent = parent
        parent.setChild(self)

    def setChild(self, child) -> None:
        self._child = child

    def setRotationBoundaries(self, rotation: dict) -> None:
        self.rot_min_x = rotation["min_x"]
        self.rot_max_x = rotation["max_x"]
        self.rot_min_y = rotation["min_y"]
        self.rot_max_y = rotation["max_y"]
        self.rot_min_z = rotation["min_z"]
        self.rot_max_z = rotation["max_z"]

    def setTranslationBoundaries(self, translation: dict) -> None:
        self.min_translation = torch.tensor(
            [translation["min_x"], translation["min_y"], translation["min_z"]],
            device=self._device,
        )
        self.max_translation = torch.tensor(
            [translation["max_x"], translation["max_y"], translation["max_z"]],
            device=self._device,
        )

    def sampleRotation(self) -> torch.tensor:
        self.test = 0.0

        self.xRot = fireflies.utils.math.uniformBetweenValues(
            self.rot_min_x, self.rot_max_x
        )
        self.yRot = fireflies.utils.math.uniformBetweenValues(
            self.rot_min_y, self.rot_max_y
        )

        if self._child is not None:
            self.zRot = self.zRot + (6.282 / 100)
        else:
            self.zRot = fireflies.utils.math.uniformBetweenValues(
                self.rot_min_z, self.rot_max_z
            )

        zMat = fireflies.utils.math.getPitchTransform(self.zRot, self._device)
        yMat = fireflies.utils.math.getYawTransform(self.yRot, self._device)
        xMat = fireflies.utils.math.getRollTransform(self.xRot, self._device)

        return fireflies.utils.transforms.toMat4x4(zMat @ yMat @ xMat)

    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        self.random_translation = fireflies.utils.randomBetweenTensors(
            self.min_translation, self.max_translation
        )

        translationMatrix[0, 3] = self.random_translation[0]
        translationMatrix[1, 3] = -self.random_translation[2]
        translationMatrix[2, 3] = self.random_translation[1]
        self._last_translation = translationMatrix
        return translationMatrix

    def randomize(self) -> None:
        self._randomized_world = (
            self.sampleTranslation() @ self.sampleRotation() @ self._world
        )

    def relative(self) -> None:
        return self._relative

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
