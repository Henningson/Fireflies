import torch
import random

from geomdl import NURBS
from typing import List

import transformable
import fireflies.utils.math
import fireflies.utils.transforms


class curve(transformable.transformable):
    count = 0.0

    def fromObj(path):
        # TODO: Implement me
        pass

    def __init__(
        self,
        name: str,
        curve: NURBS.Curve,
        config: dict,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        transformable.transformable.__init__(self, name, config, device)

        self._curve = curve
        self._curve.ctrlpts = self.convertToLocal(self._curve.ctrlpts)
        self.curve_epsilon = 0.05

        self.curve_delta = self.curve_epsilon

        self._interp_steps = 1000
        self._interp_delta = 1.0 / self._interp_steps

        self.eval_interval_start = 0.05

    def train(self) -> None:
        self._train = True
        self._continuous = False

    def eval(self) -> None:
        self._train = False
        self._continuous = True
        self._curve_delta = self.eval_interval_start

    def convertToLocal(self, controlpoints: List[List[float]]) -> List[List[float]]:
        return controlpoints

    def setContinuous(self, continuous: bool) -> None:
        self._continuous = continuous

    def sampleRotation(self) -> torch.tensor:
        t = self.curve_delta
        t_new = self.curve_delta + 0.001

        t_new = torch.tensor(self._curve.evaluate_single(t_new), device=self._device)
        t = torch.tensor(self._curve.evaluate_single(t), device=self._device)

        curve_direction = t_new - t
        curve_direction[0] *= -1.0
        curve_direction[2] *= -1.0

        # curve_normal = torch.tensor(self._curve.normal(t), device=self._device)
        # curve_direction /= torch.linalg.norm(curve_direction)
        # curve_normal /= torch.linalg.norm(curve_normal)

        # camera_up_vector = torch.tensor([0, 0, 1], device=self._device)

        camera_direction = torch.tensor([0.0, 1.0, 0.0], device=self._device)
        return fireflies.utils.transforms.toMat4x4(
            fireflies.utils.math.rotation_matrix_from_vectors(
                camera_direction, curve_direction
            )
        )

    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        translation = self._curve.evaluate_single(self.curve_delta)

        translationMatrix[0, 3] = -translation[0]
        translationMatrix[1, 3] = translation[1]
        translationMatrix[2, 3] = -translation[2]

        return translationMatrix

    def randomize(self) -> None:
        if self._train:
            self.curve_delta = random.uniform(
                0 + self.curve_epsilon, self.eval_interval_start
            )
        else:
            self.curve_delta += self._interp_delta

            if self.curve_delta > 1.0 - self.curve_epsilon:
                self.curve_delta = self.eval_interval_start

        self._randomized_world = (
            self.sampleTranslation() @ self.sampleRotation() @ self._world
        )
