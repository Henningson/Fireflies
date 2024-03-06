from typing import List
from enum import Enum
import torch
import csv
import os

from torchmetrics.image import PeakSignalNoiseRatio
from pytorch3d.loss import chamfer_distance


def RSME(x, y):
    mse_loss = torch.nn.MSELoss(reduction="mean")
    return torch.sqrt(mse_loss(x, y))


def MAE(x, y):
    l1_loss = torch.nn.L1Loss(reduction="mean")
    return l1_loss(x, y)


def CHAMFER(x, y):
    return chamfer_distance(x, y)


class EvaluationCriterion:
    def __init__(self, eval_func):
        self._total_error = 0.0
        self._errors = []
        self._eval_func = eval_func
        self._num_evals = 0

    def eval(self, _input: torch.tensor, ground_truth: torch.tensor) -> float:
        error = self._eval_func(_input, ground_truth).detach().cpu().item()
        self._errors.append(error)
        self._total_error += error
        self._num_evals += 1

        return error

    def getTotalError(self) -> float:
        return self._total_error

    def getNormalizedError(self) -> float:
        return self._total_error / self._num_evals

    def __str__(self) -> str:
        return f"{self._eval_func.__name__}, Total Error: {self.getTotalError():.5f}, Normalized Error: {self.getNormalizedError():.5f}"

    def reset(self) -> None:
        self.errors = []
        self.total_error = 0.0
        self.num_evals = 0

    def save(self, path, iter) -> None:
        fullpath = os.path.join(path, f"{self._eval_func.__name__}.csv")
        with open(fullpath, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([iter, self.getTotalError(), self.getNormalizedError()])


class ImageCriterion(EvaluationCriterion):
    def __init__(self, eval_func):
        super(ImageCriterion, self).__init__(eval_func)


class Pointcloud2dCriterion(EvaluationCriterion):
    def __init__(self, eval_func):
        super(ImageCriterion, self).__init__(eval_func)


class Pointcloud3dCriterion(EvaluationCriterion):
    def __init__(self, eval_func):
        super(ImageCriterion, self).__init__(eval_func)


class MeshCriterion(EvaluationCriterion):
    def __init__(self, eval_func):
        super(MeshCriterion, self).__init__(eval_func)


class Evaluator:
    def __init__(self, firefly_scene, mitsuba_scene, mitsuba_params):
        self._firefly_scene = firefly_scene
        self._mitsuba_scene = mitsuba_scene
        self._mitsuba_params = mitsuba_params
        self._eval_criteria = []

    def add_eval_criteria(self, criteria: List[EvaluationCriterion]) -> None:
        self._eval_criteria += criteria

    def evaluate(self) -> List[float]:
        evals = []
        for criterium in self._eval_criteria:
            # Get the necessary data here.

            if criterium.type() == EvaluationCriterion.MESH:
                pass
                # Do something
            elif criterium.type() == EvaluationCriterion.POINTCLOUD_2D:
                pass
                # Do something
            elif criterium.type() == EvaluationCriterion.POINTCLOUD_3D:
                pass
                # Do something
            elif criterium.type() == EvaluationCriterion.IMAGE:
                pass
