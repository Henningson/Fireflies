from typing import List
from enum import Enum
import torch

from torchmetrics.image import PeakSignalNoiseRatio


class EvaluationCriterion:
    def __init__(self, eval_func):
        self._total_error = 0.0
        self._eval_func = eval_func

    def eval(self, input: torch.tensor, ground_truth: torch.tensor):
        error = self._eval_func(input, ground_truth)
        self._total_error += error
        return self._total_error


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
