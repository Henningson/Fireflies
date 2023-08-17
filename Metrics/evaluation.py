from typing import List
import torch


class EvaluationCriterion:
    def __init__(self):



class Evaluator:
    def __init__(self, firefly_scene, mitsuba_scene, mitsuba_params):
        self.firefly_scene = firefly_scene
        self.mitsuba_scene = mitsuba_scene
        self.mitsuba_params = mitsuba_params
        self.eval_criteria = []

    def add_eval_criteria(criteria: List[EvaluationCriterion]) -> None:


    def evaluate() -> torch.tensor: