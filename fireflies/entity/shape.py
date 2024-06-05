import torch
import numpy as np
from typing import List

from mesh import Mesh
import fireflies.utils.math


@NotImplementedError
class ShapeModel(Mesh):
    def __init__(
        self,
        name: str,
        vertex_data: torch.tensor,
        face_data: torch.tensor,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        super(Mesh, self).__init__(name, vertex_data, face_data, device)
        self._device = device
        self._name = name

    def load_animation(self):
        return None

    def get_model_params(self) -> dict:
        return self._model_params

    def set_model_params(self, dict: dict) -> None:
        assert NotImplementedError

    def get_vertices(self):
        assert NotImplementedError
