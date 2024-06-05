import torch
import numpy as np
import flame_pytorch.flame as flame
from typing import List

import fireflies.entity.base as base
import shape

import fireflies.utils.math


class FlameShapeModel(shape.ShapeModel):
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

        flame_config = Namespace(
            **{
                "batch_size": 1,
                "dynamic_landmark_embedding_path": "./Objects/flame_pytorch/model/flame_dynamic_embedding.npy",
                "expression_params": 50,
                "flame_model_path": "./Objects/flame_pytorch/model/generic_model.pkl",
                "num_worker": 4,
                "optimize_eyeballpose": True,
                "optimize_neckpose": True,
                "pose_params": 6,
                "ring_loss_weight": 1.0,
                "ring_margin": 0.5,
                "shape_params": 100,
                "static_landmark_embedding_path": "./Objects/flame_pytorch/model/flame_static_embedding.pkl",
                "use_3D_translation": True,
                "use_face_contour": True,
            }
        )

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])
        self._animated = True
        self._stddev_range = config["stddev_range"]
        self._shape_layer = flame.FLAME(flame_config).to(self._device)
        self._faces = self._shape_layer.faces
        self._pose_params = torch.zeros(1, 6, device=self._device)
        self._expression_params = torch.zeros(1, 50, device=self._device)
        self._shape_params = (
            (torch.rand(1, 100, device=self._device) - 0.5) * 2.0 * self._stddev_range
        )
        self._shape_params[:, 20:] = 0.0

        self._shape_params *= 0.0
        self._invert = False

    def train(self) -> None:
        base.Transformable.train(self)

    def eval(self) -> None:
        base.Transformable.eval(self)

    def loadAnimation(self):
        return None

    def modelParams(self) -> dict:
        return self._shape_params

    def shapeParams(self) -> torch.tensor:
        return self._shape_params

    def expressionParams(self) -> torch.tensor:
        return self._expression_params

    def poseParams(self) -> torch.tensor:
        return self._pose_params

    def randomize(self) -> None:
        if self._shape_params[0, 0] > 2.0:
            self._invert = True
        self._shape_params = self._shape_params + (0.05 if not self._invert else -0.05)
        self._shape_params[:, 20:] = 0.0

        self._randomized_world = (
            self.sampleTranslation() @ self.sampleRotation() @ self.sampleScale()
        )

    def getVertexData(self):
        if not self._animated:
            return self._vertices, self._shape_layer.faces

        vertices, _ = self._shape_layer(
            self._shape_params, self._expression_params, self._pose_params
        )
        vertices = vertices[0]

        vertices = fireflies.utils.transforms.transform_points(
            vertices,
            self.world()
            @ fireflies.utils.transforms.toMat4x4(
                fireflies.utils.math.getXTransform(np.pi * 0.5, self._device)
            ),
        )

        return vertices, self._shape_layer.faces
