import os
import torch
import random
import numpy as np
import pywavefront
from typing import List

import transformable

import fireflies.utils.math
import fireflies.utils.transforms


class Mesh(transformable.transformable):
    def __init__(
        self,
        name: str,
        vertex_data: List[float],
        config: dict,
        device: torch.cuda.device = torch.device("cuda"),
        base_path: str = None,
        sequential_animation: bool = True,
    ):
        transformable.transformable.__init__(self, name, config, device)
        self._base_path = base_path

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])

        self._animated = bool(config["animated"])
        self._sequential_animation = sequential_animation
        self._animation_index = 0

    def animated(self) -> bool:
        return self._animated

    def train(self) -> None:
        transformable.transformable.train(self)
        self._sequential_animation = False

        if self._animated:
            self.loadAnimation(self._base_path, self._name)

    def eval(self) -> None:
        transformable.transformable.eval(self)
        self._sequential_animation = True
        if self._animated:
            eval_path = f"{self._name}_eval"
            self.loadAnimation(self._base_path, eval_path)

    def convertToLocal(self, vertices: torch.tensor) -> List[List[float]]:
        vertices = fireflies.utils.transforms.transform_points(
            vertices,
            fireflies.utils.transforms.toMat4x4(
                fireflies.utils.math.getXTransform(np.pi * 0.5, self._device)
            ),
        )
        return vertices

    def setFaces(self, faces: List[float]) -> None:
        self._faces = (
            torch.tensor(faces, device=self._device) if faces is not None else faces
        )

    def setVertices(self, vertices: List[float]) -> None:
        self._vertices = torch.tensor(vertices, device=self._device).reshape(-1, 3)
        self._vertices = self.convertToLocal(self._vertices)

    def setScaleBoundaries(self, scale: dict) -> None:
        self.min_scale = torch.tensor(
            [scale["min_x"], scale["min_y"], scale["min_z"]], device=self._device
        )
        self.max_scale = torch.tensor(
            [scale["max_x"], scale["max_y"], scale["max_z"]], device=self._device
        )

    def sampleScale(self) -> torch.tensor:
        scaleMatrix = torch.eye(4, device=self._device)
        random_scale = fireflies.utils.math.randomBetweenTensors(
            self.min_scale, self.max_scale
        )

        scaleMatrix[0, 0] = random_scale[0]
        scaleMatrix[1, 1] = random_scale[1]
        scaleMatrix[2, 2] = random_scale[2]
        return scaleMatrix

    def randomize(self) -> None:
        self._randomized_world = (
            self.sampleTranslation() @ self.sampleRotation() @ self.sampleScale()
        )

    def faces(self) -> torch.tensor:
        return self._faces

    def getVertexData(self) -> torch.tensor:
        # Sample Animations
        temp_vertex = self.sampleAnimation() if self._animated else self._vertices

        # Transform by world transform
        temp_vertex = fireflies.utils.transforms.transform_points(
            temp_vertex, self.world()
        )

        # parent = self._parent
        # while parent:
        #     temp_vertex = transforms.transform_points(temp_vertex, parent.world())

        return temp_vertex, None

    def loadAnimation(self, base_path, obj_name):
        self._vertex_offsets = []
        self._face_data = []
        for file in sorted(os.listdir(os.path.join(base_path, obj_name + "/"))):
            if file.endswith(".obj"):
                obj_path = os.path.join(base_path, obj_name, file)

                obj = pywavefront.Wavefront(obj_path, collect_faces=True)

                self._vertex_offsets.append(
                    torch.tensor(obj.vertices, device=self._device).reshape(-1, 3)
                )
                self._face_data.append(
                    torch.tensor(obj.mesh_list[0].faces, device=self._device).flatten()
                )

    def next_anim_step(self) -> None:
        self._animation_index += 1

    def sampleAnimation(self):
        if not self._animated:
            return self._vertices, None

        index = 0
        if self._sequential_animation:
            index = self._animation_index % len(self._vertex_offsets)
        else:
            num_anim_frames = len(self._vertex_offsets)
            index = random.randint(0, num_anim_frames - 1)

        return self._vertex_offsets[index]
