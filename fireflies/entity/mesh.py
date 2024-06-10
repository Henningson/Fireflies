import os
import torch
import random
import pywavefront

import fireflies.entity.base as base
import fireflies.utils.math


class Mesh(base.Transformable):
    def __init__(
        self,
        name: str,
        vertex_data: torch.tensor,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        super(Mesh, self).__init__(name, device)

        self._vertices = vertex_data.to(self._device)
        self._vertices_animation = None

        self._scale_min = torch.ones(3, device=self._device)
        self._scale_max = torch.ones(3, device=self._device)

        self._animated = False

        self._animation_data = None
        self._animation_index = 0

        self._animation_func = None
        self._animation_time = 0.0
        self._time_delta = 0.01

    def scale_x(self, min_scale: float, max_scale: float) -> None:
        self._randomizable = True
        self._scale_min[0] = min_scale
        self._scale_max[0] = max_scale

    def scale_y(self, min_scale: float, max_scale: float) -> None:
        self._randomizable = True
        self._scale_min[1] = min_scale
        self._scale_max[1] = max_scale

    def scale_z(self, min_scale: float, max_scale: float) -> None:
        self._randomizable = True
        self._scale_min[2] = min_scale
        self._scale_max[2] = max_scale

    def scale(self, min: torch.tensor, max: torch.tensor) -> None:
        self._randomizable = True
        self._scale_min = min.to(self._device)
        self._scale_max = max.to(self._device)

    def animated(self) -> bool:
        return self._animated

    def add_animation(self, animation_data: torch.tensor) -> None:
        self._animation_vertices = animation_data.to(self._device)
        self._animated = True
        self._randomizable = True

    def add_animation_func(self, func):
        self._animation_func = func
        self._animated = True
        self._randomizable = True

    def train(self) -> None:
        self._train = True
        self._sequential_animation = False

    def eval(self) -> None:
        self._train = False
        self._sequential_animation = True

    def set_faces(self, faces: torch.tensor) -> None:
        self._faces = faces.to(self._device)

    def set_vertices(self, vertices: torch.tensor) -> None:
        self._vertices = vertices.to(self._device)

    def sample_scale(self) -> torch.tensor:
        scale_matrix = torch.eye(4, device=self._device)

        random_scale = None
        if self._train:
            random_scale = fireflies.utils.math.randomBetweenTensors(
                self._scale_min, self._scale_max
            )
        else:
            random_scale = self._scale_min + (
                self._num_updates % 100
            ) * self._eval_delta * (self._scale_max - self._scale_min)

        scale_matrix[0, 0] = random_scale[0]
        scale_matrix[1, 1] = random_scale[1]
        scale_matrix[2, 2] = random_scale[2]
        return scale_matrix

    def randomize(self) -> None:
        if not self.randomizable():
            return

        if not self._train:
            self._num_updates += 1

        self._randomized_world = (
            (self.sample_translation() + self._centroid_mat)
            @ self.sample_rotation()
            @ self.sample_scale()
            @ self._world
        )

    def faces(self) -> torch.tensor:
        return self._faces

    def get_vertices(self) -> torch.tensor:
        return self._vertices

    def get_randomized_vertices(self) -> torch.tensor:
        # Sample Animations
        temp_vertex = self.sample_animation() if self._animated else self._vertices

        # Transform by world transform
        temp_vertex = fireflies.utils.math.transform_points(temp_vertex, self.world())

        return temp_vertex

    def load_animation(self, path: str) -> None:
        animation_data = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".obj"):
                obj_path = os.path.join(path, file)

                obj = pywavefront.Wavefront(obj_path, collect_faces=True)

                animation_data.append(
                    torch.tensor(obj.vertices, device=self._device).reshape(-1, 3)
                )

        self.add_animation(torch.stack(animation_data))
        self._animated = True

    def sample_animation(self):
        if not self._animated:
            return self._vertices

        if self._animation_func is not None:
            time_sample = 0.0
            if self._train:
                time_sample = fireflies.utils.math.uniformBetweenValues(0.0, 1.0)
            else:
                time_sample = self._animation_time
                self._animation_time += self._time_delta
            return self._animation_func(self._vertices, time_sample)
        elif self._animation_data is not None:
            index = 0
            if self._train:
                num_anim_frames = len(self._vertex_offsets)
                index = random.randint(0, num_anim_frames - 1)
            else:
                index = self._animation_index
                self._animation_index = (
                    self._animation_index + 1
                ) % self._animation_vertices.shape[0]

            return self._animation_vertices[index]

        return None
