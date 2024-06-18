import os
import torch
import random
import pywavefront

import fireflies.entity.base as base
import fireflies.utils.math
import fireflies.sampling


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

        ones = torch.ones(3, device=self._device)
        self._scale_sampler = fireflies.sampling.UniformSampler(
            ones.clone(), ones.clone()
        )

        self._animated = False

        self._anim_data_train = None
        self._anim_data_eval = None
        self._animation_func = None
        self._animation_sampler = None

    def set_scale_sampler(self, sampler: fireflies.sampling.Sampler) -> None:
        self._scale_sampler = sampler

    def scale_x(self, min_scale: float, max_scale: float) -> None:
        self._randomizable = True
        self.update_index_from_sampler(self._scale_sampler, min_scale, max_scale, 0)

    def scale_y(self, min_scale: float, max_scale: float) -> None:
        self._randomizable = True
        self.update_index_from_sampler(self._scale_sampler, min_scale, max_scale, 1)

    def scale_z(self, min_scale: float, max_scale: float) -> None:
        self._randomizable = True
        self.update_index_from_sampler(self._scale_sampler, min_scale, max_scale, 2)

    def scale(self, min: torch.tensor, max: torch.tensor) -> None:
        self._randomizable = True
        self._scale_sampler.set_sample_interval(
            min.to(self._device), max.to(self._device)
        )

    def set_scale_sampler(self, sampler: fireflies.sampling.Sampler) -> None:
        self._scale_sampler = sampler

    def animated(self) -> bool:
        return self._animated

    def add_animation(self, animation_data: torch.tensor) -> None:
        self._animation_vertices = animation_data.to(self._device)
        self._animated = True
        self._randomizable = True

    def add_animation_func(self, func, min_range, max_range) -> None:
        self._animation_func = func
        self._animation_sampler = fireflies.sampling.UniformSampler(
            min_range, max_range, device=self._device
        )
        self._animated = True
        self._randomizable = True

    def add_train_animation_from_obj(
        self, path: str, min: int = None, max: int = None
    ) -> None:
        self._anim_data_train = self.load_animation(path)

        if self._animation_sampler:
            self._animation_sampler.set_train_interval(
                0 if min is None else 0,
                self._anim_data_train.shape[0] if max is None else max,
            )
            return

        self._animation_sampler = fireflies.sampling.AnimationSampler(0, 1, 0, 1)
        self._animation_sampler.set_train_interval(
            0 if min is None else 0,
            self._anim_data_train.shape[0] if max is None else max,
        )

    def add_eval_animation_from_obj(
        self, path: str, min: int = None, max: int = None
    ) -> None:
        self._anim_data_eval = self.load_animation(path)

        if self._animation_sampler:
            self._animation_sampler.set_eval_interval(
                0 if min is None else 0,
                self._anim_data_eval.shape[0] if max is None else max,
            )
            return

        self._animation_sampler = fireflies.sampling.AnimationSampler(0, 1, 0, 1)
        self._animation_sampler.set_eval_interval(
            0 if min is None else 0,
            self._anim_data_eval.shape[0] if max is None else max,
        )

    def train(self) -> None:
        super(Mesh, self).train()
        self._scale_sampler.train()

        if self._animation_sampler:
            self._animation_sampler.train()

    def eval(self) -> None:
        super(Mesh, self).eval()
        self._scale_sampler.eval()

        if self._animation_sampler:
            self._animation_sampler.eval()

    def set_faces(self, faces: torch.tensor) -> None:
        self._faces = faces.to(self._device)

    def set_vertices(self, vertices: torch.tensor) -> None:
        self._vertices = vertices.to(self._device)

    def sample_scale(self) -> torch.tensor:
        scale_matrix = torch.eye(4, device=self._device)

        random_scale = self._scale_sampler.sample()

        scale_matrix[0, 0] = random_scale[0]
        scale_matrix[1, 1] = random_scale[1]
        scale_matrix[2, 2] = random_scale[2]
        return scale_matrix

    def randomize(self) -> None:
        if not self.randomizable():
            return

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

    def load_animation(self, path: str) -> torch.tensor:
        animation_data = []
        for file in sorted(os.listdir(path)):
            if file.endswith(".obj"):
                obj_path = os.path.join(path, file)

                obj = pywavefront.Wavefront(obj_path, collect_faces=True)

                animation_data.append(
                    torch.tensor(obj.vertices, device=self._device).reshape(-1, 3)
                )

        return torch.stack(animation_data)

    def sample_animation(self):
        if not self._animated:
            return self._vertices

        # Can either be an integer or a float, depending if we loaded meshes or defined an animation function
        time_sample = self._animation_sampler.sample()
        if self._animation_func is not None:
            return self._animation_func(self._vertices, time_sample)
        elif self._anim_data_train is not None and self._anim_data_eval is not None:
            return (
                self._anim_data_train[time_sample]
                if self.train()
                else self._anim_data_eval[time_sample]
            )

        return None
