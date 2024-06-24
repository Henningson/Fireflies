import torch
import math
import random
from typing import List
import fireflies.sampling.base as base


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (
        torch.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            dim=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
    )


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(
            shape, (frequency * res[0], frequency * res[1])
        )
        frequency *= 2
        amplitude *= persistence
    return noise


class NoiseTextureLerpSampler(base.Sampler):
    def __init__(
        self,
        color_a: torch.tensor,
        color_b: torch.tensor,
        texture_shape: List[int],
        eval_step_size: float = 0.01,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> None:
        super(NoiseTextureLerpSampler, self).__init__(
            torch.tensor([0.0], device=device),
            torch.tensor([1.0], device=device),
            eval_step_size,
            device,
        )
        self._color_a = color_a
        self._color_b = color_b
        self._texture_shape = texture_shape

    def sample_train(self) -> torch.tensor:
        i = 2 ** random.randint(1, 6)
        octaves = random.randint(1, 4)
        persistence = random.uniform(0.1, 2.0)
        tex = rand_perlin_2d_octaves(
            self._texture_shape, res=(i, i), octaves=octaves, persistence=persistence
        ).to(self._device)
        tex = (tex - tex.min()) / (tex.max() - tex.min())

        col_a = torch.ones_like(tex).unsqueeze(0).repeat(
            3, 1, 1
        ) * self._color_a.unsqueeze(-1).unsqueeze(-1)
        col_b = torch.ones_like(tex).unsqueeze(0).repeat(
            3, 1, 1
        ) * self._color_b.unsqueeze(-1).unsqueeze(-1)
        tex = tex.unsqueeze(0).repeat(3, 1, 1)
        return torch.lerp(col_a, col_b, tex)

    # To lazy to implement it right now.
    def sample_eval(self) -> torch.tensor:
        return self.sample_train()
