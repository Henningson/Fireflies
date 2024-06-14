import torch
import fireflies.sampling.base as base
import fireflies.utils.math


class UniformSampler(base.Sampler):
    def __init__(
        self,
        min: torch.tensor,
        max: torch.tensor,
        eval_step_size: float = 0.01,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> None:
        super(UniformSampler, self).__init__(min, max, eval_step_size, device)

    def sample_train(self) -> torch.tensor:
        return fireflies.utils.math.randomBetweenTensors(
            self._min_range, self._max_range
        )
