import torch
import fireflies.sampling.base as base
import fireflies.utils.math


class UniformScalarToVec3Sampler(base.Sampler):
    def __init__(
        self,
        min: torch.tensor,
        max: torch.tensor,
        eval_step_size: float = 0.01,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> None:
        super(UniformScalarToVec3Sampler, self).__init__(
            min, max, eval_step_size, device
        )

    def sample_train(self) -> torch.tensor:
        scalar_value = fireflies.utils.math.randomBetweenTensors(
            self._min_range, self._max_range
        )
        return torch.tensor(
            [scalar_value, scalar_value, scalar_value], device=self._device
        )

    def sample_eval(self) -> torch.tensor:
        if (self._min_range == self._max_range).all():
            return torch.tensor(
                [self._min_range, self._min_range, self._min_range], device=self._device
            )

        sample = self._current_step
        self._current_step += self._eval_step_size

        if (self._current_step > self._max_range).any():
            self._current_step = self._min_range

        return torch.tensor([sample, sample, sample], device=self._device)
