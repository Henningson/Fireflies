import torch
import fireflies.sampling.base as base
import random


class UniformIntegerSampler(base.Sampler):
    def __init__(
        self,
        min_integer: int,
        max_integer: int,
        eval_step_size: int = 1,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> None:
        """
        Will generate samples from the integer interval given by [min_integer, ..., max_integer) similar to how range() is defined in python.
        """
        super(UniformIntegerSampler, self).__init__(min, max, eval_step_size, device)
        self._current_step = 0

    def sample_eval(self) -> int:
        sample = self._current_step
        self._current_step += self._eval_step_size

        if self._current_step >= self._max_range:
            self._current_step = self._min_range

        return sample

    def sample_train(self) -> int:
        return random.randint(0, self._max_range - 1)