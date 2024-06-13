import torch


class Sampler:
    def __init__(
        self,
        min: torch.tensor,
        max: torch.tensor,
        eval_step_size: float = 0.01,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> None:
        self._device = device
        self._min_range = min
        self._max_range = max
        self._train = True

        self._eval_step_size = eval_step_size
        self._current_step = self._min_range

    def set_sample_interval(self, min: torch.tensor, max: torch.tensor) -> None:
        self._min_range = min
        self._max_range = max

    def set_sample_max(self, max: torch.tensor) -> None:
        self._max_range = max

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False

    def sample(self) -> torch.tensor:
        if self._sample_func:
            return self._sample_func

        return None

    @NotImplementedError
    def sample_train(self) -> torch.tensor:
        return None

    def sample_eval(self) -> torch.tensor:
        sample = self._current_step
        self._current_step += self._eval_step_size

        if self._current_step > self._max_range:
            self._current_step = self._min_range

        return sample

    def set_sample_func(self, func) -> None:
        self._sample_func = func
