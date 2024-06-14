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
        self._min_range = min.clone()
        self._max_range = max.clone()
        self._train = True

        self._eval_step_size = eval_step_size
        self._current_step = self._min_range.clone()

    def set_sample_interval(self, min: torch.tensor, max: torch.tensor) -> None:
        self._min_range = min.clone()
        self._max_range = max.clone()

    def get_min(self) -> torch.tensor:
        return self._min_range

    def get_max(self) -> torch.tensor:
        return self._max_range

    def set_sample_max(self, max: torch.tensor) -> None:
        self._max_range = max.clone()

    def set_sample_min(self, min: torch.tensor) -> None:
        self._min_range = min.clone()

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False

    def sample(self) -> torch.tensor:
        if self._train:
            return self.sample_train()
        else:
            return self.sample_eval()

    @NotImplementedError
    def sample_train(self) -> torch.tensor:
        return None

    def sample_eval(self) -> torch.tensor:
        if (self._min_range == self._max_range).all():
            return self._min_range

        sample = self._current_step
        self._current_step += self._eval_step_size

        if (self._current_step > self._max_range).any():
            self._current_step = self._min_range

        return sample
