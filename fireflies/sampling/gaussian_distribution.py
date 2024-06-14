import torch
import fireflies.sampling.base as base


class GaussianSampler(base.Sampler):
    def __init__(
        self,
        min: torch.tensor,
        max: torch.tensor,
        mean: torch.tensor,
        std: torch.tensor,
        eval_step_size: float = 0.01,
        device: torch.cuda.device = torch.device("cuda"),
    ) -> None:
        super(GaussianSampler, self).__init__(min, max, eval_step_size, device)
        self._mean = mean
        self._std = std

    def sample_train(self) -> torch.tensor:
        return torch.normal(self._mean, self._std, device=self._device)
