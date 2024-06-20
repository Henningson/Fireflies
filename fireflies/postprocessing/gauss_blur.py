import numpy as np
import fireflies.postprocessing.base as base
import kornia
import torch


class GaussianBlur(base.BasePostProcessingFunction):
    def __init__(
        self,
        kernel_size: tuple[int, int],
        sigma: tuple[float, float],
        probability: float,
    ):
        super(GaussianBlur, self).__init__(probability)
        self._kernel_size = kernel_size
        self._sigma = sigma

    def post_process(self, image: np.array) -> np.array:
        image = (
            kornia.filters.gaussian_blur2d(
                torch.tensor(image).unsqueeze(0).unsqueeze(0),
                self._kernel_size,
                self._sigma,
            )
            .squeeze()
            .numpy()
        )
        return image
