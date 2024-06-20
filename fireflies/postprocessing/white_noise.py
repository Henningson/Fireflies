import numpy as np
import fireflies.postprocessing.base as base


class WhiteNoise(base.BasePostProcessingFunction):
    def __init__(
        self,
        mean: float,
        std: float,
        probability: float,
    ):
        super(WhiteNoise, self).__init__(probability)
        self._mean = mean
        self._std = std

    def post_process(self, image: np.array) -> np.array:
        image += np.random.normal(
            np.ones_like(image) * self._mean, np.ones_like(image) * self._std
        )
        return np.clip(image, 0, 1)
