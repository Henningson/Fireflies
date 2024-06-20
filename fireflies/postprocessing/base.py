import numpy as np
import random
import fireflies.sampling


class BasePostProcessingFunction:
    def __init__(self, probability: float):
        self._probability = probability

    def apply(self, image: np.array) -> np.array:
        if random.uniform(0, 1) < self._probability:
            return self.post_process(image)

        return image

    @NotImplementedError
    def post_process(self, image: np.array) -> np.array:
        return None
