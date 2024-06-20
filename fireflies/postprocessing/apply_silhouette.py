import os
import numpy as np
import pywavefront

import fireflies.postprocessing.base as base
import fireflies.sampling


class ApplySilhouette(base.BasePostProcessingFunction):
    def __init__(
        self,
        silhouette_image: np.array,
        probability: float = 1.0,
    ):
        super(ApplySilhouette, self).__init__(probability)
        self._silhouette_image = silhouette_image

    def post_process(self, image: np.array) -> np.array:
        return image * self._silhouette_image
