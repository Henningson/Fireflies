import numpy as np

import fireflies.postprocessing.base as base
import cv2
import random
import kornia
import torch


class ApplySilhouette(base.BasePostProcessingFunction):
    def __init__(
        self,
        probability: float = 2.0,
    ):
        super(ApplySilhouette, self).__init__(probability)

    def post_process(self, image: np.array) -> np.array:
        silhouette_image = np.zeros_like(image)
        spawning_rect_x = [100, 200]
        spawning_rect_y = [200, 300]
        radius_interval = [170, 230]

        cc_x = random.randint(spawning_rect_x[0], spawning_rect_x[1])
        cc_y = random.randint(spawning_rect_y[0], spawning_rect_y[1])
        radius = random.randint(radius_interval[0], radius_interval[1])
        silhouette_image = cv2.circle(
            silhouette_image, (cc_x, cc_y), radius, color=1, thickness=-1
        )

        silhouette_image = (
            kornia.filters.gaussian_blur2d(
                torch.tensor(silhouette_image).unsqueeze(0).unsqueeze(0),
                (11, 11),
                (5, 5),
            )
            .squeeze()
            .numpy()
        )

        return image * silhouette_image
