import numpy as np
from .base import BasePostProcessingFunction

from typing import List


class PostProcessor:
    def __init__(
        self,
        post_process_funcs: List[BasePostProcessingFunction],
    ):
        self._post_process_functs = post_process_funcs

    def post_process(self, image: np.array) -> np.array:
        image_copy = image.copy()
        for func in self._post_process_functs:
            image_copy = func.apply(image_copy)

        return image_copy
