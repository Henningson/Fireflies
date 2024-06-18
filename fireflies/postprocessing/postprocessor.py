import numpy as np

from typing import List


class PostProcessor:
    def __init__(self, bla: List[float]):
        self._post_process_functions = bla
        pass
        # do stuff

    def post_process(self) -> np.array:
        for func in self._post_process_functions:
            pass
            # do stuff
