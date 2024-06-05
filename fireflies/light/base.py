import torch
from typing import List

import fireflies.utils.math
import fireflies.entity


class Light(fireflies.entity.Transformable):
    def __init__(
        self,
        name: str,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        super(Light, self).__init__(self, name, device)
