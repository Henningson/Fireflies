import torch
from typing import List


def retain_grads(non_leaf_tensor: List[torch.tensor]) -> None:
    for tensor in non_leaf_tensor:
        tensor.retain_grad()
