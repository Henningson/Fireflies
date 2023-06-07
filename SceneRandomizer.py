
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import torch
import numpy as np


class SceneRandomizer:
    def __init__(self, project_path):
        self._scene = 