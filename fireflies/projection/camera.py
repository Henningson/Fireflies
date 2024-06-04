import torch
import fireflies.utils.math
import fireflies.utils.transforms

import fireflies.entity.base


class Camera:
    id = 0
    MITSUBA_KEYS = {
        "fov": "x_fov",
        "f": "x_fov",
        "to_world": "to_world",
        "world": "to_world",
    }

    def __init__(
        self,
        transform: fireflies.entity.base.Transformable,
        perspective: torch.tensor,
        fov: float,
        near_clip: float = 0.01,
        far_clip: float = 1000.0,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        self.device = device

        self._transformable = transform
        self._perspective = perspective
        self._near_clip = near_clip
        self._far_clip = far_clip
        self._fov = fov

        self._key = self.generate_mitsuba_key()
        Camera.id += 1

    def full_key(self, key: str):
        return self._key + "." + Camera.MITSUBA_KEYS[key]

    def key(self) -> str:
        return self._key

    def near_clip(self) -> float:
        return self._near_clip

    def generate_mitsuba_key(self) -> str:
        if Camera.id == 0:
            return "PerspectiveCamera"

        return "PerspectiveCamera_{0}".format(id)

    def far_clip(self) -> float:
        return self._far_clip

    def fov(self) -> torch.tensor:
        return self._fov

    def origin(self) -> torch.tensor:
        return self._transformable.origin()

    def world(self) -> torch.tensor:
        return self._transformable.world()

    def randomize(self) -> None:
        self._transformable.randomize()

    def pointsToNDC(self, points) -> torch.tensor:
        view_space_points = fireflies.utils.transforms.transform_points(
            points, self.world().inverse()
        )
        ndc_points = fireflies.utils.transforms.transform_points(
            view_space_points, self._perspective
        )
        return ndc_points
