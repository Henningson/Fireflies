import torch
import camera

import fireflies.entity.base


class Projector(camera.Camera):
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
        texture: torch.tensor,
        fov: float,
        near_clip: float = 0.01,
        far_clip: float = 1000.0,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        super(Projector, self).__init__(
            transform, perspective, fov, near_clip, far_clip, device
        )
        self._texture = texture.to(self._device)
        Projector.id += 1

    def full_key(self, key: str):
        return self._key + "." + Projector.MITSUBA_KEYS[key]

    def generate_mitsuba_key(self) -> str:
        if Projector.id == 0:
            return "Projector"

        return "Projector_{0}".format(id)

    def texture(self) -> torch.tensor:
        return self._texture
