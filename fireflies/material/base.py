import torch
from typing import List

import fireflies.utils.math
import fireflies.entity
from fireflies.utils.warnings import (
    RotationAssignmentWarning,
    RelativeAssignmentWarning,
    TranslationAssignmentWarning,
    WorldAssignmentWarning,
)


class Material(fireflies.entity.Transformable):
    def __init__(
        self,
        name: str,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        super(Material, self).__init__(name, device)

    def randomize(self) -> None:
        for key, sampler in self._float_attributes.items():
            self._randomized_float_attributes[key] = sampler.sample()

        for key, sampler in self._vec3_attributes.items():
            self._randomized_vec3_attributes[key] = sampler.sample()

    @WorldAssignmentWarning
    def set_world(self, _origin: torch.tensor) -> None:
        super(Material, self).set_world(_origin)

    @RelativeAssignmentWarning
    def setParent(self, parent) -> None:
        super(Material, self).setParent(parent)

    @RelativeAssignmentWarning
    def setChild(self, child) -> None:
        super(Material, self).setChild(child)

    @RotationAssignmentWarning
    def rotate_x(self, min_rot: float, max_rot: float) -> None:
        super(Material, self).rotate_x(min_rot, max_rot)

    @RotationAssignmentWarning
    def rotate_y(self, min_rot: float, max_rot: float) -> None:
        super(Material, self).rotate_y(min_rot, max_rot)

    @RotationAssignmentWarning
    def rotate_z(self, min_rot: float, max_rot: float) -> None:
        super(Material, self).rotate_z(min_rot, max_rot)

    @RotationAssignmentWarning
    def rotate(self, min: torch.tensor, max: torch.tensor) -> None:
        super(Material, self).rotate(min, max)

    @TranslationAssignmentWarning
    def translate_x(self, min_translation: float, max_translation: float) -> None:
        super(Material, self).translate_x(min_translation, max_translation)

    @TranslationAssignmentWarning
    def translate_y(self, min_translation: float, max_translation: float) -> None:
        super(Material, self).translate_y(min_translation, max_translation)

    @TranslationAssignmentWarning
    def translate_z(self, min_translation: float, max_translation: float) -> None:
        super(Material, self).translate_z(min_translation, max_translation)

    @TranslationAssignmentWarning
    def translate(self, min: torch.tensor, max: torch.tensor) -> None:
        super(Material, self).translate(min, max)

    @RotationAssignmentWarning
    def sample_rotation(self) -> torch.tensor:
        return super(Material, self).sample_rotation()

    @TranslationAssignmentWarning
    def sample_translation(self) -> torch.tensor:
        return super(Material, self).sample_translation()

    @RelativeAssignmentWarning
    def relative(self) -> None:
        return super(Material, self).relative()

    @WorldAssignmentWarning
    def world(self) -> torch.tensor:
        return super(Material, self).world()

    @WorldAssignmentWarning
    def nonRandomizedWorld(self) -> torch.tensor:
        return super(Material, self).nonRandomizedWorld()
