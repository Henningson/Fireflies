import torch
import Utils.utils as utils
import Utils.math_helper as utilsmath
import Utils.transforms as transforms
import random
from typing import List, Tuple
import os
import pywavefront
import numpy as np


class BaseEntity:
    def __init__(
        self,
        base_path: str,
        name: str,
        config: dict,
        vertex_data: List[float],
        device: torch.cuda.device = torch.device("cuda"),
    ):
        self._device = device

        self.setVertices(vertex_data)
        self.setScale(config["scale"])
        self.setTranslation(config["translation"])
        self.setRotation(
            float(config["pitch"]), float(config["yaw"]), float(config["roll"])
        )

        self._animated = bool(config["animated"])
        if self._animated:
            self._animated = True
            self.setAnimation(config["vertex_offsets"])

    def setRotation(self, pitch: float, yaw: float, roll: float) -> None:
        self._pitch = pitch
        self._yaw = yaw
        self._roll = roll

        yaw_mat = utilsmath.getYawTransform(yaw, self._device)
        pitch_mat = utilsmath.getPitchTransform(pitch, self._device)
        roll_mat = utilsmath.getRollTransform(roll, self._device)

        self._rotation = yaw_mat @ pitch_mat @ roll_mat

    def setTranslation(self, vec: List[float]) -> None:
        self._translation = torch.eye(4, device=self._device)

        self._translation[0, 3] = vec[0]
        self._translation[1, 3] = vec[1]
        self._translation[2, 3] = vec[2]

    def setScale(self, vec: List[float]) -> None:
        self._scale = torch.tensor(vec, device=self._device)
        self._scale = torch.eye(4, device=self._device)

        self._scale[0, 0] = vec[0]
        self._scale[1, 1] = vec[1]
        self._scale[2, 2] = vec[2]

    def setAnimation(self, vertex_offsets: torch.tensor) -> None:
        self._vertex_offsets = torch.tensor(vertex_offsets, device=self._device)

    def setVertices(self, vertices: List[float]) -> None:
        self._vertices = torch.tensor(vertices, device=self._device).reshape(-1, 3)

    def loadObject(self, config: dict) -> None:
        pass

    def sampleAnimation(self) -> torch.tensor:
        if self._vertex_offsets is None:
            return self._vertices

        num_anim_frames = self._vertex_offsets.shape[0]
        rand_index = random.randint(0, num_anim_frames - 1)

        return self._vertex_offsets[rand_index]

    def getVertexData(self) -> torch.tensor:
        # Sample Animations
        temp_vertex = self.sampleAnimation()

        # Scale Object
        temp_vertex = transforms.transform_points(temp_vertex, self._scale)

        # Rotate Object
        rotMat = self._rotation
        rotMat = transforms.toMat4x4(rotMat)
        temp_vertex = transforms.transform_points(temp_vertex, rotMat)

        # Translate Object
        temp_vertex = transforms.transform_points(temp_vertex, self._translation)

        return temp_vertex

    def rotation(self) -> torch.tensor:
        return self._rotation

    def translation(self) -> torch.tensor:
        return self._translation


class RandomizableCamera:
    def __init__(self, config: dict, device: torch.cuda.device = torch.device("cuda")):

        self._device = device
        self._animated = False

        self.setTranslationLimits(config["translation"])
        self.setRotationLimits(config["rotation"])
        self.setWorld(config["to_world"])

    def setWorld(self, to_world: List[List[float]]) -> None:
        self._to_world = torch.tensor(to_world, device=self._device)

    def setRotationLimits(self, rotation: dict) -> None:
        self.rot_min_x = rotation["min_x"]
        self.rot_max_x = rotation["max_x"]
        self.rot_min_y = rotation["min_y"]
        self.rot_max_y = rotation["max_y"]
        self.rot_min_z = rotation["min_z"]
        self.rot_max_z = rotation["max_z"]

    def setTranslationLimits(self, translation: dict) -> None:
        self.min_translation = torch.tensor(
            [translation["min_x"], translation["min_y"], translation["min_z"]],
            device=self._device,
        )
        self.max_translation = torch.tensor(
            [translation["max_x"], translation["max_y"], translation["max_z"]],
            device=self._device,
        )

    def sampleRotation(self) -> torch.tensor:
        xRot = utilsmath.uniformBetweenValues(self.rot_min_x, self.rot_max_x)
        yRot = utilsmath.uniformBetweenValues(self.rot_min_y, self.rot_max_y)
        zRot = utilsmath.uniformBetweenValues(self.rot_min_z, self.rot_max_z)

        zMat = utilsmath.getYawTransform(zRot, self._device)
        yMat = utilsmath.getPitchTransform(yRot, self._device)
        xMat = utilsmath.getRollTransform(xRot, self._device)

        self._last_rotation = zMat @ yMat @ xMat
        return self._last_rotation

    def sampleTranslation(self) -> torch.tensor:
        self._last_translation = utils.randomBetweenTensors(
            self.min_translation, self.max_translation
        )
        return self._last_translation

    def getTransforms(self) -> torch.tensor:
        # Rotate Object
        randomTransform = self.sampleRotation()
        randomTransform = transforms.toMat4x4(randomTransform)
        randomTranslation = self.sampleTranslation()

        randomTransform[0, 3] = randomTranslation[0]
        randomTransform[1, 3] = randomTranslation[1]
        randomTransform[2, 3] = randomTranslation[1]

        self._last_transform = self._to_world @ randomTransform
        return self._last_transform

    def getLastTransform(self):
        return self._last_transform

    def rotation(self) -> torch.tensor:
        return self._last_rotation

    def translation(self) -> torch.tensor:
        return self._last_translation

    def is_animated(self) -> bool:
        return self._animated


class Projector:
    def __init__(self, config: dict, device: torch.cuda.device = torch.device("cuda")):

        self._device = device
        self._animated = False
        self._parent = None
        self.setTranslation(config["translation"])
        self.setRotation(config["rotation"])
        self.setWorld(config["to_world"])

    def setWorld(self, to_world: List[List[float]]) -> None:
        self._to_world = torch.tensor(to_world, device=self._device)

    def setParent(self, parent: RandomizableCamera) -> None:
        self._parent = parent

    def setRotation(self, rotation: dict) -> None:
        self.rot_min_x = rotation["min_x"]
        self.rot_max_x = rotation["max_x"]
        self.rot_min_y = rotation["min_y"]
        self.rot_max_y = rotation["max_y"]
        self.rot_min_z = rotation["min_z"]
        self.rot_max_z = rotation["max_z"]

    def setTranslation(self, translation: dict) -> None:
        self.min_translation = torch.tensor(
            [translation["min_x"], translation["min_y"], translation["min_z"]],
            device=self._device,
        )
        self.max_translation = torch.tensor(
            [translation["max_x"], translation["max_y"], translation["max_z"]],
            device=self._device,
        )

    def sampleRotation(self) -> torch.tensor:
        xRot = utilsmath.uniformBetweenValues(self.rot_min_x, self.rot_max_x)
        yRot = utilsmath.uniformBetweenValues(self.rot_min_y, self.rot_max_y)
        zRot = utilsmath.uniformBetweenValues(self.rot_min_z, self.rot_max_z)

        zMat = utilsmath.getYawTransform(zRot, self._device)
        yMat = utilsmath.getPitchTransform(yRot, self._device)
        xMat = utilsmath.getRollTransform(xRot, self._device)

        self._last_rotation = zMat @ yMat @ xMat
        return self._last_rotation

    def sampleTranslation(self) -> torch.tensor:
        self._last_translation = utilsmath.randomBetweenTensors(
            self.min_translation, self.max_translation
        )
        return self._last_translation

    def getTransforms(self) -> torch.tensor:
        parentMat = self._parent.getLastTransform()
        blendParentMat = transforms.matToBlender(parentMat, self._device)
        blendWorldMat = transforms.matToBlender(self._to_world, self._device)

        relative_world = transforms.matToMitsuba(
            blendParentMat @ blendWorldMat, self._device
        )

        return relative_world

    def getWorldMat(self) -> torch.tensor:
        translation, rotMat = self.getTransforms()
        rotMat = transforms.toMat4x4(rotMat)

        rotMat[0, 3] = translation[0]
        rotMat[1, 3] = translation[1]
        rotMat[2, 3] = translation[2]
        return rotMat

    def rotation(self) -> torch.tensor:
        return self._last_rotation

    def translation(self) -> torch.tensor:
        return self._last_translation

    def is_animated(self) -> bool:
        return self._animated


class Randomizable:
    def __init__(
        self,
        base_path: str,
        name: str,
        config: dict,
        vertex_data: List[float],
        sequential_animation: bool = False,
        device: torch.cuda.device = torch.device("cuda"),
    ):

        self._device = device

        self.setVertices(vertex_data)
        self.setScale(config["scale"])
        self.setTranslation(config["translation"])
        self.setRotation(config["rotation"])
        self._animated = bool(config["animated"])
        self._sequential_animation = sequential_animation

        if self._animated:
            self._animation_index = 0
            self.loadAnimation(base_path, name)

    def setVertices(self, vertices: List[float]) -> None:
        self._vertices = torch.tensor(vertices, device=self._device).reshape(-1, 3)

    def loadAnimation(self, base_path, obj_name):
        self._vertex_offsets = []
        self._face_data = []
        for file in sorted(os.listdir(os.path.join(base_path, obj_name + "/"))):
            if file.endswith(".obj"):
                obj_path = os.path.join(base_path, obj_name, file)
                print(os.path.join(base_path, obj_name, file))

                obj = pywavefront.Wavefront(obj_path, collect_faces=True)

                self._vertex_offsets.append(
                    torch.tensor(obj.vertices, device=self._device).reshape(-1, 3)
                )
                self._face_data.append(
                    torch.tensor(obj.mesh_list[0].faces, device=self._device).flatten()
                )

    def next_anim_step(self) -> None:
        self._animation_index += 1

    def sampleAnimation(self):
        if not self._animated:
            return self._vertices, None

        index = 0
        if self._sequential_animation:
            index = self._animation_index % len(self._vertex_offsets)
        else:
            num_anim_frames = len(self._vertex_offsets)
            index = random.randint(0, num_anim_frames - 1)

        return self._vertex_offsets[index], self._face_data[index]

    def setRotation(self, rotation: dict) -> None:
        self.rot_min_x = rotation["min_x"]
        self.rot_max_x = rotation["max_x"]
        self.rot_min_y = rotation["min_y"]
        self.rot_max_y = rotation["max_y"]
        self.rot_min_z = rotation["min_z"]
        self.rot_max_z = rotation["max_z"]

    def setTranslation(self, translation: dict) -> None:
        self.min_translation = torch.tensor(
            [translation["min_x"], translation["min_y"], translation["min_z"]],
            device=self._device,
        )
        self.max_translation = torch.tensor(
            [translation["max_x"], translation["max_y"], translation["max_z"]],
            device=self._device,
        )

    def setScale(self, scale: dict) -> None:
        self.min_scale = torch.tensor(
            [scale["min_x"], scale["min_y"], scale["min_z"]], device=self._device
        )
        self.max_scale = torch.tensor(
            [scale["max_x"], scale["max_y"], scale["max_z"]], device=self._device
        )

    def sampleScale(self) -> torch.tensor:
        scaleMatrix = torch.eye(4, device=self._device)
        random_translation = utils.randomBetweenTensors(self.min_scale, self.max_scale)

        scaleMatrix[0, 0] = random_translation[0]
        scaleMatrix[1, 1] = random_translation[1]
        scaleMatrix[2, 2] = random_translation[2]
        return scaleMatrix

    def sampleRotation(self) -> torch.tensor:
        xRot = utilsmath.uniformBetweenValues(self.rot_min_x, self.rot_max_x)
        yRot = utilsmath.uniformBetweenValues(self.rot_min_y, self.rot_max_y)
        zRot = utilsmath.uniformBetweenValues(self.rot_min_z, self.rot_max_z)

        zMat = utilsmath.getYawTransform(zRot, self._device)
        yMat = utilsmath.getPitchTransform(yRot, self._device)
        xMat = utilsmath.getRollTransform(xRot, self._device)

        self._last_rotation = zMat @ yMat @ xMat
        return self._last_rotation

    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        random_translation = utils.randomBetweenTensors(
            self.min_translation, self.max_translation
        )

        translationMatrix[0, 3] = random_translation[0]
        translationMatrix[1, 3] = random_translation[1]
        translationMatrix[2, 3] = random_translation[2]
        self._last_translation = translationMatrix
        return translationMatrix

    def getVertexData(self) -> torch.tensor:
        # Sample Animations
        temp_vertex, temp_faces = self.sampleAnimation()

        # Scale Object
        temp_vertex = transforms.transform_points(
            temp_vertex.reshape(-1, 3), self.sampleScale()
        )

        # Rotate Object
        rotMat = self.sampleRotation()
        rotMat = transforms.toMat4x4(rotMat)
        temp_vertex = transforms.transform_points(temp_vertex, rotMat)

        # Translate Object
        temp_vertex = transforms.transform_points(temp_vertex, self.sampleTranslation())

        return temp_vertex, temp_faces

    def rotation(self) -> torch.tensor:
        return self._last_rotation

    def translation(self) -> torch.tensor:
        return self._last_translation

    def is_animated(self) -> bool:
        return self._animated


class RelativeEntity(BaseEntity):
    def __init__(
        self,
        file_path: str,
        parent: BaseEntity,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        BaseEntity.__init__(self, file_path=file_path, device=device)
        self._parent = parent

    def getVertexData(self) -> torch.tensor:
        # Sample Animations
        temp_vertex = self.sampleAnimation() if self._animated else self._vertices

        # Scale Object
        temp_vertex = transforms.transform_points(temp_vertex, self._scale)

        # Rotate Object
        rotMat = self._rotation
        rotMat = transforms.toMat4x4(rotMat)
        temp_vertex = transforms.transform_points(temp_vertex, rotMat)

        # Translate Object
        temp_vertex = transforms.transform_points(temp_vertex, self._translation)

        # Rotate by parent
        p_rotMat = self._parent.rotation()
        p_rotMat = transforms.toMat4x4(p_rotMat)
        temp_vertex = transforms.transform_points(temp_vertex, p_rotMat)

        # Translate by parent
        temp_vertex = transforms.transform_points(
            temp_vertex, self._parent.translation()
        )

        return temp_vertex


def _test():
    print("_____ RANDOMIZED OBJECT _____")
    test_random_obj = RandomizableEntity("test_random_object.yaml")
    print(test_random_obj.getVertexData())

    print("_____ BASE OBJECT _____")
    test_base_obj = BaseEntity("test_base_object.yaml")
    print(test_base_obj.getVertexData())

    print("_____ RELATIVE OBJECT _____")
    test_rel_obj = RelativeEntity("test_relative_object.yaml", test_base_obj)
    print(test_rel_obj.getVertexData())


if __name__ == "__main__":
    _test()
