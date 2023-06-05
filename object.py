import torch
import utils_io
import utils_math
import utils_torch
import transforms_torch
import random
from typing import List

class RandomizableObject:
    def __init__(self, file_path: str, device: torch.cuda.device = torch.device("cuda")):
        self._device = device
        self.loadObject(file_path)

        self._last_rotation = None
        self._last_translation = None

    def setRotations(self, 
                     pitch_min: float, 
                     pitch_max: float, 
                     yaw_min: float, 
                     yaw_max: float,
                     roll_min: float,
                     roll_max: float) -> None:
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.yaw_min   = yaw_min
        self.yaw_max   = yaw_max
        self.roll_min  = roll_min
        self.roll_max  = roll_max

    def setTranslations(self,
                        vecA: torch.tensor,
                        vecB: torch.tensor) -> None:
        self.min_translation = vecA.to(self._device)
        self.max_translation = vecB.to(self._device)

    def setScales(self,
                  vecA: torch.tensor,
                  vecB: torch.tensor) -> None:
        self.min_scale = vecA.to(self._device)
        self.max_scale = vecB.to(self._device)

    def setAnimation(self,
                    vertex_offsets: torch.tensor,
                    replace_vertices: bool = False) -> None:
        self.vertex_offsets = vertex_offsets.to(self._device)
        self.replace_vertices = replace_vertices

    def setVertices(self, vertices: torch.tensor) -> None:
        self.vertices = vertices.to(self._device)

    def loadObject(self, file_path: str) -> None:
        object_dict = utils_io.read_config_yaml(file_path)
        self.setVertices(torch.tensor(object_dict["vertices"]))
        self.setAnimation(torch.tensor(object_dict["vertex_offsets"]), bool(object_dict["replace_vertices"]))
        self.setScales(torch.tensor(object_dict["min_scale"]), torch.tensor(object_dict["max_scale"]))
        self.setTranslations(torch.tensor(object_dict["min_translation"]), torch.tensor(object_dict["max_translation"]))
        self.setRotations(float(object_dict["min_pitch"]),
                          float(object_dict["max_pitch"]),
                          float(object_dict["min_yaw"]),
                          float(object_dict["max_yaw"]),
                          float(object_dict["min_roll"]),
                          float(object_dict["max_roll"]))
    
    def sampleScale(self) -> torch.tensor:
        scaleMatrix = torch.eye(4, device=self._device)
        random_translation = utils_torch.randomBetweenTensors(self.min_scale, self.max_scale)

        scaleMatrix[0, 0] = random_translation[0]
        scaleMatrix[1, 1] = random_translation[1]
        scaleMatrix[2, 2] = random_translation[2]
        return scaleMatrix

    def sampleRotations(self) -> torch.tensor:
        pitch = utils_math.uniformBetweenValues(self.pitch_min, self.pitch_max)
        yaw = utils_math.uniformBetweenValues(self.yaw_min, self.yaw_max)
        roll = utils_math.uniformBetweenValues(self.roll_min, self.roll_max)

        yaw_mat   = utils_math.getYawTransform(yaw, self._device)
        pitch_mat = utils_math.getPitchTransform(pitch, self._device)
        roll_mat  = utils_math.getRollTransform(roll, self._device)

        self._last_rotation = yaw_mat @ pitch_mat @ roll_mat
        return self._last_rotation

    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        random_translation = utils_torch.randomBetweenTensors(self.min_translation, self.max_translation)

        translationMatrix[0, 3] = random_translation[0]
        translationMatrix[1, 3] = random_translation[1]
        translationMatrix[2, 3] = random_translation[2]
        self._last_translation = translationMatrix
        return translationMatrix

    def sampleAnimation(self) -> torch.tensor:
        if self.vertex_offsets is None:
            return self.vertices

        num_anim_frames = self.vertex_offsets.shape[0]
        rand_index = random.randint(0, num_anim_frames - 1)

        if self.replace_vertices:
            return self.vertex_offsets[rand_index]
        
        return self.vertices + self.vertex_offsets[rand_index]

    def getRandomizedVertices(self) -> torch.tensor:
        # Sample Animations
        temp_vertex = self.sampleAnimation()

        # Scale Object
        temp_vertex = transforms_torch.transform_points(temp_vertex, self.sampleScale())

        # Rotate Object
        rotMat = self.sampleRotations()
        rotMat = transforms_torch.toMat4x4(rotMat)
        temp_vertex = transforms_torch.transform_points(temp_vertex, rotMat)

        # Translate Object
        temp_vertex = transforms_torch.transform_points(temp_vertex, self.sampleTranslation())

        return temp_vertex

    def rotation(self) -> torch.tensor:
        return self._last_rotation
    
    def translation(self) -> torch.tensor:
        return self._last_translation






class RelativeObject:
    def __init__(self, file_path: str, parent: RandomizableObject, device: torch.cuda.device = torch.device("cuda")):
        self._device = device
        self.loadObject(file_path)
        self._parent = parent

    def setRotation(self, 
                     pitch: float, 
                     yaw: float,
                     roll: float) -> None:
        self._pitch = pitch
        self._yaw   = yaw
        self._roll  = roll

        yaw_mat   = utils_math.getYawTransform(yaw, self._device)
        pitch_mat = utils_math.getPitchTransform(pitch, self._device)
        roll_mat  = utils_math.getRollTransform(roll, self._device)

        self._rotation = yaw_mat @ pitch_mat @ roll_mat

    def setTranslation(self, vec: List[float]) -> None:
        self._translation = torch.tensor(vec, device=self._device)

    def setScale(self, vec: List[float]) -> None:
        self._scale = torch.tensor(vec, device=self._device)

    def setAnimation(self,
                    vertex_offsets: torch.tensor,
                    replace_vertices: bool = False) -> None:
        self._vertex_offsets = vertex_offsets.to(self._device)
        self._replace_vertices = replace_vertices

    def setVertices(self, vertices: List[List[float]]) -> None:
        self._vertices = torch.tensor(vertices, device=self._device)

    def loadObject(self, file_path: str) -> None:
        object_dict = utils_io.read_config_yaml(file_path)
        self.setVertices(object_dict["vertices"])
        self.setAnimation(object_dict["vertex_offsets"]), bool(object_dict["replace_vertices"])
        self.setScale(object_dict["scale"])
        self.setTranslation(object_dict["translation"])
        self.setRotation(float(object_dict["pitch"]),
                          float(object_dict["yaw"]),
                          float(object_dict["roll"]))
    
    def sampleScale(self) -> torch.tensor:
        scaleMatrix = torch.eye(4, device=self._device)
        random_translation = utils_torch.randomBetweenTensors(self.min_scale, self.max_scale)

        scaleMatrix[0, 0] = random_translation[0]
        scaleMatrix[1, 1] = random_translation[1]
        scaleMatrix[2, 2] = random_translation[2]
        return scaleMatrix

    def sampleRotations(self) -> torch.tensor:
        pitch = utils_math.uniformBetweenValues(self.pitch_min, self.pitch_max)
        yaw = utils_math.uniformBetweenValues(self.yaw_min, self.yaw_max)
        roll = utils_math.uniformBetweenValues(self.roll_min, self.roll_max)

        return self._last_rotation

    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        random_translation = utils_torch.randomBetweenTensors(self.min_translation, self.max_translation)

        translationMatrix[0, 3] = random_translation[0]
        translationMatrix[1, 3] = random_translation[1]
        translationMatrix[2, 3] = random_translation[2]
        self._last_translation = translationMatrix
        return translationMatrix

    def sampleAnimation(self) -> torch.tensor:
        if self._vertex_offsets is None:
            return self._vertices

        num_anim_frames = self._vertex_offsets.shape[0]
        rand_index = random.randint(0, num_anim_frames - 1)

        if self._replace_vertices:
            return self._vertex_offsets[rand_index]
        
        return self._vertices + self._vertex_offsets[rand_index]

    def getVertexData(self) -> torch.tensor:
        # Sample Animations
        temp_vertex = self._verticses

        # Scale Object
        temp_vertex = transforms_torch.transform_points(temp_vertex, self._scale)

        # Rotate Object
        rotMat = self._rotation
        rotMat = transforms_torch.toMat4x4(rotMat)
        temp_vertex = transforms_torch.transform_points(temp_vertex, rotMat)

        # Translate Object
        temp_vertex = transforms_torch.transform_points(temp_vertex, self._translation)

        # Rotate by parent
        p_rotMat = self._parent.rotation()
        p_rotMat = transforms_torch.toMat4x4(p_rotMat)
        temp_vertex = transforms_torch.transform_points(temp_vertex, p_rotMat)

        # Translate by parent
        temp_vertex = transforms_torch.transform_points(temp_vertex, self._parent.translation())

        return temp_vertex


    def rotation(self) -> torch.tensor:
        return self._last_rotation
    

    def translation(self) -> torch.tensor:
        return self._last_translation


def _test():
    test_obj = RandomizableObject("test.yaml")
    test_obj.sampleAnimation()
    test_obj.sampleScale()
    test_obj.sampleTranslation()
    test_obj.sampleRotations()

    test_obj.getRandomizedVertices()
    a = 1


if __name__ == "__main__":
    _test()