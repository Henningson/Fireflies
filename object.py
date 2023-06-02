import torch
import utils_io
import utils_math
import utils_torch
import random

class Object:
    def __init__(self, file_path: str, device: torch.cuda.device = torch.device("cuda")):
        self.loadObject(file_path)
        self._device = device
        self._vertices = torch.tensor([9, 3], device=self._device)

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
        self.setScale(torch.tensor(object_dict["min_scale"]), torch.tensor(object_dict["max_scale"]))
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
        pitch = utils_math.randomBetweenTwoValues(self.min_pitch, self.max_pitch)
        yaw = utils_math.randomBetweenTwoValues(self.min_yaw, self.max_yaw)
        roll = utils_math.randomBetweenTwoValues(self.min_roll, self.max_roll)

        yaw_mat   = utils_math.getYawTransform(yaw, self._device)
        pitch_mat = utils_math.getPitchTransform(pitch, self._device)
        roll_mat  = utils_math.getRollTransform(roll, self._device)

        return yaw_mat*pitch_mat*roll_mat

    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        random_translation = utils_torch.randomBetweenTensors(self.min_translation, self.max_translation)

        translationMatrix[3, 0] = random_translation[0]
        translationMatrix[3, 1] = random_translation[1]
        translationMatrix[3, 2] = random_translation[2]
        return translationMatrix

    def sampleAnimation(self) -> torch.tensor:
        num_anim_frames = self.vertex_offsets.shape[0]
        rand_index = random.randint(num_anim_frames - 1)

        if self.replace_vertices:
            return self.vertex_offsets[rand_index]
        
        return self.vertices + self.vertex_offsets[rand_index]

    def randomizeObject(self, params) -> None:
        # TODO: Change parameters inplace
        pass