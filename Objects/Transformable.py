import torch
import Utils.utils as utils
import Utils.math as utilsmath
import Utils.transforms as transforms
import random
from typing import List, Tuple
import os
import pywavefront
import numpy as np
from geomdl import NURBS


class Transformable:
    def __init__(self,
                 name: str, 
                 config: dict, 
                 device: torch.cuda.device = torch.device("cuda")):
        
        self._device = device
        self._name = name

        self.setTranslationBoundaries(config["translation"])
        self.setRotationBoundaries(config["rotation"])
        self.setWorld(config['to_world'])

        self._randomizable = bool(config["randomizable"])
        self._relative = bool(config["is_relative"])

        self._parent_name = config["parent_name"] if self._relative else None
        # Is loaded in a second step
        self._parent = None
        self._child = None

    def parent(self):
        return self._parent

    def child(self):
        return self._child

    def name(self):
        return self._name

    def parentName(self) -> str:
        return self._parent_name

    def setWorld(self, _origin: List[List[float]]) -> None:
        self._origin = transforms.matToBlender(torch.tensor(_origin, device=self._device), self._device)
        self._world = self._origin.clone()

    def setParent(self, parent) -> None:
        self._parent = parent
        parent.setChild(self)

    def setChild(self, child) -> None:
        self._child = child

    def setRotationBoundaries(self, rotation: dict) -> None:
        self.rot_min_x = rotation["min_x"]
        self.rot_max_x = rotation["max_x"]
        self.rot_min_y = rotation["min_y"]
        self.rot_max_y = rotation["max_y"]
        self.rot_min_z = rotation["min_z"]
        self.rot_max_z = rotation["max_z"]


    def setTranslationBoundaries(self, translation: dict) -> None:
        self.min_translation = torch.tensor([translation["min_x"], translation["min_y"], translation["min_z"]], device=self._device)
        self.max_translation = torch.tensor([translation["max_x"], translation["max_y"], translation["max_z"]], device=self._device)


    def sampleRotation(self) -> torch.tensor:
        xRot = utilsmath.uniformBetweenValues(self.rot_min_x, self.rot_max_x)
        yRot = utilsmath.uniformBetweenValues(self.rot_min_y, self.rot_max_y)
        zRot = utilsmath.uniformBetweenValues(self.rot_min_z, self.rot_max_z)

        zMat = utilsmath.getPitchTransform(zRot, self._device)
        yMat = utilsmath.getYawTransform(yRot, self._device)
        xMat = utilsmath.getRollTransform(xRot, self._device)

        return transforms.toMat4x4(zMat @ yMat @ xMat)


    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        random_translation = utils.randomBetweenTensors(self.min_translation, self.max_translation)

        translationMatrix[0, 3] = random_translation[0]
        translationMatrix[1, 3] = random_translation[1]
        translationMatrix[2, 3] = random_translation[2]
        self._last_translation = translationMatrix
        return translationMatrix


    def randomize(self) -> None:
        self._world = self.sampleTranslation() @ self.sampleRotation() @ self.origin()
        #print(self._name)
        #print(self._world)


    def relative(self) -> None:
        return self._relative


    def origin(self) -> None:
        if self._parent is None:
            return self._origin

        return self._origin @ self._parent.origin()


    def world(self) -> torch.tensor:
        # If no parent exists, just return the current translation
        if self._parent is None:
            return self._world

        return self._parent.world() @ self._world


class Curve(Transformable):
    def __init__(self,
                 name: str,
                 curve: NURBS.Curve,
                 config: dict,
                 device: torch.cuda.device = torch.device("cuda")):
        Transformable.__init__(self, name, config, device)
        
        self._curve = curve
        self._origin = torch.eye(4, device=self._device)
        self._world  = torch.eye(4, device=self._device)


    def sampleRotation(self) -> torch.tensor:
        return utils.torch.eye(4, device=self._device)


    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        random_translation = random.uniform(0, 1)

        translation = self._curve.evaluate_single(random_translation)

        translationMatrix[0, 3] = translation[0]
        translationMatrix[1, 3] = translation[1]
        translationMatrix[2, 3] = translation[2]
        return translationMatrix

    def randomize(self) -> None:
        self._origin = self.sampleTranslation() @ self.sampleRotation()
        self._world = self._origin



class Mesh(Transformable):
    def __init__(self,
                 name: str,
                 vertex_data: List[float],
                 config: dict, 
                 device: torch.cuda.device = torch.device("cuda"),
                 base_path: str = None,
                 sequential_animation: bool = False):
        Transformable.__init__(self, name, config, device)

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])
        self._animated = bool(config["animated"])
        self._sequential_animation = sequential_animation

        if self._animated:
            self._animation_index = 0
            self.loadAnimation(base_path, name)


    def animated(self) -> bool:
        return self._animated


    def setVertices(self, vertices: List[float]) -> None:
        self._vertices = torch.tensor(vertices, device=self._device).reshape(-1, 3)

    
    def setScaleBoundaries(self, scale: dict) -> None:
        self.min_scale = torch.tensor([scale["min_x"], scale["min_y"], scale["min_z"]], device=self._device)
        self.max_scale = torch.tensor([scale["max_x"], scale["max_y"], scale["max_z"]], device=self._device)


    def sampleScale(self) -> torch.tensor:
        scaleMatrix = torch.eye(4, device=self._device)
        random_scale = utils.randomBetweenTensors(self.min_scale, self.max_scale)

        scaleMatrix[0, 0] = random_scale[0]
        scaleMatrix[1, 1] = random_scale[1]
        scaleMatrix[2, 2] = random_scale[2]
        return scaleMatrix


    def randomize(self) -> None:
        self._world = self.sampleTranslation() @ \
                      self.sampleRotation() @ \
                      self.sampleScale()


    def getVertexData(self) -> torch.tensor:
        # Sample Animations
        temp_vertex = self.sampleAnimation() if self._animated else self._vertices

        # Transform by world transform
        temp_vertex = transforms.transform_points(temp_vertex, self.world())

        if self._relative:
            # Transform by parent transform
            temp_vertex = transforms.transform_points(temp_vertex, self._parent.world())

        return temp_vertex
    

    def loadAnimation(self, base_path, obj_name):
        self._vertex_offsets = []
        self._face_data = []    
        for file in sorted(os.listdir(os.path.join(base_path, obj_name + "/"))):
            if file.endswith(".obj"):
                obj_path = os.path.join(base_path, obj_name, file)
                print(os.path.join(base_path, obj_name, file))

                obj = pywavefront.Wavefront(obj_path, collect_faces=True)

                self._vertex_offsets.append(torch.tensor(obj.vertices, device=self._device).reshape(-1, 3))
                self._face_data.append(torch.tensor(obj.mesh_list[0].faces, device=self._device).flatten())
                

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

        return self._vertex_offsets[index]