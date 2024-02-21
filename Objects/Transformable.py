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
import Objects.flame_pytorch.flame as flame
from argparse import Namespace


class Transformable:
    def __init__(
        self, name: str, config: dict, device: torch.cuda.device = torch.device("cuda")
    ):

        self._device = device
        self._name = name

        self.setTranslationBoundaries(config["translation"])
        self.setRotationBoundaries(config["rotation"])
        self.setWorld(config["to_world"])

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
        self._world = torch.tensor(_origin, device=self._device)
        self._randomized_world = self._world.clone()

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
        self.min_translation = torch.tensor(
            [translation["min_x"], translation["min_y"], translation["min_z"]],
            device=self._device,
        )
        self.max_translation = torch.tensor(
            [translation["max_x"], translation["max_y"], translation["max_z"]],
            device=self._device,
        )

    def sampleRotation(self) -> torch.tensor:
        self.xRot = utilsmath.uniformBetweenValues(self.rot_min_x, self.rot_max_x)
        self.yRot = utilsmath.uniformBetweenValues(self.rot_min_y, self.rot_max_y)
        self.zRot = utilsmath.uniformBetweenValues(self.rot_min_z, self.rot_max_z)

        zMat = utilsmath.getPitchTransform(self.zRot, self._device)
        yMat = utilsmath.getYawTransform(self.yRot, self._device)
        xMat = utilsmath.getRollTransform(self.xRot, self._device)

        return transforms.toMat4x4(zMat @ yMat @ xMat)

    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        self.random_translation = utils.randomBetweenTensors(
            self.min_translation, self.max_translation
        )

        translationMatrix[0, 3] = self.random_translation[0]
        translationMatrix[1, 3] = -self.random_translation[2]
        translationMatrix[2, 3] = self.random_translation[1]
        self._last_translation = translationMatrix
        return translationMatrix

    def randomize(self) -> None:
        self._randomized_world = (
            self.sampleTranslation() @ self.sampleRotation() @ self._world
        )
        pass

    def relative(self) -> None:
        return self._relative

    def world(self) -> torch.tensor:
        # If no parent exists, just return the current translation
        if self._parent is None:
            temp = self._randomized_world.clone()
            # temp[0:3, 0:3] = temp[0:3, 0:3] @ utilsmath.getYTransform(np.pi, self._device)
            return temp

        return self._parent.world() @ self._randomized_world

    def nonRandomizedWorld(self) -> torch.tensor:
        if self._parent is None:
            return self._world

        return self._parent.nonRandomizedWorld() @ self._world


class Curve(Transformable):
    count = 0.0

    def __init__(
        self,
        name: str,
        curve: NURBS.Curve,
        config: dict,
        device: torch.cuda.device = torch.device("cuda"),
    ):
        Transformable.__init__(self, name, config, device)

        self._curve = curve
        self._curve.ctrlpts = self.convertToLocal(self._curve.ctrlpts)

        self._continuous = True
        self._interp_steps = 300
        self._interp_delta = 1.0 / self._interp_steps

    def convertToLocal(self, controlpoints: List[List[float]]) -> List[List[float]]:
        a = torch.tensor(controlpoints).to(self._device)
        # a = transforms.transform_points(a, transforms.toMat4x4(utilsmath.getXTransform(np.pi, self._device)))
        a = transforms.transform_points(a, self._world.inverse())
        # a[:, 0] *= -1.0
        #
        return a.tolist()

    def setContinuous(self, continuous: bool) -> None:
        self._continuous = continuous

    def sampleRotation(self) -> torch.tensor:
        return utils.torch.eye(4, device=self._device)

    def sampleTranslation(self) -> torch.tensor:
        translationMatrix = torch.eye(4, device=self._device)
        random_translation = random.uniform(0, 1)

        if self._continuous:
            Curve.count += self._interp_delta
            if Curve.count > 1.0:
                Curve.count = 0.0

            random_translation = Curve.count

        translation = self._curve.evaluate_single(random_translation)

        translationMatrix[0, 3] = translation[0]
        translationMatrix[1, 3] = translation[1]
        translationMatrix[2, 3] = translation[2]
        return translationMatrix

    def randomize(self) -> None:
        self._randomized_world = (
            self.sampleTranslation() @ self.sampleRotation() @ self._world
        )
        pass


class Mesh(Transformable):
    def __init__(
        self,
        name: str,
        vertex_data: List[float],
        config: dict,
        device: torch.cuda.device = torch.device("cuda"),
        base_path: str = None,
        sequential_animation: bool = True,
    ):
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

    def convertToLocal(self, vertices: torch.tensor) -> List[List[float]]:
        vertices = transforms.transform_points(
            vertices,
            transforms.toMat4x4(utilsmath.getXTransform(np.pi * 0.5, self._device)),
        )
        # vertices = transforms.transform_points(vertices, self._world)
        # a[:, 0] *= -1.0
        #
        return vertices

    def setFaces(self, faces: List[float]) -> None:
        self._faces = (
            torch.tensor(faces, device=self._device) if faces is not None else faces
        )

    def setVertices(self, vertices: List[float]) -> None:
        self._vertices = torch.tensor(vertices, device=self._device).reshape(-1, 3)
        self._vertices = self.convertToLocal(self._vertices)

    def setScaleBoundaries(self, scale: dict) -> None:
        self.min_scale = torch.tensor(
            [scale["min_x"], scale["min_y"], scale["min_z"]], device=self._device
        )
        self.max_scale = torch.tensor(
            [scale["max_x"], scale["max_y"], scale["max_z"]], device=self._device
        )

    def sampleScale(self) -> torch.tensor:
        scaleMatrix = torch.eye(4, device=self._device)
        random_scale = utils.randomBetweenTensors(self.min_scale, self.max_scale)

        scaleMatrix[0, 0] = random_scale[0]
        scaleMatrix[1, 1] = random_scale[1]
        scaleMatrix[2, 2] = random_scale[2]
        return scaleMatrix

    def randomize(self) -> None:
        self._randomized_world = (
            self.sampleTranslation() @ self.sampleRotation() @ self.sampleScale()
        )

    def faces(self) -> torch.tensor:
        return self._faces

    def getVertexData(self) -> torch.tensor:
        # Sample Animations
        temp_vertex = self.sampleAnimation() if self._animated else self._vertices

        # Transform by world transform
        temp_vertex = transforms.transform_points(temp_vertex, self.world())

        # parent = self._parent
        # while parent:
        #     temp_vertex = transforms.transform_points(temp_vertex, parent.world())

        return temp_vertex, None

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

        return self._vertex_offsets[index]


class ShapeModel(Mesh):
    def __init__(
        self,
        name: str,
        vertex_data: List[float],
        config: dict,
        device: torch.cuda.device = torch.device("cuda"),
        base_path: str = None,
        sequential_animation: bool = False,
    ):

        self._device = device
        self._name = name

        self.setTranslationBoundaries(config["translation"])
        self.setRotationBoundaries(config["rotation"])
        self.setWorld(config["to_world"])
        self._world = self._world @ transforms.toMat4x4(
            utilsmath.getXTransform(np.pi * 0.5, self._device)
        )
        self._randomized_world = self._world.clone()

        self._randomizable = bool(config["randomizable"])
        self._relative = bool(config["is_relative"])

        self._parent_name = config["parent_name"] if self._relative else None
        # Is loaded in a second step
        self._parent = None
        self._child = None

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])
        self._animated = bool(config["animated"])
        self._sequential_animation = sequential_animation

        self._animation_index = 0

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])
        self._stddev_range = config["stddev_range"]
        self._shape_layer = None
        self._model_params = {}

    def loadAnimation(self):
        return None

    def modelParmas(self) -> dict:
        return self._model_params

    def setModelParams(self, dict: dict) -> None:
        assert NotImplementedError

    def getVertexData(self):
        assert NotImplementedError


class FlameShapeModel(ShapeModel):
    def __init__(
        self,
        name: str,
        vertex_data: List[float],
        config: dict,
        device: torch.cuda.device = torch.device("cuda"),
        base_path: str = None,
        sequential_animation: bool = False,
    ):

        self._device = device
        self._name = name

        self.setTranslationBoundaries(config["translation"])
        self.setRotationBoundaries(config["rotation"])
        self.setWorld(config["to_world"])
        self._world = self._world @ transforms.toMat4x4(
            utilsmath.getXTransform(np.pi * 0.5, self._device)
        )
        self._randomized_world = self._world.clone()

        self._randomizable = bool(config["randomizable"])
        self._relative = bool(config["is_relative"])

        self._parent_name = config["parent_name"] if self._relative else None
        # Is loaded in a second step
        self._parent = None
        self._child = None

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])
        self._animated = bool(config["animated"])
        self._sequential_animation = sequential_animation

        self._animation_index = 0

        flame_config = Namespace(
            **{
                "batch_size": 1,
                "dynamic_landmark_embedding_path": "./Objects/flame_pytorch/model/flame_dynamic_embedding.npy",
                "expression_params": 50,
                "flame_model_path": "./Objects/flame_pytorch/model/generic_model.pkl",
                "num_worker": 4,
                "optimize_eyeballpose": True,
                "optimize_neckpose": True,
                "pose_params": 6,
                "ring_loss_weight": 1.0,
                "ring_margin": 0.5,
                "shape_params": 100,
                "static_landmark_embedding_path": "./Objects/flame_pytorch/model/flame_static_embedding.pkl",
                "use_3D_translation": True,
                "use_face_contour": True,
            }
        )

        self.setVertices(vertex_data)
        self.setScaleBoundaries(config["scale"])
        self._stddev_range = config["stddev_range"]
        self._shape_layer = flame.FLAME(flame_config).to(self._device)
        self._faces = self._shape_layer.faces

    def loadAnimation(self):
        return None

    def modelParams(self) -> dict:
        return self._shape_params

    def shapeParams(self) -> torch.tensor:
        return self._shape_params

    def expressionParams(self) -> torch.tensor:
        return self._expression_params

    def poseParams(self) -> torch.tensor:
        return self._pose_params

    def getVertexData(self):
        if not self._animated:
            return self._vertices, self._shape_layer.faces

        self._shape_params = (
            (torch.rand(1, 100, device=self._device) - 0.5)
            * 2.0
            * self._stddev_range
            * 2.0
        )
        self._pose_params = torch.zeros(1, 6, device=self._device)
        self._expression_params = torch.zeros(1, 50, device=self._device)

        vertices, _ = self._shape_layer(
            self._shape_params, self._expression_params, self._pose_params
        )
        vertices = vertices[0]

        vertices = transforms.transform_points(
            vertices,
            self.world()
            @ transforms.toMat4x4(utilsmath.getXTransform(np.pi * 0.5, self._device)),
        )

        return vertices, self._shape_layer.faces


if __name__ == "__main__":
    fsm = FlameShapeModel(
        "Bla",
    )
