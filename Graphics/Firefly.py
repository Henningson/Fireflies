from bs4 import BeautifulSoup
from pathlib import Path
import os
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import Objects.entity as entity
import Utils.utils as utils
import torch
import Objects.laser as laser
import Objects.Camera as Camera
import Objects.Transformable as Transformable
import Utils.transforms as transforms
import numpy as np
import Utils.math as math

class Scene:
    def __init__(self, 
                 scene_params, 
                 base_path: str, 
                 sequential_animation: bool = False, 
                 steps_per_frame: int = 1, 
                 device: torch.cuda.device = torch.device("cuda")):
        
        self.mi_xml = self.getMitsubaXML(os.path.join(base_path, "scene.xml"))
        self.firefly_path = os.path.join(base_path, "Firefly")
        self.scene_params = scene_params

        self.base_path = base_path

        # Here, only objects are saved, that have a "randomizable"-tag inside the yaml file.
        self.meshes = {}
        self.projector = None
        self.camera = None
        self.lights = {}
        self.curves = []

        self._parent_transformables = []

        self._device = device

        self._num_updates = 0
        self._sequential_animation = sequential_animation
        self._steps_per_frame = steps_per_frame

        self.initScene()


    def getMitsubaXML(self, path):
        data = None
        with open(path, 'r') as f:
            data = f.read()
        return BeautifulSoup(data, "xml")


    def loadProjector(self):
        sensor_name = "Projector"
        sensor_yaml_path = os.path.join(self.firefly_path, sensor_name + ".yaml")
        sensor_config = utils.read_config_yaml(sensor_yaml_path)
        
        self.projector = Transformable.Transformable(sensor_name, sensor_config, self._device)
        self._parent_transformables.append(self.projector)

    
    def loadCameras(self):
        sensor_name = "Camera"
        sensor_yaml_path = os.path.join(self.firefly_path, sensor_name + ".yaml")
        sensor_config = utils.read_config_yaml(sensor_yaml_path)

        self.camera = Transformable.Transformable(sensor_name, sensor_config, self._device)
        self._parent_transformables.append(self.camera)
        
    
    def loadLights(self):
        # TODO: Implement me
        a = 1
        pass


    def loadMeshes(self):
        meshes = self.mi_xml.find_all('shape')
        param_mesh = "PLYMesh"

        count = 0
        for mesh in meshes:
            if mesh.find("emitter") is not None:
                continue

            temp_param_mesh = param_mesh    
            if count > 0:
                temp_param_mesh += "_{0}".format(count)

            mesh_name = self.getMeshName(mesh)
            mesh_yaml_path = os.path.join(self.firefly_path, mesh_name + ".yaml")
            mesh_config = utils.read_config_yaml(mesh_yaml_path)

            if not mesh_config["randomizable"]:
                continue
            
            # Object is randomizable => Create randomizable object, and connect it to the mitsuba parameter.
            if "is_flame" in mesh_config and mesh_config["is_flame"]:
                self.meshes[temp_param_mesh] = Transformable.FlameShapeModel(name = mesh_name, 
                                                              config=mesh_config, 
                                                              vertex_data =self.scene_params[temp_param_mesh + ".vertex_positions"], 
                                                              sequential_animation=self._sequential_animation,
                                                              base_path=self.firefly_path,
                                                              device=self._device)
            else:
                self.meshes[temp_param_mesh] = Transformable.Mesh(name = mesh_name, 
                                                                config=mesh_config, 
                                                                vertex_data =self.scene_params[temp_param_mesh + ".vertex_positions"], 
                                                                sequential_animation=self._sequential_animation,
                                                                base_path=self.firefly_path,
                                                                device=self._device)
            


            self._parent_transformables.append(self.meshes[temp_param_mesh])
            count += 1


    def loadCurves(self):
        nurbs_files = [f for f in os.listdir(self.firefly_path) 
                       if os.path.isfile(os.path.join(self.firefly_path, f)) 
                       and 'path' in f.lower()]

        for nurbs_path in nurbs_files:
            yaml_path = os.path.join(self.firefly_path, nurbs_path)
            config = utils.read_config_yaml(yaml_path)

            object_name = os.path.splitext(nurbs_path)[0]
            curve = utils.importBlenderNurbsObj(os.path.join(self.firefly_path, object_name, object_name + '.obj'))
            transformable_curve = Transformable.Curve(object_name, curve, config, self._device)

            self.curves.append(transformable_curve)
            self._parent_transformables.append(transformable_curve)



    def connectParents(self):
        for a in self._parent_transformables:
            if not a.relative():
                continue

            for b in self._parent_transformables:
                if a == b:
                    continue

                # CHECK FOR RELATIVES HERE
                if a.parentName() == b.name():
                    a.setParent(b)


    def cleanParentTransformables(self) -> None:
        self._parent_transformables = [transformable for transformable in self._parent_transformables if transformable.parent() is None]


    def initScene(self) -> None:
        self.loadMeshes()
        self.loadCameras()
        self.loadProjector()
        self.loadLights()
        self.loadCurves()
        
        # Connect relative objects
        self.connectParents()

        # Filter non-parent objects from transformable list
        self.cleanParentTransformables()


    def updateMeshes(self) -> None:
        for key, mesh in self.meshes.items():
            rand_verts, faces = mesh.getVertexData()
            self.scene_params[key + ".vertex_positions"] = mi.Float32(rand_verts.flatten())

            if faces is not None:
                self.scene_params[key + ".faces"] = mi.UInt32(faces.flatten())

            if mesh.animated():
                if self._num_updates % self._steps_per_frame == 0:
                    mesh.next_anim_step()


    def updateCamera(self) -> None:
        if self.camera is None:
            return

        # TODO: Remove key
        key = "PerspectiveCamera"

        # Couldn't find a better way to get this torch tensor into mitsuba Transform4f
        worldMatrix = self.camera.world()
        worldMatrix[0:3, 0:3] = worldMatrix[0:3, 0:3] @ math.getYTransform(np.pi, self._device)
        #worldMatrix[0:3, 0:3] = worldMatrix[0:3, 0:3] 
        self.scene_params[key + ".to_world"] = mi.Transform4f(worldMatrix.tolist())


    def updateProjector(self) -> None:
        if self.projector is None:
            return
        
        # TODO: Remove key
        key = "Projector"
        worldMatrix = self.projector.world()
        worldMatrix[0:3, 0:3] = worldMatrix[0:3, 0:3] @ math.getYTransform(np.pi, self._device)

        # TODO: Is there a better way here?
        # Couldn't find a better way to get this torch tensor into mitsuba Transform4f
        self.scene_params[key + ".to_world"] = mi.Transform4f(worldMatrix.tolist())


    def updateLights(self) -> None:
        # TODO: Implement me
        return None


    def randomize(self) -> None:
        # We first randomize all of our objects
        for transformable in self._parent_transformables:
            transformable.randomize()

            iterator_child = transformable.child()
            while iterator_child is not None:
                iterator_child.randomize()
                iterator_child = iterator_child.child()

        # And then copy the updates to the mitsuba parameters
        self.updateMeshes()
        self.updateCamera()
        self.updateProjector()
        self.updateLights()

        # We finally update the mitsuba scene graph itself
        self.scene_params.update()
        self._num_updates += 1


    def getMeshName(self, mesh) -> str:
        for child in mesh.find_all("string"):
            if child.has_attr("name") and child.attrs["name"] == "filename":
                return Path(child.attrs["value"]).stem
        
        return None
    

def generate_epipolar_shadow(scene):
    pass


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import cv2

    base_path = "scenes/FLAME/"

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    #mitsuba_params["PerspectiveCamera.film.size"] = [1080//2, 1080//2]
    mitsuba_params['Projector.to_world'] = mitsuba_params['PerspectiveCamera_1.to_world']
    print(mitsuba_params['PerspectiveCamera_1.to_world'])



    render = mi.render(mitsuba_scene, spp=20)
    import matplotlib.pyplot as plt

    plt.axis("off")
    plt.imshow(render ** (1.0 / 2.2))
    plt.show()

    firefly_scene = Scene(mitsuba_params, base_path)



    for i in tqdm(range(100000)):
        firefly_scene.randomize()
        render = mi.render(mitsuba_scene, spp=10)
        render = torch.clamp(render.torch(), 0, 1)[:, :, [2, 1, 0]].cpu().numpy()
        cv2.imshow("Render.png", render)
        cv2.waitKey(1)

 