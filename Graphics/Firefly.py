from bs4 import BeautifulSoup
from pathlib import Path
import os
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import Objects.entity as entity
import Utils.utils as utils
import torch

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
        # TODO: Allow loading of multiple cameras
        #param_camera = "PerspectiveCamera"

        # In this case, we enfore the name Camera
        sensor_name = "Projector"
        sensor_yaml_path = os.path.join(self.firefly_path, sensor_name + ".yaml")
        sensor_config = utils.read_config_yaml(sensor_yaml_path)

        if self.camera is None:
            return
        
            # Object is randomizable => Create randomizable object, and connect it to the parameter.
        self.projector = entity.Projector(sensor_config, self._device)
        self.projector.setParent(self.camera)

    
    def loadCameras(self):
        # TODO: Allow loading of multiple cameras
        #param_camera = "PerspectiveCamera"

        # In this case, we enfore the name Camera
        sensor_name = "Camera"
        sensor_yaml_path = os.path.join(self.firefly_path, sensor_name + ".yaml")
        sensor_config = utils.read_config_yaml(sensor_yaml_path)

        if not sensor_config["randomizable"]:
            return

            # Object is randomizable => Create randomizable object, and connect it to the parameter.
        self.camera = entity.RandomizableCamera(sensor_config, self._device)
        
    
    def loadLights(self):
        # TODO: Implement me
        a = 1
        pass


    def loadMeshes(self):
        meshes = self.mi_xml.find_all('shape')
        param_mesh = "PLYMesh"
        for count, mesh in enumerate(meshes):
            temp_param_mesh = param_mesh    
            if count > 0:
                temp_param_mesh += "_{0}".format(count)

            mesh_name = self.getMeshName(mesh)
            mesh_yaml_path = os.path.join(self.firefly_path, mesh_name + ".yaml")
            mesh_config = utils.read_config_yaml(mesh_yaml_path)

            if not mesh_config["randomizable"]:
                continue
            
            # Object is randomizable => Create randomizable object, and connect it to the mitsuba parameter.
            self.meshes[temp_param_mesh] = entity.Randomizable(self.firefly_path, 
                                                                            mesh_name, 
                                                                            mesh_config, 
                                                                            self.scene_params[temp_param_mesh + ".vertex_positions"], 
                                                                            self._sequential_animation,
                                                                            self._device)


    def initScene(self) -> None:
        self.loadMeshes()
        self.loadCameras()
        self.loadProjector()
        self.loadLights()


    def randomizeMeshes(self) -> None:
        for key, mesh in self.meshes.items():

            rand_verts, rand_faces = mesh.getVertexData()
            self.scene_params[key + ".vertex_positions"] = mi.Float32(rand_verts.flatten())

            if rand_faces is not None:
                self.scene_params[key + ".faces"] = mi.UInt32(rand_faces.cpu().numpy())

            if mesh.is_animated():
                if self._num_updates % self._steps_per_frame == 0:
                    mesh.next_anim_step()


    def randomizeCamera(self) -> None:
        if self.camera is None:
            return

        key = "PerspectiveCamera"
        worldMatrix = self.camera.getTransforms()

        # TODO: Is there a better way here?
        # Couldn't find a better way to get this torch tensor into mitsuba Transform4f
        self.scene_params[key + ".to_world"] = mi.Transform4f(worldMatrix.tolist())


    def randomizeProjector(self) -> None:
        if self.projector is None:
            return
        
        if self.projector._parent is None:
            return
        
        # TODO: Get rid of hardcoded stuff here.
        key = "Projector"
        worldMatrix = self.projector.getTransforms()

        # TODO: Is there a better way here?
        # Couldn't find a better way to get this torch tensor into mitsuba Transform4f
        self.scene_params[key + ".to_world"] = mi.Transform4f(worldMatrix.tolist())


    def randomizeLights(self) -> None:
        # TODO: Implement me
        return None


    def randomize(self) -> None:
        self.randomizeMeshes()
        self.randomizeCamera()
        self.randomizeProjector()
        self.randomizeLights()        
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

    base_path = "scenes/EasyCube/"
    sequential = True

    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)
    mitsuba_params["PerspectiveCamera.film.size"] = [32, 32]
    firefly_scene = Scene(mitsuba_params, base_path, sequential_animation=sequential)


    for i in tqdm(range(100000)):
        firefly_scene.randomize()
        mitsuba_params.update()
        render = mi.render(mitsuba_scene, spp=1)
        image = mi.util.convert_to_bitmap(render)