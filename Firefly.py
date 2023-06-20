from bs4 import BeautifulSoup
from pathlib import Path
import utils_io
import os
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
dr.set_flag(dr.JitFlag.LoopRecord, False)
import entity
import torch

class Scene:
    def __init__(self, scene_params, base_path: str, device: torch.cuda.device = torch.device("cuda")):
        self.mi_xml = self.getMitsubaXML(base_path + "scene.xml")
        self.firefly_path = os.path.join(base_path, "Firefly")
        self.scene_params = scene_params

        self.base_path = base_path
        self.customizable_meshes = {}
        self.customizable_camera = None
        self.customizable_lights = {}
        self._device = device

        self.initScene()


    def getMitsubaXML(self, path):
        data = None
        with open(path, 'r') as f:
            data = f.read()
        return BeautifulSoup(data, "xml")

    
    def loadRandomizableCamera(self):
        # TODO: Implement me
        pass

    
    def loadRandomizableLights(self):
        # TODO: Implement me
        pass


    def loadRandomizableMeshes(self):
        meshes = self.mi_xml.find_all('shape')
        param_mesh = "PLYMesh"
        for count, mesh in enumerate(meshes):
            temp_param_mesh = param_mesh    
            if count > 0:
                temp_param_mesh += "_{0}".format(count)

            mesh_name = self.getMeshName(mesh)
            mesh_yaml_path = os.path.join(self.firefly_path, mesh_name + ".yaml")
            mesh_config = utils_io.read_config_yaml(mesh_yaml_path)

            if not mesh_config["randomizable"]:
                continue

            # Object is randomizable => Create randomizable object, and connect it to the parameter.
            self.customizable_meshes[temp_param_mesh] = entity.Randomizable(self.firefly_path, 
                                                                            mesh_name, 
                                                                            mesh_config, 
                                                                            self.scene_params[temp_param_mesh + ".vertex_positions"], 
                                                                            self._device)


    def initScene(self):
        self.loadRandomizableMeshes()
        self.loadRandomizableCamera()
        self.loadRandomizableLights()


    def randomizeMeshes(self):
        for key, mesh in self.customizable_meshes.items():
            rand_verts, rand_faces = mesh.getVertexData()
            self.scene_params[key + ".vertex_positions"] = mi.Float32(rand_verts.flatten())

            if rand_faces is not None:
                self.scene_params[key + ".faces"] = mi.UInt32(rand_faces.cpu().numpy())


    def randomizeCamera(self):
        # TODO: Implement me
        return None


    def randomizeLights(self):
        # TODO: Implement me
        return None


    def randomize(self):
        self.randomizeMeshes()
        self.randomizeCamera()
        self.randomizeLights()        
        self.scene_params.update()


    def getMeshName(self, mesh) -> str:
        for child in mesh.find_all("string"):
            if child.has_attr("name") and child.attrs["name"] == "filename":
                return Path(child.attrs["value"]).stem
        
        return None
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    base_path = "/home/nu94waro/Desktop/TestMitsubaScene/"
    mitsuba_scene = mi.load_file(os.path.join(base_path, "scene.xml"))
    mitsuba_params = mi.traverse(mitsuba_scene)

    obj_path = os.path.join(base_path, "Firefly", "Cloth",  "Cloth_000000.obj")
    #scene = pywavefront.Wavefront(obj_path, collect_faces=True)

    render_one = mi.render(mitsuba_scene, spp=1)
    image_one = mi.util.convert_to_bitmap(render_one)

    scene = FireflyScene(mitsuba_params, base_path)
    scene.randomize()


    render_two = mi.render(mitsuba_scene, spp=1)
    image_two = mi.util.convert_to_bitmap(render_two)
    print("Init | GT | Depth")
    plt.axis("off")
    plt.title("GT")
    plt.imshow(np.hstack([image_one, image_two]))
    plt.show(block=True)