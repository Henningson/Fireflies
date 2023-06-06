import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import hello_world
import cv2
import numpy as np
import torch
import entity
import imageio
from tqdm import tqdm
import utils_np

def main():
    scene   = mi.load_file("TestScene/test_scene.xml", spp=16)
    params = mi.traverse(scene)

    rand_obj = entity.RandomizableEntity("test_random_object.yaml")
    verts_base = params["PLYMesh.vertex_positions"]
    rand_obj.setVertices(torch.tensor(verts_base).reshape(-1, 3).tolist())


    difference_depth_maps = []
    writer = imageio.get_writer('test.mp4', fps=25)
    for i in tqdm(range(100)):
        vertex_data = rand_obj.getVertexData()
        params["PLYMesh.vertex_positions"] = mi.Float32(vertex_data.flatten())
        params.update()
        depth_map_a = hello_world.get_depth_map(scene, spp=1)
        
        vertex_data = rand_obj.getVertexData()
        params["PLYMesh.vertex_positions"] = mi.Float32(vertex_data.flatten())
        params.update()
        depth_map_b = hello_world.get_depth_map(scene, spp=1)
        
        diff_depth_map = np.abs(depth_map_a - depth_map_b)
        difference_depth_maps.append(diff_depth_map)

        variance_map = np.stack(difference_depth_maps).var(axis=0)
        normalized_variance_map = utils_np.normalize(variance_map)
        prob_distribution = variance_map / variance_map.sum()

        try:
            candidates = np.arange(0, variance_map.size)
            chosen_points = np.random.choice(candidates, 500, p=prob_distribution.flatten(), replace=False)
            colormap = cv2.applyColorMap((normalized_variance_map*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            colormap.reshape(-1, 3)[chosen_points, :] = [0, 0, 0]
            writer.append_data(colormap)
        except:
            continue
    writer.close()

if __name__ == "__main__":
    main()