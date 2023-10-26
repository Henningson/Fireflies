import os
import cv2
import torch
import Objects.intersections as intersections
import Objects.laser as laser
import Objects.Transformable as Transformable
import Graphics.rasterization as rasterization

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import mitsuba as mi

if DEVICE == "cuda":
    mi.set_variant("cuda_ad_rgb")
else:
    mi.set_variant("llvm_ad_rgb")

import drjit as dr



class LaserPostProcessor:
    def __init__(self, firefly_scene, mitsuba_scene, device):
        self._firefly_scene = firefly_scene
        self._mitsuba_scene = mitsuba_scene
        self._device = device
   
    def removeClosePoints(self, laser, sigma, tex_size):
        points = laser.projectRaysToNDC()[:, 0:2]
        points *= tex_size.unsqueeze(0)
        sigma_tensor = torch.tensor(sigma, device=self._device).unsqueeze(0)

        merge_candidates = []
        for i in range(points.shape[0]):
                 hit = intersections.sphereSphere(points[i].unsqueeze(0), sigma_tensor, points[j].unsqueeze(0), sigma_tensor)

                 if hit.any():
                     merge_candidates.append((i, j))

        print("Number of merges in this iteration: {0}".format(len(merge_candidates)))

        merged_rays = []
        rays = laser._rays
        for candidate_a, candidate_b in merge_candidates:
            merged_ray = (rays[candidate_a] + rays[candidate_b]) / 2.0
            merged_rays.append(merged_ray)

        merged_rays = torch.stack(merged_rays)
        candidates = torch.tensor(merge_candidates, device=DEVICE).unique()
        indices = torch.arange(0, rays.shape[0], 1, device=DEVICE)
        non_colliding_ray_indices = ~(indices.unsqueeze(-1).repeat(1, candidates.shape[0]) == candidates.unsqueeze(0)).any(dim=1)

        new_rays = torch.concatenate([rays[non_colliding_ray_indices], merged_rays])
        return new_rays
   
    def newRemoveClosePoints(self, laser, sigma, tex_size):

        hit_found = True
        rays = laser._rays.clone()
        rays_old = laser._rays.clone()

        while hit_found:

            i = 0
            while i < rays.shape[0]:
                print(str(i) + " | " + str(rays.shape[0]))
                laser._rays = rays
                points = laser.projectRaysToNDC()[:, 0:2]
                points *= tex_size.unsqueeze(0)
                sigma_tensor = torch.tensor(sigma, device=self._device).unsqueeze(0).repeat(rays.shape[0], 1)

                indices = torch.arange(0, rays.shape[0], device=DEVICE)

                points = points[indices]
                points_only_i = points[i].unsqueeze(0).repeat(rays.shape[0], 1)

                hits = intersections.sphereSphere(points, sigma_tensor, points_only_i, sigma_tensor)[:, 0]
                
                # Set i-th hit to False when its the only hit that was found
                if len(hits.nonzero()) == 1:
                    hits[i] = False

                # If we do not have any hits continue
                if not hits.any():
                     hit_found = False
                     i += 1
                     continue

                hit_found = True

                support_rays = rays[hits]
                new_ray = support_rays.sum(dim=0, keepdim=True) / support_rays.shape[0]

                # Remove close rays
                rays = rays[~hits]
                rays = torch.cat([rays, new_ray])
                i = 0

        return rays
        






                


        for i in range(points.shape[0]):
                 hit = intersections.sphereSphere(points[i].unsqueeze(0), sigma_tensor, points[j].unsqueeze(0), sigma_tensor)

                 if hit.any():
                     merge_candidates.append((i, j))

        print("Number of merges in this iteration: {0}".format(len(merge_candidates)))

        merged_rays = []
        rays = laser._rays
        for candidate_a, candidate_b in merge_candidates:
            merged_ray = (rays[candidate_a] + rays[candidate_b]) / 2.0
            merged_rays.append(merged_ray)

        merged_rays = torch.stack(merged_rays)
        candidates = torch.tensor(merge_candidates, device=DEVICE).unique()
        indices = torch.arange(0, rays.shape[0], 1, device=DEVICE)
        non_colliding_ray_indices = ~(indices.unsqueeze(-1).repeat(1, candidates.shape[0]) == candidates.unsqueeze(0)).any(dim=1)

        new_rays = torch.concatenate([rays[non_colliding_ray_indices], merged_rays])
        return new_rays






if __name__ == "__main__":
    import numpy as np
    import cv2

    config = {
        "translation": {'min_x': 0.0, 'min_y': 0.0, 'min_z': 0.0, 'max_x': 0.0, 'max_y': 0.0, 'max_z': 0.0}, 
        "rotation": {'min_x': 0.0, 'min_y': 0.0, 'min_z': 0.0, 'max_x': 0.0, 'max_y': 0.0, 'max_z': 0.0}, 
        "is_relative": False,
        "randomizable": True,
        "parent_name": None,
        "to_world": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}
    
    x_beams = 18
    y_beams = 18
    num_beams = x_beams * y_beams
    sigma = 5.0
    sigma_tensor = torch.tensor(sigma).unsqueeze(0)

    tex_size = torch.tensor([512, 512], device=DEVICE)

    transform = Transformable.Transformable("Laser", config, DEVICE)
    laser_test = laser.Laser(transform, laser.Laser.generate_uniform_rays(0.015, x_beams, y_beams, DEVICE), 20, 0.01, 30.0, DEVICE)
    laser_test._rays[:, 0:2] += torch.rand(laser_test._rays.shape[0], 2, device=DEVICE) * 0.5
    laser_test.randomize_out_of_bounds()


    diff = torch.ones(10, device=DEVICE)
    
    while diff.sum() != 0:
        tex = laser_test.generateTexture(sigma, tex_size)
        tex = rasterization.softor(tex).detach().cpu().numpy()
        cv2.imshow("Tex", tex)
        cv2.waitKey(1)


        lpp = LaserPostProcessor(None, None, DEVICE)
        laser_test._rays = lpp.newRemoveClosePoints(laser_test, 4*sigma, tex_size)


        new_tex = laser_test.generateTexture(sigma, tex_size)
        new_tex = rasterization.softor(new_tex).detach().cpu().numpy()
        cv2.imshow("New Tex", new_tex)
        cv2.waitKey(1)


        diff = ((tex-new_tex)*255).astype(np.uint8)
        cv2.imshow("Diff", cv2.applyColorMap(diff, cv2.COLORMAP_JET))
        cv2.waitKey(0)
    


        '''
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(a_coords.shape[0]):
            cv2.circle(image, (a_coords[i].detach().cpu().numpy()*512).astype(np.int), (a_radius[i,0].detach().cpu().numpy()*512).astype(np.int), color= (0, 0, 255) if intersections[i, 0] else (0, 255, 0))
            cv2.circle(image, (b_coords[i].detach().cpu().numpy()*512).astype(np.int), (b_radius[i,0].detach().cpu().numpy()*512).astype(np.int), color= (0, 0, 255) if intersections[i, 0] else (0, 255, 0))

        cv2.imshow("Test", image)
        cv2.waitKey(0)'''