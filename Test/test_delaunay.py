import torch
import scipy
import Utils.utils as utils
import Utils.transforms as transforms
import pytorch3d.structures
import pytorch3d.ops
import pytorch3d.loss


def main():
    N = 100
    DEVICE = "cuda"

    im_size = [512, 512]

    # Create pointcloud
    start_pc = torch.rand([N, 3], device=DEVICE)
    start_pc.requires_grad = True
    random_pc = torch.rand([N, 3], device=DEVICE)

    # Create Intrinsic Camera Matrix
    K = utils.build_projection_matrix(90.0, 0.0001, 1.0, DEVICE)

    optim = torch.optim.Adam([start_pc])


    for _ in range(10000):
        optim.zero_grad()

        # Create delaunay triangulation
        with torch.no_grad():
            # Project pointcloud to camera
            transformed_pc = transforms.transform_points(start_pc, K)[:, 0:2]
            pos = transformed_pc.cpu().numpy()
            tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
            face = torch.from_numpy(tri.simplices)
            face = face.contiguous().to(DEVICE, torch.long)

        # Sample from triangulation
        pc_mesh = pytorch3d.structures.Meshes([start_pc], [face])
        new_pc = pytorch3d.ops.sample_points_from_meshes(pc_mesh, N)

        loss = pytorch3d.loss.chamfer_distance(new_pc, random_pc.unsqueeze(0))[0]
        print(loss)
        loss.backward()



if __name__ == "__main__":
    main()