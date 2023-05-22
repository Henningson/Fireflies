import torch
import matplotlib.pyplot as plt


# We assume points to be in NDC [-1, 1]
def rasterize_points(points: torch.tensor, sigma: float, texture_size: torch.tensor) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32)
    tex = tex[None, ...]
    tex = tex.repeat((points.shape[0], 1, 1, 1))

    # Convert points to 0 -> 1
    points = points*0.5 + 0.5
    
    
    # Somewhere between [0, texture_size] but in float
    points *= texture_size

    # Generate x, y indices
    y, x = torch.meshgrid(torch.arange(0, texture_size[1]), torch.arange(0, texture_size[0]), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    
    point_distances = (y_dist*y_dist + x_dist*x_dist).sqrt() / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances, 2) / (2 * sigma * sigma))

    point_distances = point_distances.sum(dim=0)
    point_distances -= point_distances.min()
    point_distances /= point_distances.max()

    return point_distances



def main():
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")


    points = (torch.rand([1000, 2]) - 0.5) * 2.0
    sigma = 0.001
    texture_size = torch.tensor([512, 512])


    texture = rasterize_points(points, sigma, texture_size)
    scene_init = mi.load_file("scenes/proj_cbox.xml", spp=1024)
    params = mi.traverse(scene_init)
    
    params["tex.data"] = mi.TensorXf(texture.cuda().unsqueeze(-1).repeat(1, 1, 3))
    params.update()

    render_init = mi.render(scene_init, spp=1024)
    image_init = mi.util.convert_to_bitmap(render_init)



    print("Init | GT | Depth")
    plt.axis("off")
    plt.title("GT")
    plt.imshow(image_init)
    plt.show(block=True)



if __name__ == "__main__":
    main()