import torch
import matplotlib.pyplot as plt


# We assume points to be in NDC [-1, 1]
def rasterize_points(points: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((points.shape[0], 1, 1, 1))

    # Convert points to 0 -> 1
    points = points*0.5 + 0.5
    
    
    # Somewhere between [0, texture_size] but in float
    points *= texture_size

    # Generate x, y indices
    y, x = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = (y_dist*y_dist + x_dist*x_dist).sqrt() / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances, 2) / (2 * sigma * sigma))

    point_distances = point_distances.sum(dim=0)
    point_distances = point_distances - point_distances.min()
    #a_points_distances = point_distances - point_distances.min()
    #b_point_distances = a_points_distances / a_points_distances.max()

    return torch.clamp(point_distances, min=0.0, max=1.0)


# We assume points to be in NDC [-1, 1]
def rasterize_depth(points: torch.tensor, depth_vals: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((points.shape[0], 1, 1, 1))

    # Convert points to 0 -> 1
    points = points*0.5 + 0.5
    
    
    # Somewhere between [0, texture_size] but in float
    points *= texture_size

    # Generate x, y indices
    y, x = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = (y_dist*y_dist + x_dist*x_dist).sqrt() / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances, 2) / (2 * sigma * sigma))

    # normalize
    point_distances = point_distances / point_distances.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]

    # scale by depth in range [0, 1]
    return point_distances * depth_vals.unsqueeze(1)


def rasterize_lines(lines: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    # lines are in NDC

    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((lines.shape[0], 1, 1, 1))

    # Convert points to 0 -> 1
    lines = lines*0.5 + 0.5

    lines_start = lines[:, 0:2]
    lines_end = lines[:, 2:4]
    
    # Somewhere between [0, texture_size] but in float
    lines_start *= texture_size
    lines_end *= texture_size

    lines_start = lines_start.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
    lines_end = lines_end.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
    

    y, x = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((lines.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((lines.shape[0], 1, 1))   
    xy = torch.stack([x, y])

    m = lines_end - lines_start
    t = ((xy - lines_start) * m) / (m*m + torch.finfo().eps)

    distance_smaller_zero = torch.linalg.norm(torch.where(t <= 0, 1, 0) * (xy - lines_start), dim=0)
    distance_inbetween = torch.linalg.norm(torch.bitwise_and(torch.where(t > 0, 1, 0), torch.where(t < 1, 1, 0)) * (xy - (lines_start + t * m)), dim=0)
    distance_greater_one = torch.linalg.norm(torch.where(t >= 1, 1, 0) * (xy - lines_end), dim=0)

    distances = distance_smaller_zero + distance_inbetween + distance_greater_one

    #distances = torch.exp(-torch.linalg.norm(distances, dim=0) / sigma*sigma)

    return torch.exp(-(distances*distances) / (sigma * sigma))





def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")


    points = (torch.rand([1000, 2], device=device) - 0.5) * 2.0
    lines = (torch.rand([1, 4], device=device) - 0.5) * 2.0
    sigma = 6
    texture_size = torch.tensor([512, 512], device=device)


    #point_texture = rasterize_points(points, sigma, texture_size)
    line_texture = rasterize_lines(lines, sigma, texture_size)
    
    #scene_init = mi.load_file("scenes/proj_cbox.xml", spp=1024)
    #params = mi.traverse(scene_init)
    
    #params["tex.data"] = mi.TensorXf(texture.cuda().unsqueeze(-1).repeat(1, 1, 3))
    #params.update()

    #render_init = mi.render(scene_init, spp=1024)
    #image_init = mi.util.convert_to_bitmap(render_init)

    #plt.axis("off")
    #plt.title("GT")
    #plt.imshow(point_texture.detach().cpu().numpy())
    #plt.show(block=True)

    plt.axis("off")
    plt.title("GT")
    plt.imshow(line_texture[0].detach().cpu().numpy())
    plt.show(block=True)



if __name__ == "__main__":
    main()