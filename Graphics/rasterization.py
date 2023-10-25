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
    x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = (y_dist*y_dist + x_dist*x_dist).sqrt()# / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances, 2) / (sigma * sigma))


    return point_distances


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
    x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = (y_dist*y_dist + x_dist*x_dist).sqrt() / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances, 2) / (sigma * sigma))

    # normalize
    point_distances = point_distances / point_distances.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]

    # scale by depth in range [0, 1]
    return point_distances * depth_vals.unsqueeze(1)


def rasterize_lines(lines: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    # lines are in NDC [-1,  1]
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((lines.shape[0], 1, 1, 1))

    # Convert points to 0 -> 1
    lines = lines*0.5 + 0.5

    lines_start = lines[:, 0, :]
    lines_end = lines[:, 1, :]
    
    # Somewhere between [0, texture_size] but in float
    lines_start *= texture_size
    lines_end *= texture_size

    lines_start = lines_start.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
    lines_end = lines_end.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
    
    y, x = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((lines.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((lines.shape[0], 1, 1))   
    xy = torch.stack([x, y])



    # See: https://github.com/jonhare/DifferentiableSketching/blob/main/dsketch/raster/disttrans.py
    # If you found this, you should definitely give them a star. That's beautiful code they wrote there.

    pa = (xy - lines_start)
    pb = (xy - lines_end)
    m = lines_end - lines_start

    t0 = (pa * m).sum(dim=0) / ((m * m).sum(dim=0) + torch.finfo().eps)
    patm = xy - (lines_start + t0.unsqueeze(0) * m)

    distance_smaller_zero = (t0 <= 0) * (pa * pa).sum(dim=0)
    distance_inbetween = (t0 > 0) * (t0 < 1) * (patm * patm).sum(dim=0)
    distance_greater_one = (t0 >= 1) * (pb * pb).sum(dim=0)

    distances = distance_smaller_zero + distance_inbetween + distance_greater_one
    #distances = distances.sqrt()
    return torch.exp(-(distances*distances) / (sigma * sigma))


def softor(texture: torch.tensor, dim=0, keepdim: bool = False) -> torch.tensor:  
    return 1 - torch.prod(1 - texture, dim=dim, keepdim=keepdim)


def test_point_reg():
    import cv2
    import numpy as np
    from tqdm import tqdm
    from pygifsicle import optimize
    import imageio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    points = (torch.rand([500, 2], device=device) - 0.5) * 2.0
    points.requires_grad = True
    sigma = torch.tensor([100.0], device=device)
    texture_size = torch.tensor([512, 512], device=device)
    loss_func = torch.nn.L1Loss()

    opt_steps = 200

    optim = torch.optim.Adam([
        {'params': points,  'lr': 0.05}
        ])
    
    images = []
    for i in tqdm(range(opt_steps)):
        optim.zero_grad()

        rasterized_points = rasterize_points(points, sigma, texture_size)

        softored = softor(rasterized_points)
        summed = rasterized_points.sum(dim=0)

        loss = -loss_func(softored, summed)
        loss.backward()
        optim.step()

        with torch.no_grad():
            points[points > 1.0] = 1.0
            points[points < -1.0] = -1.0
            np_points = cv2.applyColorMap((softored.detach().cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            images.append(np_points[:, :, [2, 1, 0]])
            cv2.imshow("Optim Lines", np_points)
            cv2.waitKey(1)
            #lines.requires_grad = True
    imageio.v3.imwrite("point_regularization.mp4", np.stack(images, axis=0), fps=25)



def test_line_reg():
    import cv2
    import numpy as np
    from tqdm import tqdm
    from pygifsicle import optimize
    import imageio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    lines = (torch.rand([50, 2, 2], device=device) - 0.5) * 2.0
    lines.requires_grad = True
    sigma = torch.tensor([200.0], device=device)
    texture_size = torch.tensor([512, 512], device=device)
    loss_func = torch.nn.L1Loss()

    opt_steps = 1000

    optim = torch.optim.Adam([
        {'params': lines,  'lr': 0.005},
        {'params': sigma,  'lr': 1.0},
        ])
    
    images = []
    for i in tqdm(range(opt_steps)):
        optim.zero_grad()

        rasterized_lines = rasterize_lines(lines, sigma, texture_size)

        softored = softor(rasterized_lines)
        summed = rasterized_lines.sum(dim=0)

        loss = -loss_func(softored, summed)
        loss.backward()
        optim.step()

        with torch.no_grad():
            lines[lines > 1.0] = 1.0
            lines[lines < -1.0] = torch.rand(1, device=device)
            np_lines = cv2.applyColorMap((softored.detach().cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            images.append(np_lines[:, :, [2, 1, 0]])
            cv2.imshow("Optim Lines", np_lines)
            cv2.waitKey(1)
            #lines.requires_grad = True
    #imageio.v3.imwrite("line_regularization.mp4", np.stack(images, axis=0), fps=25)
    #optimize("line_regularization.gif")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    points = (torch.rand([100, 2], device=device) - 0.5) * 2.0
    lines = (torch.rand([30, 2, 2], device=device) - 0.5) * 2.0
    texture_size = torch.tensor([512, 512], device=device)
    
    sigma = 1000

    line_texture = rasterize_lines(lines, sigma, texture_size, device=device)
    line_texture = softor(line_texture)

    point_texture = rasterize_points(points, sigma, texture_size, device=device)
    point_texture = softor(point_texture)

    texture = torch.hstack([point_texture, line_texture])

    plt.axis("off")
    plt.title("GT")
    plt.imshow(texture.detach().cpu().numpy())
    plt.show(block=True)

def test_square_reg():
    import cv2
    import numpy as np
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sigma = torch.tensor([200.0], device=device)
    texture_size = torch.tensor([512, 512], device=device)
    loss_func = torch.nn.L1Loss()


    square_centroids = torch.rand([50, 2], device=device) * texture_size
    square_sizes = torch.rand([50, 1], device=device) * texture_size[0]

    square_centroids.requires_grad = True



    opt_steps = 1000

    optim = torch.optim.Adam([
        {'params': square_centroids,  'lr': 0.005},
        {'params': sigma,  'lr': 1.0},
        ])
    
    images = []
    for i in tqdm(range(opt_steps)):
        optim.zero_grad()

        rasterized_lines = rasterize_squares(square_centroids, square_sizes, sigma, texture_size)

        softored = softor(rasterized_lines)
        summed = rasterized_lines.sum(dim=0)

        loss = -loss_func(softored, summed)
        loss.backward()
        optim.step()

        with torch.no_grad():
            square_centroids[square_centroids > texture_size[0] - 1] = texture_size[0] - 1
            square_centroids[square_centroids < 0.0] = 0.0
            np_lines = cv2.applyColorMap((softored.detach().cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            images.append(np_lines[:, :, [2, 1, 0]])
            cv2.imshow("Optim Lines", np_lines)
            cv2.waitKey(1)

if __name__ == "__main__":
    test_point_reg()
    #test_line_reg()
    #main()