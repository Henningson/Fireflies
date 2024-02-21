import torch
import matplotlib.pyplot as plt
import numpy as np
import math

torch.manual_seed(0)

# We assume points to be in camera space [0, 1]
def rasterize_points(points: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((points.shape[0], 1, 1, 1))

    
    # Somewhere between [0, texture_size] but in float
    points = points.clone() * texture_size

    # Generate x, y indices
    x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = (y_dist*y_dist + x_dist*x_dist)# / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances / sigma, 2))


    return point_distances



def rasterize_points_in_non_ndc(points: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = (y_dist*y_dist + x_dist*x_dist)# / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances / sigma, 2))


    return point_distances


# We assume points to be in NDC [-1, 1]
def rasterize_depth(points: torch.tensor, depth_vals: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((points.shape[0], 1, 1, 1))
    
    
    # Somewhere between [0, texture_size] but in float
    points = points.clone() * texture_size

    # Generate x, y indices
    x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = y_dist*y_dist + x_dist*x_dist #/ (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances / sigma, 2))

    # normalize
    point_distances = point_distances / point_distances.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]

    # scale by depth in range [0, 1]
    return point_distances * depth_vals.unsqueeze(1)


def rasterize_lines(lines: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    # lines are in NDC [-1,  1]
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((lines.shape[0], 1, 1, 1))

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


def sum(texture: torch.tensor, dim=0, keepdim: bool = False) -> torch.tensor:
    return torch.sum(texture, dim=dim, keepdim=keepdim)


def baked_sum(points: torch.tensor, sigma: torch.tensor, texture_size: torch.tensor, num_std: int = 4, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    
    # Somewhere between [0, texture_size] but in float
    points = points.clone() * texture_size

    for i in range(points.shape[0]):
        point = points[i]

        # We use 3*sigma^2 here, to include most of the gaussian
        footprint = math.floor(sigma.sqrt().item()) * num_std
        footprint = footprint + 1 if footprint % 2 == 0 else footprint
        half_footprint = int((footprint - 1) / 2)

        point_in_middle_of_footprint = point - point.floor() + half_footprint

        footprint_origin_in_original_image = (point - half_footprint).floor() # [Y, X]

        y, x = torch.meshgrid(torch.arange(0, footprint, device=device), torch.arange(0, footprint, device=device), indexing='ij')

        y_dist = y - point_in_middle_of_footprint[0:1]
        x_dist = x - point_in_middle_of_footprint[1:2]
        dist = y_dist*y_dist + x_dist*x_dist
        dist = torch.exp(-torch.pow(dist / sigma, 2))
        
        wo = (point.floor() - half_footprint).int()
        re = [footprint, footprint]
        rs = [0, 0]


        #tex[wo[0]:wo[0]+re[0]-rs[0], wo[1]:wo[1]+re[1]-rs[1]] = tex[wo[0]:wo[0]+re[0]-rs[0], wo[1]:wo[1]+re[1]-rs[1]].clone() + dist[rs[0]:re[0], rs[1]:re[1]]

        # This is probably the worst code I've ever written.
        # It is an out of bounds check for rectangles such that we can copy the correct parts from our tensors.
        # There's certainly better and more precise ways to solve this, but for know it works.
        
        rect_start = torch.tensor([0, 0], dtype=torch.int32, device=device)
        rect_end = torch.tensor([footprint, footprint], dtype=torch.int32, device=device)

        if footprint_origin_in_original_image[0] < 0:
            rect_start[0] = footprint_origin_in_original_image.abs()[0]
            footprint_origin_in_original_image[0] = 0

        if footprint_origin_in_original_image[1] < 0:
            rect_start[1] = footprint_origin_in_original_image.abs()[1]
            footprint_origin_in_original_image[1] = 0

        if footprint_origin_in_original_image[0] + footprint >= texture_size[0]:
            rect_end[0] = texture_size[0] - footprint_origin_in_original_image[0]
        
        if footprint_origin_in_original_image[1] + footprint >= texture_size[1]:
            rect_end[1] = texture_size[1] - footprint_origin_in_original_image[1]

        wo = footprint_origin_in_original_image.int()
        rs = rect_start.int()
        re = rect_end.int()

        tex[wo[0]:wo[0]+re[0]-rs[0], wo[1]:wo[1]+re[1]-rs[1]] = tex[wo[0]:wo[0]+re[0]-rs[0], wo[1]:wo[1]+re[1]-rs[1]].clone() + dist[rs[0]:re[0], rs[1]:re[1]]
        

    return tex.T

def baked_sum_2(points: torch.tensor, sigma: torch.tensor, texture_size: torch.tensor, num_std: int = 4, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    # Somewhere between [0, texture_size] but in float
    points = points.clone() * texture_size

    # We use 3*sigma^2 here, to include most of the gaussian
    footprint = math.floor(sigma.sqrt().item()) * num_std
    footprint = footprint + 1 if footprint % 2 == 0 else footprint
    half_footprint = int((footprint - 1) / 2)

    point_in_middle_of_footprint = points - points.floor() + half_footprint

    footprint_origin_in_original_image = (points - half_footprint).floor() # [Y, X]

    y, x = torch.meshgrid(torch.arange(0, footprint, device=device), torch.arange(0, footprint, device=device), indexing='ij')
    
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    
    
    y_dist = y - point_in_middle_of_footprint[:, 0:1].unsqueeze(-1)
    x_dist = x - point_in_middle_of_footprint[:, 1:2].unsqueeze(-1)

    dist = y_dist*y_dist + x_dist*x_dist
    dist = torch.exp(-torch.pow(dist / sigma, 2))
        
    wo = footprint_origin_in_original_image.int()
    rect_start = torch.zeros(points.shape[0], 2, dtype=torch.int32, device=device)
    rect_end = torch.zeros(points.shape[0], 2, dtype=torch.int32, device=device) + footprint

    if (wo[:, 0] < 0).any():
        rect_start[:, 0] = torch.where(wo[:, 0] < 0, wo.abs()[:, 0], rect_start[:, 0])
        wo[:, 0] = torch.where(wo[:, 0] < 0, 0, wo[:, 0])

    if (wo[:, 1] < 0).any():
        rect_start[:, 1] = torch.where(wo[:, 1] < 1, wo.abs()[:, 1], rect_start[:, 1])
        wo[:, 1] = torch.where(wo[:, 1] < 0, 0, wo[:, 1])

    if (wo[:, 0] + footprint >= texture_size[0]).any():
        rect_end[:, 0] = torch.where(wo[:, 0] + footprint >= texture_size[0], texture_size[0] - wo[:, 0], rect_end[:, 0])
    
    if (wo[:, 1] + footprint >= texture_size[1]).any():
        rect_end[:, 1] = torch.where(wo[:, 1] + footprint >= texture_size[1], texture_size[1] - wo[:, 1], rect_end[:, 1])

    re = rect_end.clone()
    rs = rect_start.clone()

    for i in range(points.shape[0]):
        tex[wo[i, 0]:wo[i, 0]+re[i, 0]-rs[i, 0], wo[i, 1]:wo[i, 1]+re[i, 1]-rs[i, 1]] = tex[wo[i, 0]:wo[i, 0]+re[i, 0]-rs[i, 0], wo[i, 1]:wo[i, 1]+re[i, 1]-rs[i, 1]].clone() + dist[i, rs[i, 0]:re[i, 0], rs[i, 1]:re[i, 1]]

    return tex

def baked_softor(points: torch.tensor, sigma: torch.tensor, texture_size: torch.tensor, num_std: int = 5, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.ones(texture_size.tolist(), dtype=torch.float32, device=device)
    # Somewhere between [0, texture_size] but in float
    points = points.clone() * texture_size

    for i in range(points.shape[0]):
        point = points[i]

        # We use 3*sigma^2 here, to include most of the gaussian
        footprint = math.floor(sigma.sqrt().item()) * num_std
        footprint = footprint + 1 if footprint % 2 == 0 else footprint
        half_footprint = int((footprint - 1) / 2)

        point_in_middle_of_footprint = point - point.floor() + half_footprint

        footprint_origin_in_original_image = (point - half_footprint).floor() # [Y, X]

        y, x = torch.meshgrid(torch.arange(0, footprint, device=device), torch.arange(0, footprint, device=device), indexing='ij')

        y_dist = y - point_in_middle_of_footprint[0:1]
        x_dist = x - point_in_middle_of_footprint[1:2]
        dist = y_dist*y_dist + x_dist*x_dist
        dist = torch.exp(-torch.pow(dist / sigma, 2))
        
        wo = (point.floor() - half_footprint).int()
        re = [footprint, footprint]
        rs = [0, 0]


        #tex[wo[0]:wo[0]+re[0]-rs[0], wo[1]:wo[1]+re[1]-rs[1]] = tex[wo[0]:wo[0]+re[0]-rs[0], wo[1]:wo[1]+re[1]-rs[1]].clone() + dist[rs[0]:re[0], rs[1]:re[1]]

        # This is probably the worst code I've ever written.
        # It is an out of bounds check for rectangles such that we can copy the correct parts from our tensors.
        # There's certainly better and more precise ways to solve this, but for know it works.
        
        rect_start = torch.tensor([0, 0], dtype=torch.int32, device=device)
        rect_end = torch.tensor([footprint, footprint], dtype=torch.int32, device=device)

        if footprint_origin_in_original_image[0] < 0:
            rect_start[0] = footprint_origin_in_original_image.abs()[0]
            footprint_origin_in_original_image[0] = 0

        if footprint_origin_in_original_image[1] < 0:
            rect_start[1] = footprint_origin_in_original_image.abs()[1]
            footprint_origin_in_original_image[1] = 0

        if footprint_origin_in_original_image[0] + footprint >= texture_size[0]:
            rect_end[0] = texture_size[0] - footprint_origin_in_original_image[0]
        
        if footprint_origin_in_original_image[1] + footprint >= texture_size[1]:
            rect_end[1] = texture_size[1] - footprint_origin_in_original_image[1]

        wo = footprint_origin_in_original_image.int()
        rs = rect_start.int()
        re = rect_end.int()

        tex[wo[0]:wo[0]+re[0]-rs[0], wo[1]:wo[1]+re[1]-rs[1]] = tex[wo[0]:wo[0]+re[0]-rs[0], wo[1]:wo[1]+re[1]-rs[1]].clone() * (1-dist[rs[0]:re[0], rs[1]:re[1]])

    return (1 - tex).T


def baked_softor_2(points: torch.tensor, sigma: torch.tensor, texture_size: torch.tensor, num_std: int = 5, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.ones(texture_size.tolist(), dtype=torch.float32, device=device)
    # Somewhere between [0, texture_size] but in float
    points = points.clone() * texture_size

    # We use 3*sigma^2 here, to include most of the gaussian
    footprint = math.floor(sigma.sqrt().item()) * num_std
    footprint = footprint + 1 if footprint % 2 == 0 else footprint
    half_footprint = int((footprint - 1) / 2)

    point_in_middle_of_footprint = points - points.floor() + half_footprint

    footprint_origin_in_original_image = (points - half_footprint).floor() # [Y, X]

    y, x = torch.meshgrid(torch.arange(0, footprint, device=device), torch.arange(0, footprint, device=device), indexing='ij')
    
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    
    
    y_dist = y - point_in_middle_of_footprint[:, 0:1].unsqueeze(-1)
    x_dist = x - point_in_middle_of_footprint[:, 1:2].unsqueeze(-1)

    dist = y_dist*y_dist + x_dist*x_dist
    dist = torch.exp(-torch.pow(dist / sigma, 2))
        
    wo = footprint_origin_in_original_image.int()
    rect_start = torch.zeros(points.shape[0], 2, dtype=torch.int32, device=device)
    rect_end = torch.zeros(points.shape[0], 2, dtype=torch.int32, device=device) + footprint

    if (wo[:, 0] < 0).any():
        rect_start[:, 0] = torch.where(wo[:, 0] < 0, wo.abs()[:, 0], rect_start[:, 0])
        wo[:, 0] = torch.where(wo[:, 0] < 0, 0, wo[:, 0])

    if (wo[:, 1] < 0).any():
        rect_start[:, 1] = torch.where(wo[:, 1] < 1, wo.abs()[:, 1], rect_start[:, 1])
        wo[:, 1] = torch.where(wo[:, 1] < 0, 0, wo[:, 1])

    if (wo[:, 0] + footprint >= texture_size[0]).any():
        rect_end[:, 0] = torch.where(wo[:, 0] + footprint >= texture_size[0], texture_size[0] - wo[:, 0], rect_end[:, 0])
    
    if (wo[:, 1] + footprint >= texture_size[1]).any():
        rect_end[:, 1] = torch.where(wo[:, 1] + footprint >= texture_size[1], texture_size[1] - wo[:, 1], rect_end[:, 1])

    re = rect_end.clone()
    rs = rect_start.clone()

    for i in range(points.shape[0]):
        tex[wo[i, 0]:wo[i, 0]+re[i, 0]-rs[i, 0], wo[i, 1]:wo[i, 1]+re[i, 1]-rs[i, 1]] = tex[wo[i, 0]:wo[i, 0]+re[i, 0]-rs[i, 0], wo[i, 1]:wo[i, 1]+re[i, 1]-rs[i, 1]].clone() * (1-dist[i, rs[i, 0]:re[i, 0], rs[i, 1]:re[i, 1]])

    return (1 - tex).T


def rasterize_points_baked_softor(points: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.ones(texture_size.tolist(), dtype=torch.float32, device=device)
    
    # Somewhere between [0, texture_size] but in float
    points = points.clone() * texture_size

    for i in range(points.shape[0]):
        # Generate x, y indices
        x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')

        y_dist = y - points[i, 0:1].unsqueeze(-1)
        x_dist = x - points[i, 1:2].unsqueeze(-1)
        
        point_distances = (y_dist*y_dist + x_dist*x_dist)# / (texture_size * texture_size).sum().sqrt()
        point_distances = torch.exp(-torch.pow(point_distances / sigma, 2))
        tex = tex.clone() * (1-point_distances)


    return 1-tex

# We assume points to be in camera space [0, 1]
def rasterize_points_baked_sum(points: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    
    # Somewhere between [0, texture_size] but in float
    points = points.clone() * texture_size

    for i in range(points.shape[0]):
        # Generate x, y indices
        x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')

        y_dist = y - points[i, 0:1].unsqueeze(-1)
        x_dist = x - points[i, 1:2].unsqueeze(-1)
        
        point_distances = (y_dist*y_dist + x_dist*x_dist)# / (texture_size * texture_size).sum().sqrt()
        point_distances = torch.exp(-torch.pow(point_distances / sigma, 2))
        tex += point_distances


    return tex


def get_mpl_colormap(cmap):
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)



def test_point_reg(reduce_overlap: bool = True):
    import cv2
    import numpy as np
    from tqdm import tqdm
    import imageio
    import matplotlib.colors
    import timeit
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    points = torch.rand([500, 2], device=device)
    points.requires_grad = True
    sigma = torch.tensor([15.0], device=device)**2
    texture_size = torch.tensor([512, 512], device=device)
    loss_func = torch.nn.L1Loss()

    opt_steps = 200

    optim = torch.optim.Adam([
        {'params': points,  'lr': 0.001}
        ])
    
    images = []
    for i in tqdm(range(opt_steps)):
        optim.zero_grad()

        summed = baked_sum_2(points, sigma, texture_size)
        softored = baked_softor_2(points, sigma, texture_size)

        #rasterized_points = rasterize_points(points, sigma, texture_size)
        #softored = softor(rasterized_points)
        #summed = rasterized_points.sum(dim=0)

        loss = loss_func(softored, summed) if reduce_overlap else -loss_func(softored, summed)
        print(loss.item())
        loss.backward()
        optim.step()

        with torch.no_grad():
            points[points >= 1.0] = 0.999
            points[points <= 0.0] = 0.001

            # Apply custom colormap
            colors = [(0.0, 0.1921, 0.4156), (0, 0.69, 0.314)]  # R -> G -> B
            
            #fig = plt.figure(frameon=False)
            #fig.set_size_inches(10, 10)
            #ax = plt.Axes(fig, [0., 0., 1., 1.])
            #ax.set_axis_off()
            #ax.set_aspect(aspect='equal')
            #ax.set_facecolor(colors[0])
            #fig.add_axes(ax)

            #ax.scatter(points.detach().cpu().numpy()[:, 0], points.detach().cpu().numpy()[:, 1], s=60.0*10, color=colors[0])
            #fig.canvas.draw()
            #img_plot = np.array(fig.canvas.renderer.buffer_rgba())
            #np_points = img_plot
            #images.append(np_points)
            bla = cv2.applyColorMap((softored.detach().cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            cv2.imshow("Show", bla)
            cv2.waitKey(1)
            #if i == 0 or i == opt_steps - 1:
            #    fig.savefig("assets/point_reduced_overlap{0}.eps".format(i) if reduce_overlap else "assets/point_increased_overlap{0}.eps".format(i), 
            #        facecolor=ax.get_facecolor(), 
            #        edgecolor='none',
            #        bbox_inches = 'tight',
            #        pad_inches=0)
            plt.close()
            
            #cv2.imshow("Optim Lines", np_points)
            #cv2.waitKey(1)
            #lines.requires_grad = True
    #imageio.v3.imwrite("assets/point_regularization.mp4", np.stack(images, axis=0), fps=25)



def test_line_reg():
    import cv2
    import numpy as np
    from tqdm import tqdm
    import imageio
    import matplotlib.colors

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # We define line as: 
        # P0: x + -t*d
        # P1: x +  t*d
        # Where d is the direction, x the location vector and t half its length.
    # We want to optimize the location vector x of all lines, such that they do not overlap.

    num_lines = 50
    sigma = 10.0
    opt_steps = 250

    t = torch.tensor([0.5], device=device)
    direction = torch.rand([2], device=device).unsqueeze(0)
    direction = direction / direction.norm()
    
    location_vector = (torch.rand([num_lines, 2], device=device) - 0.5) * 2.0 / 10.0 # Every line should be roughly in the middle of our frame
    location_vector.requires_grad = True

    sigma = torch.tensor([sigma], device=device)
    texture_size = torch.tensor([512, 512], device=device)
    loss_func = torch.nn.L1Loss()


    optim = torch.optim.Adam([
        {'params': location_vector,  'lr': 0.005}
        ])
    
    images = []
    for i in tqdm(range(opt_steps)):
        optim.zero_grad()

        p0 = location_vector + t*direction
        p1 = location_vector - t*direction
        lines = torch.concat([p0.unsqueeze(-1), p1.unsqueeze(-1)], dim=-1).transpose(1, 2)

        rasterized_lines = rasterize_lines(lines, sigma, texture_size)

        softored = softor(rasterized_lines)
        summed = rasterized_lines.sum(dim=0)

        loss = loss_func(softored, summed)
        loss.backward()
        optim.step()

        with torch.no_grad():
            location_vector[p0 > 1.0] -= 0.01
            location_vector[p0 < -1.0] += 0.01
            location_vector[p1 > 1.0] -= 0.01
            location_vector[p1 < -1.0] += 0.01

            colors = [(0.0, 0.1921, 0.4156), (0, 0.69, 0.314)]  # R -> G -> B
            fig = plt.figure(frameon=False)
            fig.set_size_inches(10, 10)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_axis_off()
            ax.set_aspect(aspect='equal')
            fig.add_axes(ax)

            lines_copy = lines.transpose(1, 2).detach().cpu().numpy()
            for j in range(lines_copy.shape[0]):
                ax.plot(lines_copy[j, 0, :], lines_copy[j, 1, :], c=colors[0], linewidth=9.5, solid_capstyle='round')# c=colors[0], linewidth=60)

            fig.canvas.draw()
            img_plot = np.array(fig.canvas.renderer.buffer_rgba())
            np_points = img_plot
            images.append(np_points)

            if i == 0 or i == opt_steps - 1:
                fig.savefig("assets/line_reduced_overlap{0}.eps".format(i), 
                    facecolor=ax.get_facecolor(), 
                    edgecolor='none',
                    bbox_inches = 'tight',
                    pad_inches=0)
            plt.close()

    imageio.v3.imwrite("assets/line_regularization.mp4", np.stack(images, axis=0), fps=25)
    #optimize("line_regularization.gif")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    points_a = torch.rand(10, 2, device=device) # (Y, X)
    
    #points_a = torch.tensor([[0.565, 0.5555]], device=device)
    #points_b = torch.tensor([[0.51,0.51]], device=device)
    
    
    texture_size = torch.tensor([100, 100], device=device) # (Y, X)
    
    sigma = torch.tensor([10.0], device=device)**2
    sum = baked_sum(points_a, sigma, texture_size, device=device)
    #sum_og = baked_sum(points_b, sigma, texture_size, device=device)
    sum_og = rasterize_points(points_a, sigma, texture_size, device=device).sum(dim=0)
    #sum_og = rasterize_points(points_b, sigma, texture_size, device=device).sum(dim=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(sum_og.detach().cpu().numpy())
    ax2.imshow(sum.detach().cpu().numpy())
    ax3.imshow((sum_og - sum).detach().cpu().numpy())
    ax1.set_title('OG')
    ax2.set_title('NEW')
    ax3.set_title('DIFF')
    fig.show()
    plt.show()

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


def time_it():
    import timeit, functools

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    points = torch.rand([500, 2], device=device)
    points.requires_grad = True
    sigma = torch.tensor([10.0], device=device)**2
    texture_size = torch.tensor([512, 512], device=device)
    repeats = 50

    og_sum = timeit.Timer(lambda: rasterize_points(points, sigma, texture_size, device).sum(dim=0))
    print(og_sum.timeit(repeats))

    t_baked_sum = timeit.Timer(lambda: baked_sum(points, sigma, texture_size, 4, device)) 
    print(t_baked_sum.timeit(repeats))

    t_baked_sum2 = timeit.Timer(lambda: baked_sum_2(points, sigma, texture_size, 4, device)) 
    print(t_baked_sum2.timeit(repeats))

    og_softor = timeit.Timer(lambda: softor(rasterize_points(points, sigma, texture_size, device))) 
    print(og_softor.timeit(repeats))

    t_baked_softor1 = timeit.Timer(lambda: baked_softor(points, sigma, texture_size, 4, device)) 
    print(t_baked_softor1.timeit(repeats))

    t_baked_softor2 = timeit.Timer(lambda: baked_softor_2(points, sigma, texture_size, 4, device)) 
    print(t_baked_softor2.timeit(repeats))



if __name__ == "__main__":
    #time_it()
    test_point_reg(reduce_overlap=True)
    #test_point_reg(reduce_overlap=False)
    #test_line_reg()
    #main()