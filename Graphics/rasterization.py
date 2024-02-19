import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

# We assume points to be in camera space [0, 1]
def rasterize_points(points: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((points.shape[0], 1, 1, 1))

    
    # Somewhere between [0, texture_size] but in float
    points *= texture_size

    # Generate x, y indices
    x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = (y_dist*y_dist + x_dist*x_dist)# / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances, 2) / (sigma * sigma))


    return point_distances



def rasterize_points_in_non_ndc(points: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = (y_dist*y_dist + x_dist*x_dist)# / (texture_size * texture_size).sum().sqrt()
    point_distances = torch.exp(-torch.pow(point_distances, 2) / (sigma * sigma))


    return point_distances


# We assume points to be in NDC [-1, 1]
def rasterize_depth(points: torch.tensor, depth_vals: torch.tensor, sigma: float, texture_size: torch.tensor, device: torch.cuda.device = torch.device("cuda")) -> torch.tensor:
    tex = torch.zeros(texture_size.tolist(), dtype=torch.float32, device=device)
    tex = tex[None, ...]
    tex = tex.repeat((points.shape[0], 1, 1, 1))
    
    
    # Somewhere between [0, texture_size] but in float
    points *= texture_size

    # Generate x, y indices
    x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=device), torch.arange(0, texture_size[0], device=device), indexing='ij')
    y = y.unsqueeze(0).repeat((points.shape[0], 1, 1))
    x = x.unsqueeze(0).repeat((points.shape[0], 1, 1))    


    y_dist = y - points[:, 0:1].unsqueeze(-1)
    x_dist = x - points[:, 1:2].unsqueeze(-1)
    
    point_distances = y_dist*y_dist + x_dist*x_dist #/ (texture_size * texture_size).sum().sqrt()
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    points = (torch.rand([500, 2], device=device) - 0.5) * 2.0
    points.requires_grad = True
    sigma = torch.tensor([125.0], device=device)
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

        loss = loss_func(softored, summed) if reduce_overlap else -loss_func(softored, summed)
        loss.backward()
        optim.step()

        with torch.no_grad():
            points[points >= 1.0] = 0.999
            points[points <= -1.0] = -0.999

            # Apply custom colormap
            colors = [(0.0, 0.1921, 0.4156), (0, 0.69, 0.314)]  # R -> G -> B
            
            fig = plt.figure(frameon=False)
            fig.set_size_inches(10, 10)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            ax.set_aspect(aspect='equal')
            ax.set_facecolor(colors[0])
            fig.add_axes(ax)

            ax.scatter(points.detach().cpu().numpy()[:, 0], points.detach().cpu().numpy()[:, 1], s=60.0*10, color=colors[0])
            fig.canvas.draw()
            img_plot = np.array(fig.canvas.renderer.buffer_rgba())
            np_points = img_plot
            images.append(np_points)

            if i == 0 or i == opt_steps - 1:
                fig.savefig("assets/point_reduced_overlap{0}.eps".format(i) if reduce_overlap else "assets/point_increased_overlap{0}.eps".format(i), 
                    facecolor=ax.get_facecolor(), 
                    edgecolor='none',
                    bbox_inches = 'tight',
                    pad_inches=0)
            plt.close()
            
            #cv2.imshow("Optim Lines", np_points)
            #cv2.waitKey(1)
            #lines.requires_grad = True
    imageio.v3.imwrite("assets/point_regularization.mp4", np.stack(images, axis=0), fps=25)



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
    test_point_reg(reduce_overlap=True)
    #test_point_reg(reduce_overlap=False)
    test_line_reg()
    #main()