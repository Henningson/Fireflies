import torch


class PointRasterizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, sigma, texture_size):
        ctx.points = points
        ctx.sigma = sigma
        ctx.texture_size = texture_size
            
        raster_image = torch.zeros(texture_size.tolist(), device=points.device)
        for i in range(points.shape[0]):
            x, y = torch.meshgrid(torch.arange(0, texture_size[1], device=points.device), torch.arange(0, texture_size[0], device=points.device), indexing='ij')
            y_dist = y - points[i, 0:1]
            x_dist = x - points[i, 1:2]
            point_distances = (y_dist*y_dist + x_dist*x_dist).sqrt()
            point_distances = torch.exp(-torch.pow(point_distances, 2) / (sigma * sigma))
            raster_image = raster_image + point_distances

        return raster_image
    
    @staticmethod
    def backward(ctx, grad_output):
        