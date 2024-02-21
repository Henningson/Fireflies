import torch
import numpy as np
import cv2
import imageio

import Utils.bridson as bridson
import Graphics.rasterization as rasterization

from tqdm import tqdm


def optimize_points(points, image, sigma, Niter):
    points = points.clone()
    image = image.clone().float() / 255
    image_size = torch.tensor(image.shape[0:2])

    points.requires_grad = True
    optim = torch.optim.Adam([{"params": points, "lr": 0.1}])
    loss_func = torch.nn.MSELoss()

    writer = imageio.get_writer("sample_optimization.mp4", fps=25)

    for _ in (progress_bar := tqdm(range(Niter))):
        optim.zero_grad()
        rastered_points = rasterization.rasterize_points_in_non_ndc(
            points, sigma, image_size, device="cpu"
        )
        sampled_image = rasterization.softor(rastered_points).unsqueeze(-1) * image
        loss = loss_func(sampled_image, image) + loss_func(
            rasterization.softor(rastered_points), rastered_points.sum(dim=0)
        )
        loss.backward()
        optim.step()

        sampled_image = (sampled_image.detach().cpu() * 255).numpy().astype(np.uint8)
        writer.append_data(sampled_image[:, :, [2, 1, 0]])
        cv2.imshow("Sample", sampled_image)
        cv2.waitKey(1)

        progress_bar.set_description("Loss: {0:.4f}".format(loss.item()))

    writer.close()
    return points


def generate_mask_from_points(points, image):
    floored_points = torch.floor(points).int()
    mask = torch.zeros_like(image)
    mask[floored_points[:, 0], floored_points[:, 1], :] = 1
    return image.clone() * mask


def main():
    image = cv2.imread("./assets/Lenna.png")
    image = cv2.resize(image, [256, 256])
    image = torch.from_numpy(image)
    height, width, _ = image.shape
    radius = 5
    Niter = 50

    blue_noise_points = torch.tensor(
        bridson.poisson_disc_samples(width, height, radius, 5)
    )
    random_points = torch.rand_like(blue_noise_points) * torch.tensor([height, width])
    optim_points = optimize_points(blue_noise_points, image, sigma=3, Niter=Niter)

    blue_noise_image = generate_mask_from_points(blue_noise_points, image)
    random_image = generate_mask_from_points(random_points, image)
    optim_image = generate_mask_from_points(optim_points, image)

    concat = torch.concat([image, optim_image, blue_noise_image, random_image], dim=1)
    cv2.imwrite("OptimizedPointSamples.png", concat.detach().cpu().numpy())

    # optimized_points = optimize_points(blue_noise_points, image)


if __name__ == "__main__":
    main()
