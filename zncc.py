"""
Normalized Cross-Correlation for pattern matching.
pytorch implementation
roger.bermudez@epfl.ch
CVLab EPFL 2019
"""

import logging
import torch
from torch.nn import functional as F


ncc_logger = logging.getLogger(__name__)


def patch_mean(images, patch_shape):
    """
    Computes the local mean of an image or set of images.
    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)
    Returns:
        Tensor same size as the image, with local means computed independently for each channel.
    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> means = patch_mean(images, patch_shape)
        >>> expected_mean = images[3, 2, :5, :5].mean()  # mean of the third image, channel 2, top left 5x5 patch
        >>> computed_mean = means[3, 2, 5//2, 5//2]      # computed mean whose 5x5 neighborhood covers same patch
        >>> computed_mean.isclose(expected_mean).item()
        1
    """
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).byte()
    weights[1 - channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)

    return result


def patch_std(image, patch_shape):
    """
    Computes the local standard deviations of an image or set of images.
    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)
    Returns:
        Tensor same size as the image, with local standard deviations computed independently for each channel.
    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> stds = patch_std(images, patch_shape)
        >>> patch = images[3, 2, :5, :5]
        >>> expected_std = patch.std(unbiased=False)     # standard deviation of the third image, channel 2, top left 5x5 patch
        >>> computed_std = stds[3, 2, 5//2, 5//2]        # computed standard deviation whose 5x5 neighborhood covers same patch
        >>> computed_std.isclose(expected_std).item()
        1
    """
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    """
    Z-normalize image channels independently.
    """
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)


class NCC(torch.nn.Module):
    """
    Computes the [Zero-Normalized Cross-Correlation][1] between an image and a template.
    Example:
        >>> lena_path = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        >>> lena_tensor = torch.Tensor(plt.imread(lena_path)).permute(2, 0, 1).cuda()
        >>> patch_center = 275, 275
        >>> y1, y2 = patch_center[0] - 25, patch_center[0] + 25
        >>> x1, x2 = patch_center[1] - 25, patch_center[1] + 25
        >>> lena_patch = lena_tensor[:, y1:y2 + 1, x1:x2 + 1]
        >>> ncc = NCC(lena_patch)
        >>> ncc_response = ncc(lena_tensor[None, ...])
        >>> ncc_response.max()
        tensor(1.0000, device='cuda:0')
        >>> np.unravel_index(ncc_response.argmax(), lena_tensor.shape)
        (0, 275, 275)
    [1]: https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    """
    def __init__(self, template, keep_channels=False):
        super().__init__()

        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        dimensions = len(template_shape)
        self.padding = tuple(side // 2 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1, )
        self.normalized_template = self.normalized_template.repeat(channels, *ones)
        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).byte()
        self.normalized_template[1 - channel_selector] = 0
        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image):
        result = self.conv_f(image, self.normalized_template, padding=self.padding, bias=None)
        std = patch_std(image, self.normalized_template.shape[1:])
        result.div_(std)
        if not self.keep_channels:
            result = result.mean(dim=1)

        return result


import matplotlib.pyplot as plt

def softargmax1d(input, beta=100):
    *_, n = input.shape
    input = torch.nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))



import cv2
import torchvision
import numpy as np

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# We do not pad, nor reshape here.
# This needs to be done explicitly after unfolding the image into singular components.
def unfold_padded_image(tensor: torch.tensor, window_size_y: int, window_size_x: int) -> torch.tensor:
    assert window_size_y % 2 == 1, "Window size for y is even."
    assert window_size_x % 2 == 1, "Window size for x is even."
    return tensor.unfold(-2, window_size_y, 1).unfold(-2, window_size_x, 1)

# Pad an image uneven formly, given a specific window size, such that for zncc every pixel is considered.
def pad_image_uniformly(tensor: torch.tensor, kernel_size: int):
    pad = kernel_size // 2
    return F.pad(tensor, (pad, pad, pad, pad), mode="constant", value=0)

# Pad an image or projection pattern, such that epipolar constraints are considered.
def pad_for_epipolar(tensor: torch.tensor, kernel_size: int, epipolar_search_size: int):
    pad_ud = kernel_size // 2
    pad_lr = epipolar_search_size // 2
    return F.pad(tensor, (pad_lr, pad_lr, pad_ud, pad_ud), mode="constant", value=0.0)

# We assume that patches and dim are of shape (XN, H, W) and 
# the cross correlation for each XN should be calculated.
def find_likely_match_MAD(patches: torch.tensor, kernel: torch.tensor, metric="CONV2D") -> torch.tensor:
    cc_tensor_list = []
    for i in range(patches.shape[-3]):
        per_patch_values = []
        for x in range(0, patches.shape[-1] - kernel.shape[-1] + 1):
            temp = (patches[i, :, x:x+kernel.shape[-1]] - kernel[i, :]).abs().mean()
            per_patch_values.append(temp)
        cc_tensor_list.append(torch.tensor(per_patch_values).unsqueeze(0))

    cc = torch.concatenate(cc_tensor_list, dim=-2)
    return 1 - (cc / cc.max(dim=1, keepdim=True)[0])


# We assume patches to be of size H X Wp and kernel to be of size W x W.
# We can do this, since we expect images to be rectified.
def per_match_MAD(patches: torch.tensor, kernel: torch.tensor) -> torch.tensor:
    window_sized_patches = patches.unfold(-2, kernel.shape[-1], 1).unfold(-2, kernel.shape[-1], 1)
    per_patch_mse = (window_sized_patches - kernel).abs().mean(dim=(-2, -1))
    return 1 - (per_patch_mse / per_patch_mse.max(dim=-1, keepdim=True)[0])

# We assume patches to be of size H X Wp and kernel to be of size W x W.
# We can do this, since we expect images to be rectified.
def per_match_MSE(patches: torch.tensor, kernel: torch.tensor) -> torch.tensor:
    window_sized_patches = patches.unfold(-2, kernel.shape[-1], 1).unfold(-2, kernel.shape[-1], 1)
    per_patch_mse = (window_sized_patches - kernel).pow(2).mean(dim=(-2, -1))
    return 1 - (per_patch_mse / per_patch_mse.max(dim=-1, keepdim=True)[0])

# We assume that patches and dim are of shape (XN, H, W) and 
# the cross correlation for each XN should be calculated.
def find_likely_match_CONV2D(patches: torch.tensor, kernel: torch.tensor, metric="CONV2D") -> torch.tensor:
    cc_tensor_list = []
    for i in range(patches.shape[-3]):
        cc_tensor_list.append(F.conv2d(patches[i:i+1].unsqueeze(0), kernel[i:i+1].unsqueeze(0)))
    cc = torch.concatenate(cc_tensor_list, dim=-2)

# We assume that patches and dim are of shape (XN, H, W) and 
# the cross correlation for each XN should be calculated.
def find_likely_match_MSE(patches: torch.tensor, kernel: torch.tensor, metric="CONV2D") -> torch.tensor:
    cc_tensor_list = []
    for i in range(patches.shape[-3]):
        per_patch_values = []
        for x in range(0, patches.shape[-1] - kernel.shape[-1] + 1):
            temp = (patches[i, :, x:x+kernel.shape[-1]] - kernel[i, :]).pow(2).mean()
            per_patch_values.append(temp)
        cc_tensor_list.append(torch.tensor(per_patch_values).unsqueeze(0))

    cc = torch.concatenate(cc_tensor_list, dim=-2).squeeze()
    return 1 - (cc / cc.max(dim=1, keepdim=True)[0])


# imgA is moved over imgB
# Differential disparity map estimation
def compute_disparity_map(imgA: torch.tensor, imgB: torch.tensor, kernel_size: int, epipolar_size: int, metric: str="MSE"):
    # Pad image uniformly such that it gets padded on each size by kernel_size // 2
    a_padded = pad_image_uniformly(imgA, kernel_size)

    # Unfold image into singular patches, such that we generate a tensor of size (H*W x WINDOW_SIZE x WINDOW_SIZE)
    a_unfolded = unfold_padded_image(a_padded, kernel_size, kernel_size)
    a = a_unfolded.squeeze().reshape(-1, kernel_size, kernel_size)
    
    # Pad image such that top and bottom are padded by kernel_size // 2 and left and right by epipolar_size //2
    b_padded = pad_for_epipolar(imgB, kernel_size, epipolar_size)

    # Unfold image into singular patches, such that we generate a tensor of size (H*W x WINDOW_SIZE x EPIPOLAR_SIZE)
    b_unfolded = unfold_padded_image(b_padded, kernel_size, epipolar_size)
    b = b_unfolded.squeeze().reshape(-1, kernel_size, epipolar_size)

    # Compute cross correlation for each kernel and patch
    # By transforming smaller function into batch-wise function using vmap
    f = torch.vmap(per_match_MAD)
    cross_corr_a = f(b, a).squeeze()

    # Compute differentiable argmax, such that we can use auto-differentiation
    argmax = softargmax1d(cross_corr_a)

    # Finally shift argmax by added padding
    return argmax.reshape(imgA.shape) - (epipolar_size - kernel_size) / 2

def compute_depth_map(disparity: torch.tensor, focal_length: float, baseline: float) -> torch.tensor:
    return baseline * focal_length / disparity

    

if __name__ == "__main__":

    kernel_size = 15
    epipolar_size = 25
    
    imgA = torch.tensor(cv2.imread("imL.png", 0)).float() / 255.0
    imgB = torch.tensor(cv2.imread("imR.png", 0)).float() / 255.0
    #y0, y1 = img.shape[0] // 2 - crop_size, img.shape[0] // 2 + crop_size
    #x0, x1 = img.shape[1] // 2 - crop_size, img.shape[1] // 2 + crop_size
    #img = img[y0:y1, x0:x1]

    disparity_map = compute_disparity_map(imgA, imgB, kernel_size, epipolar_size, metric="MSE")
    print("GT")
    plt.axis("off")
    plt.title("GT")
    plt.imshow(disparity_map.detach().cpu().numpy())
    plt.show()

    exit()

    test_tensor = torch.randn(1, 5, 50).cuda()
    patch_center = 2, 25
    y1, y2 = patch_center[0] - 2, patch_center[0] + 2
    x1, x2 = patch_center[1] - 2, patch_center[1] + 2
    test_patch = test_tensor[:, y1:y2 + 1, x1:x2 + 1]
    ncc = NCC(test_patch)
    ncc_response = ncc(test_tensor[None, ...])
    ncc_minimized = ncc_response[:, ncc_response.shape[1] // 2, :]
    print(ncc_response.max())
    print(unravel_index(ncc_response.argmax(), test_tensor.shape))
    print(softargmax1d(ncc_minimized))