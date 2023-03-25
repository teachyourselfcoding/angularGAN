import cv2
import numpy as np
import torch
import torch.nn.functional as F

def linearize_img(img):
    # Convert the image from sRGB to linear RGB
    linear_img = np.power(img / 255.0, 2.2)
    return linear_img

def unlinearize_img(img):
    # Convert the balanced image back to sRGB
    output_img = np.power(img, 1/2.2) * 255.0
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)   
    return output_img

def gw_balance(img):
     gw_balanced_img = cv2.xphoto.balanceWhite(img, None, cv2.xphoto.WHITE_BALANCE_SIMPLE)
     return gw_balanced_img

def local_illuminant_loss(image_tensor, kernel_size=5, variance_penalty_weight=1.0):
    """
    Calculates the loss based on the difference between the local average illuminant
    and the global average illuminant to encourage a uniform illuminant in the image,
    and penalizes the model if the illuminant has a significant variation in color.

    Args:
        image_tensor (torch.Tensor): A 3D tensor representing the input image in the shape
                                     (C, H, W) where C is the number of color channels (3 for RGB),
                                     H is the height, and W is the width.
        kernel_size (int): The size of the kernel used for local average illuminant calculation.
        variance_penalty_weight (float): The weight for the penalty term based on the variance
                                         in the illuminant color.

    Returns:
        torch.Tensor: The calculated loss value based on the difference between local and
                      global average illuminant and the penalty for color variance.
    """
    assert image_tensor.dim() == 3, "Input tensor must be 3D."
    assert image_tensor.size(0) == 3, "Input tensor must have 3 color channels (RGB)."

    # Calculate the global average illuminant
    global_avg_illuminant = torch.mean(image_tensor, dim=(1, 2)).view(3, 1, 1)

    # Calculate the local average illuminant using a convolution operation
    kernel = torch.ones(3, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    kernel = kernel.to(image_tensor.device)
    local_avg_illuminant = F.conv2d(image_tensor.unsqueeze(0), kernel, padding=kernel_size//2)
    local_avg_illuminant = local_avg_illuminant.squeeze(0)

    # Calculate the loss based on the difference between the local and global average illuminant
    illuminant_diff = local_avg_illuminant - global_avg_illuminant
    loss = torch.mean(illuminant_diff ** 2)

    # Calculate the variance of the local average illuminant color and add the penalty term
    mean_local_color = torch.mean(local_avg_illuminant, dim=(1, 2))
    color_variance = torch.mean((local_avg_illuminant - mean_local_color.view(3, 1, 1)) ** 2, dim=(1, 2))
    variance_penalty = torch.mean(color_variance)
    loss += variance_penalty_weight * variance_penalty

    return loss