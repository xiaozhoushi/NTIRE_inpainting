import cv2
import numpy as np

#from metrics.metric_util import reorder_image, to_y_channel
import skimage.metrics as metrics
import torch


def calculate_psnr(img1,
                   img2,
                   ):
    #Calculate PSNR (Peak Signal-to-Noise Ratio).

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if len(img1.shape) == 4 and img1.shape[0] != 1:
        psnr = 0
        for i in range(img1.shape[0]):
            img1_i, img2_i = img1[i], img2[i]
            if type(img1[i]) == torch.Tensor:
                img1_i = img1_i.detach().cpu().numpy().transpose(1,2,0)
                img1_i = ((img1_i + 1) * 127.5).astype(np.uint8)
            if type(img2[i]) == torch.Tensor:
                img2_i = img2_i.detach().cpu().numpy().transpose(1,2,0)
                img2_i = ((img2_i + 1) * 127.5).astype(np.uint8)
        
            psnr += metrics.peak_signal_noise_ratio(img2_i, img1_i)
        psnr /= img1.shape[0]
    else:
        if type(img1) == torch.Tensor:
            if len(img1.shape) == 4:
                img1 = img1.squeeze(0)
            img1 = img1.detach().cpu().numpy().transpose(1,2,0)
            img1 = ((img1 + 1) * 127.5).astype(np.uint8)
        if type(img2) == torch.Tensor:
            if len(img2.shape) == 4:
                img2 = img2.squeeze(0)
            img2 = img2.detach().cpu().numpy().transpose(1,2,0)
            img2 = ((img2 + 1) * 127.5).astype(np.uint8)
        
        psnr = metrics.peak_signal_noise_ratio(img2, img1)
    return psnr



def calculate_ssim(img1,
                   img2,
                   ):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if len(img1.shape) == 4 and img1.shape[0] != 1:
        ssim = 0
        for i in range(img1.shape[0]):
            img1_i, img2_i = img1[i], img2[i]
            if type(img1[i]) == torch.Tensor:
                img1_i = img1_i.detach().cpu().numpy().transpose(1,2,0)
                img1_i = ((img1_i + 1) * 127.5).astype(np.uint8)
            if type(img2[i]) == torch.Tensor:
                img2_i = img2_i.detach().cpu().numpy().transpose(1,2,0)
                img2_i = ((img2_i + 1) * 127.5).astype(np.uint8)
        
            ssim += metrics.structural_similarity(img2_i, img1_i, channel_axis = -1)
        ssim /= img1.shape[0]
    else:
        if type(img1) == torch.Tensor:
            if len(img1.shape) == 4:
                img1 = img1.squeeze(0)
            img1 = img1.detach().cpu().numpy().transpose(1,2,0)
            img1 = ((img1 + 1) * 127.5).astype(np.uint8)
        if type(img2) == torch.Tensor:
            if len(img2.shape) == 4:
                img2 = img2.squeeze(0)
            img2 = img2.detach().cpu().numpy().transpose(1,2,0)
            img2 = ((img2 + 1) * 127.5).astype(np.uint8)
        ssim = metrics.structural_similarity(img2, img1, channel_axis = -1)
    return ssim
