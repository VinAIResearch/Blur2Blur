import os
from os.path import join as osp

import cv2
import numpy as np
import torch
from tqdm import tqdm
import math


def calc_psnr(img1, img2):
    mse = np.mean(np.power((img1 / 255.0 - img2 / 255.0), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def reorder_image(img, input_order="HWC"):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ["HWC", "CHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are " "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    return img


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.0


def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out


def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode="replicate")
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d


def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()

    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1**2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2**2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


##########################################################################
##########################################################################
##########################################################################


def calculate_psnr(img1, img2, input_order="HWC", test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are " '"HWC" and "CHW"')
    if type(img1) is torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) is torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # if crop_border != 0:
    #     img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
    #     img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_value = 1.0 if img1.max() <= 1 else 255.0
    return 20.0 * np.log10(max_value / np.sqrt(mse))


def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window, borderType=bt)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, input_order="HWC", test_y_channel=False):
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

    assert img1.shape == img2.shape, f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are " '"HWC" and "CHW"')

    if type(img1) is torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) is torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        return _ssim_cly(img1[..., 0], img2[..., 0])

    ssims = []
    # ssims_before = []

    # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
    # print('.._skimage',
    #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
    max_value = 1 if img1.max() <= 1 else 255
    with torch.no_grad():
        final_ssim = _ssim_3d(img1, img2, max_value)
        ssims.append(final_ssim)

    # for i in range(img1.shape[2]):
    #     ssims_before.append(_ssim(img1, img2))

    # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
    # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

    return np.array(ssims).mean()


if __name__ == "__main__":
    # dataset_name = "Face"
    dataset_name = "Street"

    if dataset_name == "Face":
        deblur_path = (
            "/home/dangpb1/Research/CamSpecDeblurring/blur2blur/results/CycleGAN_FaceMod2_right/test_deb_latest/images"
        )
        gt_path = "/home/dangpb1/Research/CamSpecDeblurring/blur2blur/datasets/datasets/Blur2Vid/deblurring-b2b/mix_mod2_sharp/face"
    elif dataset_name == "Street":
        deblur_path = "/home/dangpb1/Research/CamSpecDeblurring/uvcgan2/outdir/b2b_base/model_m(b2b_base)_d(basic)_g(unet_256)_unet-bsd_FixP2P_Streetmod2_unet_idt2.0_gp0.005_(True:0.005:0.0001:2.0)/evals/epoch_0425/images_eval-test/real_a"
        gt_path = "/home/dangpb1/Research/CamSpecDeblurring/blur2blur/datasets/datasets/Blur2Vid/deblurring-b2b/mix_mod2_sharp/street"
    elif dataset_name == "REDS":
        deblur_path = ""
        gt_path = "/home/dangpb1/Research/CamSpecDeblurring/blur2blur/datasets/datasets/REDS/b2b/mod2_sharp/trainA"

    file_img = sorted(os.listdir(deblur_path))

    psnr_total, ssim_total = 0.0, 0.0
    for each in tqdm(file_img):
        deblur_img = cv2.imread(osp(deblur_path, each))
        gt_img = cv2.imread(osp(gt_path, each))
        if gt_img.shape[0] != 256:
            height, width = gt_img.shape[:2]

            if height < width:
                new_height = 256
                new_width = int(width * (new_height / height))
            else:
                new_width = 256
                new_height = int(height * (new_width / width))

            gt_img = cv2.resize(gt_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # deblur_img = cv2.resize(deblur_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # breakpoint()
        psnr_total += calculate_psnr(deblur_img, gt_img)
        ssim_total += calculate_ssim(deblur_img, gt_img)
        # breakpoint()

    print("PSNR: ", psnr_total / len(file_img))
    print("SSIM: ", ssim_total / len(file_img))
