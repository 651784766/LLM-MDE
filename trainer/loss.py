import torch
import torch.nn.functional as F

def ssim(prediction, target, window_size=11, size_average=True, full=False):
    k1 = 0.01
    k2 = 0.03
    L = 1  # 像素值的动态范围

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mu1 = F.avg_pool2d(prediction, window_size, stride=1)
    mu2 = F.avg_pool2d(target, window_size, stride=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(prediction.pow(2), window_size, stride=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(target.pow(2), window_size, stride=1) - mu2_sq
    sigma12 = F.avg_pool2d(prediction * target, window_size, stride=1) - mu1_mu2

    if full:
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    else:
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator

    if size_average:
        return torch.mean((1.0 - ssim_map) / 2.0)
    else:
        return torch.mean((1.0 - ssim_map) / 2.0, dim=(1, 2, 3))


import torch
import torch.nn as nn
from torchmetrics.functional import image_gradients

def adjust_index(old_index, old_shape, new_shape):
    old_size = old_shape[2] * old_shape[3]
    new_size = new_shape[2] * new_shape[3]
    
    # 计算索引百分比
    index_percent = old_index / old_size
    
    # 将百分比映射到新的尺寸上
    new_index = index_percent * new_size
    
    return new_index


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    # print("prediction.shape=", prediction.shape)

    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    # print("a_00.shape=", a_00.shape)
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    #valid = torch.where(det != 0)

    valid = det.nonzero()
    #print("x_0 size:", x_0.size())
    #print("valid indices:", valid)
    x_0_index = adjust_index(valid[0], prediction.shape, target.shape)
    x_1_index = adjust_index(valid[1], prediction.shape, target.shape)

    x_0[x_0_index] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[x_1_index] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


class CeAndMse(nn.Module):
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    def __init__(self, wt_ce=0.5):
        super().__init__()
        self.wt_ce = wt_ce

    def forward(self, *args, **kwargs):
        y0 = self.ce.forward(*args, **kwargs)
        y1 = self.mse.forward(*args, **kwargs)
        return self.wt_ce * y0 / 1000 + (1. - self.wt_ce) * y1


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target):
        # preprocessing
        mask = target > 0

        # calculate
        scale, shift = compute_scale_and_shift(prediction, target, mask)

        # print(f"scale={scale.item()}, shift={shift.item()}")
        scale[
            not (0.95 < scale < 1.05)
        ] = 1  # added. But why did authors not clamp scale and shift???
        shift[not (-5 < shift < 5)] = 0
        # print(f"scale={scale.item()}, shift={shift.item()}.....")

        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target, mask
            )

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
