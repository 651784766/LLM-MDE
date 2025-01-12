from __future__ import division
import matplotlib
import numpy as np
from path import Path
import datetime
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display
import torch.nn as nn
import os
from torchvision.utils import save_image
import imageio
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms
import os, sys
import math


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_path']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp

def tensor2array(tensor, max_value=255, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
        array = array.transpose(1,2,0)
    return array

def save_depth_tensor(tensor_img,img_dir,filename):
    result = tensor_img.cpu().detach().numpy()
    max_value = result.max()
    if (result.shape[0]==1):
        result = result.squeeze(0)
        result = result/max_value
    elif (result.ndim==2):
        result = result/max_value
    else:
        print("file dimension is not proper!!")
        exit()
    imageio.imwrite(img_dir + '/' + filename,result)

def plot_loss(data, apath, epoch,train,filename):
    axis = np.linspace(1, epoch, epoch)
    
    label = 'Total Loss'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(data), label=label)
    plt.legend()
    if train is False:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('x100 = Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.savefig(os.path.join(apath, filename))
    plt.close(fig)
    plt.close('all')

def train_plot(save_dir,tot_loss, rmse, loss_list, rmse_list, tot_loss_dir,rmse_dir,loss_pdf, rmse_pdf, count,istrain):
    open_type = 'a' if os.path.exists(tot_loss_dir) else 'w'
    loss_log_file = open(tot_loss_dir, open_type)
    rmse_log_file = open(rmse_dir,open_type)

    loss_list.append(tot_loss)
    rmse_list.append(rmse)

    plot_loss(loss_list, save_dir, count, istrain, loss_pdf)
    plot_loss(rmse_list, save_dir, count, istrain, rmse_pdf)
    loss_log_file.write(('%.5f'%tot_loss) + '\n')
    rmse_log_file.write(('%.5f'%rmse) + '\n')
    loss_log_file.close()
    rmse_log_file.close()

def validate_plot(save_dir,tot_loss, loss_list, tot_loss_dir,loss_pdf, count,istrain):
    open_type = 'a' if os.path.exists(tot_loss_dir) else 'w'
    loss_log_file = open(tot_loss_dir, open_type)

    loss_list.append(tot_loss)

    plot_loss(loss_list, save_dir, count, istrain, loss_pdf)
    loss_log_file.write(('%.5f'%tot_loss) + '\n')
    loss_log_file.close()

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()

    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x

def imgrad_loss(pred, gt, mask=None):
    N,C,_,_ = pred.size()
    grad_y, grad_x = imgrad(pred)
    grad_y_gt, grad_x_gt = imgrad(gt)
    grad_y_diff = torch.abs(grad_y - grad_y_gt)
    grad_x_diff = torch.abs(grad_x - grad_x_gt)
    if mask is not None:
        grad_y_diff[~mask] = 0.1*grad_y_diff[~mask]
        grad_x_diff[~mask] = 0.1*grad_x_diff[~mask]
    return (torch.mean(grad_y_diff) + torch.mean(grad_x_diff))

def BerHu_loss(valid_out, valid_gt):         
    diff = valid_out - valid_gt
    diff_abs = torch.abs(diff)
    c = 0.2*torch.max(diff_abs.detach())         
    mask2 = torch.gt(diff_abs.detach(),c)
    diff_abs[mask2] = (diff_abs[mask2]**2 +(c*c))/(2*c)
    return diff_abs.mean()

def scale_invariant_loss(valid_out, valid_gt):
    logdiff = torch.log(valid_out) - torch.log(valid_gt)
    scale_inv_loss = torch.sqrt((logdiff ** 2).mean() - 0.85*(logdiff.mean() ** 2))*10.0
    return scale_inv_loss

def make_mask(depths, crop_mask, dataset):
    # masking valied area
    if dataset == 'KITTI':
        valid_mask = depths > 0.001
    else:
        valid_mask = depths > 0.001
    
    if dataset == "KITTI":
        if(crop_mask.size(0) != valid_mask.size(0)):
            crop_mask = crop_mask[0:valid_mask.size(0),:,:,:]
        final_mask = crop_mask|valid_mask
    else:
        final_mask = valid_mask
        
    return valid_mask, final_mask


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)

# for test
def normalize_result_test(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.


    value = np.expand_dims(value, 0)  # Keep this line to expand dimensions
    return torch.tensor(value)  # Convert back to a Tensor


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


# def compute_errors(gt, pred):

#     gt = gt.cpu().numpy()# 自己加的
#     pred = pred.cpu().numpy()# 自己加的

#     thresh = np.maximum((gt / pred), (pred / gt))
#     d1 = (thresh < 1.25).mean()
#     d2 = (thresh < 1.25 ** 2).mean()
#     d3 = (thresh < 1.25 ** 3).mean()

#     rms = (gt - pred) ** 2
#     rms = np.sqrt(rms.mean())

#     log_rms = (np.log(gt) - np.log(pred)) ** 2
#     log_rms = np.sqrt(log_rms.mean())

#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     sq_rel = np.mean(((gt - pred) ** 2) / gt)

#     err = np.log(pred) - np.log(gt)
#     silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

#     err = np.abs(np.log10(pred) - np.log10(gt))
#     log10 = np.mean(err)

#     return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def compute_errors(gt, pred):
    
    #mask = gt > 0.1
    #gt=gt[mask]
    #pred=pred[mask]

    # 将gt和pred转换为NumPy数组
    depth_gt1 = gt.detach()
    depth_est = pred.detach()

    # 归一化 depth_est
    depth_est_min = depth_est.min()
    depth_est_max = depth_est.max()
    depth_est = (depth_est - depth_est_min) / (depth_est_max - depth_est_min)
    
    # 归一化 depth_gt
    depth_gt_min = depth_gt1.min()
    depth_gt_max = depth_gt1.max()
    depth_gt1 = (depth_gt1 - depth_gt_min) / (depth_gt_max - depth_gt_min)

    # 更新 pred 和 gt 的值
    pred = depth_est.cpu().numpy()
    gt = depth_gt1.cpu().numpy()

    pred[pred < 1e-3] = 1e-3
    gt[gt < 1e-3] = 1e-3
    
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()  #创建了一个bool数组，通过mean方法计算其中true和false比例为d1
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    metrics = {'d1': d1,'d2': d2,'d3': d3}
    return metrics



# # Scale-Invariant Logarithmic (SILOG) 
# class silog_loss(nn.Module):
#     def __init__(self, variance_focus):
#         super(silog_loss, self).__init__()
#         self.variance_focus = variance_focus

#     def forward(self, depth_est, depth_gt, mask):
#         d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
#         return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
    

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        squared_diff = (d ** 2).mean()
        mean_diff = (d.mean() ** 2)
        
        # 添加一个小的常数以避免负值
        loss = torch.sqrt(torch.clamp(squared_diff - self.variance_focus * mean_diff, min=1e-6)) * 10.0
        return loss

class SILOG_loss(nn.Module):
    def __init__(self, variance_focus, delta=1.0):
        super(SILOG_loss, self).__init__()
        self.variance_focus = variance_focus
        self.delta = delta


    def forward(self, depth_est, depth_gt):
        depth_est = torch.clamp_min(depth_est, 1e-3)
        depth_gt = torch.clamp_min(depth_gt, 1e-3)

        # 计算 Huber 损失
        huber_loss = nn.functional.huber_loss(depth_est, depth_gt, delta=self.delta)
        # 计算 log(d_est) - log(d_gt) 的差值
        d = torch.log(depth_est) - torch.log(depth_gt)
        # 计算 mean(d^2) 和 mean(huber_loss)^2 的差值
        mean_d_squared = (d ** 2).mean()
        mean_huber_loss_squared = (huber_loss.mean() ** 2)
        mean_d_squared_diff = mean_d_squared - self.variance_focus * mean_huber_loss_squared

        # 返回 SILOG 损失
        return torch.sqrt(torch.max(mean_d_squared_diff, torch.tensor(0.0).to(depth_est.device)))


    # def forward(self, depth_est, depth_gt):
    #     delta =1.0
    #     depth_est = torch.clamp_min(depth_est, 1e-3)
    #     depth_gt = torch.clamp_min(depth_gt, 1e-3)

    #     d = torch.log(depth_est) - torch.log(depth_gt)
    #     loss = torch.where(
    #         torch.abs(depth_est - depth_gt) <= delta,
    #         0.5 * (depth_est - depth_gt) ** 2,
    #         delta * (torch.abs(depth_est - depth_gt) - delta)
    #     )
    #     mean_d = loss.mean()
    #     mean_d_squared = (mean_d ** 2)
    #     mean_d_squared_diff = (d ** 2).mean() - self.variance_focus * mean_d_squared
        
    #     return torch.sqrt(torch.max(mean_d_squared_diff, torch.tensor(0.0).to(depth_est.device)))

def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])


def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused


class DistributedSamplerNoEvenlyDivisible(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        rest = len(self.dataset) - num_samples * self.num_replicas
        if self.rank < rest:
            num_samples += 1
        self.num_samples = num_samples
        self.total_size = len(dataset)
        # self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        # assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch