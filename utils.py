from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from IPython import display
import itertools
import torch.nn as nn
import os
from torchvision.utils import save_image
import imageio


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

def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()