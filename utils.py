import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import logging
from ssim_torch import ssim
import h5py


def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + 'mask.mat')
    mask = mask['mask128']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 31))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def LoadTrain(path):
    hsi_list = []
    scene_list = os.listdir(path)
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        hsi_dict = h5py.File(scene_path)
        hsi = hsi_dict['rad']
        hsi = hsi/np.max(hsi)
        hsi = hsi.astype(np.float32)
        hsi_list.append(hsi)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return hsi_list


def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    test_data = np.zeros((len(scene_list), 31, 128, 128))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        hsi_dict = h5py.File(scene_path)
        hsi = hsi_dict['rad']
        hsi = hsi[:, 512:640, 512:640]
        hsi = hsi/np.max(hsi)
        hsi = hsi.astype(np.float32)
        test_data[i, :, :, :] = hsi
        print(i, hsi.shape, hsi.max(), hsi.min())
    test_data = torch.from_numpy(test_data)
    return test_data


def psnr(img1, img2):
    psnr_list = []
    for i in range(img1.shape[0]):
        total_psnr = 0
        # PIXEL_MAX = img2.max()
        PIXEL_MAX = img2[i, :, :, :].max()
        for ch in range(31):
            mse = np.mean((img1[i, :, :, ch] - img2[i, :, :, ch]) ** 2)
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr_list.append(total_psnr / img1.shape[3])
    return psnr_list


def torch_psnr(img, ref):  # input [31,256,256]
    nC = img.shape[0]
    pixel_max = torch.max(ref)
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr / nC


def torch_ssim(img, ref):  # input [31,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def shuffle_crop(train_data, batch_size):
    index = np.random.choice(range(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, 31, 128, 128), dtype=np.float32)

    for i in range(batch_size):
        _, h, w = train_data[index[i]].shape
        x_index = np.random.randint(0, h - 128)
        y_index = np.random.randint(0, w - 128)
        processed_data[i, :, :, :] = train_data[index[i]][:, x_index:x_index + 128, y_index:y_index + 128]
    gt_batch = torch.from_numpy(processed_data)
    return gt_batch


def gen_meas_torch(data_batch, mask3d_batch, is_training=True):
    nC = data_batch.shape[1]

    if is_training is False:
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0, :, :, :]).expand([batch_size, nC, H, W]).cuda().float()
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1) / nC * 2  # meas scale

    y_temp = shift_back(meas)
    PhiTy = torch.mul(y_temp, mask3d_batch)
    return PhiTy


# def gen_meas_torch(data_batch, mask3d_batch, is_training=True):
#     nC = data_batch.shape[1]
#
#     if is_training is False:
#         [batch_size, nC, H, W] = data_batch.shape
#         mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()
#     temp = shift(mask3d_batch*data_batch, 2)
#     meas = torch.sum(temp, 1)/nC*2          # meas scale
#     y_temp = shift_back(meas)
#     return y_temp

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output


def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 31
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
