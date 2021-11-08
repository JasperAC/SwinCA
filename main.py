# from dataloader import dataset
from models import Swin_Net
from utils import *
# from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

training_path = "../Data/training/"
validating_path = "../Data/validating/"
mask_path = "../Data/"

batch_size = 4
last_train = 0  # for finetune
model_save_filename = ''  # for finetune
max_epoch = 300
learning_rate = 0.0004
epoch_sam_num = 5000
batch_num = int(np.floor(epoch_sam_num / batch_size))

mask3d_batch = generate_masks(mask_path, batch_size)

training_set = LoadTraining(training_path)
validating_data = LoadValidation(validating_path)
model = Swin_Net().cuda()

if last_train != 0:
    model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
mse = torch.nn.MSELoss().cuda()


def train(epoch, learning_rate, logger):
    epoch_loss = 0
    begin = time.time()
    for i in range(batch_num):
        gt_batch = shuffle_crop(training_set, batch_size)
        gt = Variable(gt_batch).cuda().float()
        y = gen_meas_torch(gt, mask3d_batch, is_training=True)
        optimizer.zero_grad()
        model_out = model(y)
        Loss = torch.sqrt(mse(model_out, gt))
        epoch_loss += Loss.data
        Loss.backward()
        optimizer.step()
    end = time.time()
    logger.info(
        "===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, epoch_loss / batch_num, (end - begin)))


def validate(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = validating_data.cuda().float()
    test_PhiTy = gen_meas_torch(test_gt, mask3d_batch, is_training=False)
    model.eval()
    model_out = torch.zeros(test_PhiTy.shape).cuda()
    for k in range(test_gt.shape[0]):
        with torch.no_grad():
            model_out[k, :, :, :] = model(test_PhiTy[k, :, :, :].unsqueeze(0)).squeeze(0)
        print(model_out[k,:,:,:].shape)
        print(test_gt[k,:,:,:].shape)
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: validating psnr = {:.2f}, ssim = {:.3f}'.format(epoch, psnr_mean, ssim_mean))
    model.train()
    return (pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean)


def checkpoint(epoch, model_path, logger):
    model_out_path = './' + model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


def main(learning_rate):
    if model_save_filename == '':
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
    else:
        date_time = model_save_filename
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(learning_rate, batch_size))
    psnr_max = 0

    for epoch in range(last_train + 1, last_train + max_epoch + 1):
        train(epoch, learning_rate, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = validate(epoch, logger)

        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 27:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(epoch, model_path, logger)

        # if (epoch % lr_epoch == 0) and (epoch < 200):
        # learning_rate = learning_rate * lr_scale
        # logger.info('Current learning rate: {}\n'.format(learning_rate))


if __name__ == '__main__':
    main(learning_rate)


