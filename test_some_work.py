import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import h5py

# Load and generate Mask
# mask_dict = io.loadmat('../Data/mask_old.mat')
# mask = mask_dict['mask']
# mask = np.array(mask)
# plt.imshow(mask)
# plt.show()
# print(mask.shape)
# mask = np.random.randint(0, 2, (128, 128), int)
# plt.imshow(mask)
# plt.show()
# print(mask.shape)
# io.savemat('../Data/mask.mat', {'mask128': mask})


# path = 'D:/Data/ICVL/'
# path1 = 'D:/Data/Havard/CZ_hsdb/'
# path2 = 'D:/Data/Havard/CZ_hsdbi/'
# hsi_list = os.listdir(path)
# hsi_list1 = os.listdir(path1)
# hsi_list2 = os.listdir(path2)
# print(len(hsi_list))
# load ICVL
# max=0
# min=10000
# for i in hsi_list:
#     hsi_path = path + i
#     hsi_dict = h5py.File(hsi_path)
#     print(hsi_dict.keys())
#     hsi_rad = hsi_dict['rad']
#     hsi_rgb = np.asarray(hsi_dict['rgb'])
#     hsi_rad = np.asarray(hsi_rad)
#     print(np.max(hsi_rad), np.min(hsi_rad))
#     if(np.max(hsi_rad) > max):
#         max = np.max(hsi_rad)
#     if(np.min(hsi_rad) < min):
#         min = np.min(hsi_rad)
    # plt.imshow(hsi_rgb.transpose((2, 1, 0)))
    # plt.show()
# load Havard
# for i in hsi_list2:
#     hsi_path = path2+i
#     if 'mat' not in hsi_path:
#         continue
#     hsi_dict = io.loadmat(hsi_path)
#     print(hsi_dict.keys())
#     hsi_ref = hsi_dict['ref']
#     plt.imshow(hsi_ref[:,:,0])
#     plt.show()


# Load Cave data set

# def LoadPngSet(path):
#     imgs = []
#     scene_list = os.listdir(path)
#     scene_list.sort()
#     print('all sences:', len(scene_list))
#     max_ = 0
#     for i in range(len(scene_list)):
#         scene_path = path + scene_list[i]
#         if 'png' not in scene_path:
#             continue
#         scene = plt.imread(scene_path)
#         img = scene
#         img = img.astype(np.float32)
#         imgs.append(img)
#         print('Sence {} is loaded. {}'.format(i, scene_list[i]))
#     return imgs
#
# dataset_path = 'D:/Data/CAVE/'
# dataset_list = os.listdir(dataset_path)
# print('dataset num:', len(dataset_list))
# for i in range(len(dataset_list)):
#     data_path = dataset_path+dataset_list[i]+'/'+dataset_list[i]+'/'
#     imgs = LoadPngSet(data_path)


# Show the reconstruction results

# path_TSA = 'recon/Test_197_31.68_0.920.mat'
# path_SwinCA = 'recon/Test_191_33.96_0.950.mat'
# recon_TSA = io.loadmat(path_TSA)
# recon_SwinCA = io.loadmat(path_SwinCA)
# truth = recon_TSA['truth']
# pred_TSA = recon_TSA['pred']
# psnr_TSA = recon_TSA['psnr_list']
# ssim_TSA = recon_TSA['ssim_list']
# pred_SwinCA = recon_SwinCA['pred']
# psnr_SwinCA = recon_SwinCA['psnr_list']
# ssim_SwinCA = recon_SwinCA['ssim_list']
# # for i in range(truth.shape[0]):
# #     for j in range(truth.shape[3]):
# #         imgs = np.hstack([truth[i, :, :, j], pred_TSA[i, :, :, j], pred_SwinCA[i, :, :, j]])
# #         winName = 'Truth-TSA-SwinCA in img:' + str(i) + ' Channel:'+str(j)
# #         cv2.namedWindow(winName)
# #         cv2.moveWindow(winName, 400, 200)
# #         cv2.imshow(winName, imgs)
# #         cv2.waitKey()
# #         cv2.destroyAllWindows()
# for i in range(10):
#     print(ssim_TSA[0][i], ssim_SwinCA[0][i])

path = '../Data/test/'
def loadmatset(path):
    list = os.listdir(path)
    for i in list:
        imgpath = path+i
        if 'mat' not in imgpath:
            continue
        imgdict = io.loadmat(imgpath)
        print(imgdict.keys())
