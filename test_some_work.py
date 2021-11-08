import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# mask_dict = io.loadmat('Data/mask_old.mat')
# mask = mask_dict['mask']
# mask = np.array(mask)
# plt.imshow(mask)
# plt.show()
# print(mask.shape)
# mask = np.random.randint(0, 2, (256, 256), int)
# plt.imshow(mask)
# plt.show()
# print(mask.shape)
# io.savemat('Data/mask.mat', {'mask': mask})\

# path = 'D:/Data/TSA_test_data/Truth/'
# mask_list = os.listdir(path)
# for i in mask_list:
#     mask_path = path + i
#     mask_dict = io.loadmat(mask_path)
#     print(mask_dict.keys())
#


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
#
# dataset_path = 'D:/Data/CAVE/'
# dataset_list = os.listdir(dataset_path)
# print('dataset num:', len(dataset_list))
# for i in range(len(dataset_list)):
#     data_path = dataset_path+dataset_list[i]+'/'+dataset_list[i]+'/'
#     imgs = LoadPngSet(data_path)

def LoadData(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('sences:', len(scene_list))
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = io.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        img = img.astype(np.float32)
        plt.imshow(img[:, :, 0])
        plt.show()
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

        # imgs = cv2.imread(scene_path)
        # cv2.imshow('show', imgs)
    return imgs

path1 = 'D:/Data/TSA_test_data/truth/'
path2 = 'D:/Data/TSA_training_data/'
path3 = 'D:/Data/KAIST/'
LoadData(path2)


