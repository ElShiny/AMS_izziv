import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import json
from utils.config import args
from utils.helper_functions import *

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class AMS_Dataset(Data.Dataset):
    def __init__(self, json_path):
        # 
        self.json = json.load(open(json_path, 'r'))
        self.json_pairs = self.json["training_paired_images"]
        


    def __len__(self):
        # 
        return self.json["numPairedTraining"]


    def __getitem__(self, index):
        #load the fixed and moving images
        t_img_fx = sitk.ReadImage(args.train_dir + "\\" + self.json_pairs[index]["fixed"][1:].split('/')[2])
        t_img_mv = sitk.ReadImage(args.train_dir + "\\" + self.json_pairs[index]["moving"][1:].split('/')[2])
        t_img_fx_msk = sitk.ReadImage(args.mask_dir + "\\"  + self.json_pairs[index]["fixed"][1:].split('/')[2])
        t_img_mv_msk = sitk.ReadImage(args.mask_dir + "\\"  + self.json_pairs[index]["moving"][1:].split('/')[2])


        #
        fixed_img = sitk.GetArrayFromImage(t_img_fx)[np.newaxis, ...]
        moving_img = sitk.GetArrayFromImage(t_img_mv)[np.newaxis, ...]
        fixed_msk = sitk.GetArrayFromImage(t_img_fx_msk)[np.newaxis, ...]
        moving_msk = sitk.GetArrayFromImage(t_img_mv_msk)[np.newaxis, ...]



        fixed_img = (fixed_img - fixed_img.mean()) / fixed_img.std()
        moving_img = (moving_img - moving_img.mean()) / moving_img.std()

        #normalise the images
        #min_fx = fixed_img.min()
        #max_fx = fixed_img.max()
        #fixed_img = (fixed_img-min_fx)/(max_fx-min_fx)
#
        #min_mv = moving_img.min()
        #max_mv = moving_img.max()
        #moving_img = (moving_img-min_mv)/(max_mv-min_mv)
        
        #apply mask
        fixed_img = apply_mask(fixed_img, fixed_msk)
        moving_img = apply_mask(moving_img, moving_msk)
        moving_img = apply_mask(moving_img, fixed_msk)

        #decimate images to 1/4 of the original size
        fixed_img_dec = fixed_img[:, ::2, ::2, ::2]
        moving_img_dec = moving_img[:, ::2, ::2, ::2]

        #plt.figure(1)
        #plt.subplot(1, 2, 1)
        ##add imagew name
        #plt.title("Fixed Image")
        #plt.imshow(fixed_img_dec[0, :, :, 50], cmap='gray')
        #
        #plt.subplot(1, 2, 2)
        #plt.title("Moving Image")
        #plt.imshow(moving_img_dec[0, :, :, 50], cmap='gray')
        #plt.figure(4)
        #plt.hist(fixed_img_dec.flatten(), bins=100)
        #plt.hist(moving_img_dec.flatten(), bins=100)
        #plt.show()

        index_mv = self.json_pairs[index]["fixed"].split('/')[-1]
        index_fx = self.json_pairs[index]["moving"].split('/')[-1]
        # 
        return fixed_img_dec, moving_img_dec, index_mv, index_fx


class AMS_Dataset_val(Data.Dataset):
    def __init__(self, json_path):
        # 
        self.json = json.load(open(json_path, 'r'))
        self.json_pairs = self.json["registration_val"]
        


    def __len__(self):
        # 
        return self.json["numRegistration_val"]


    def __getitem__(self, index):
        #load the fixed and moving images
        t_img_fx = sitk.ReadImage(args.train_dir + "\\"  + self.json_pairs[index]["fixed"][1:].split('/')[2])
        t_img_mv = sitk.ReadImage(args.train_dir + "\\"  + self.json_pairs[index]["moving"][1:].split('/')[2])
        t_img_label_fx = sitk.ReadImage(args.label_dir + "\\"  + self.json_pairs[index]["fixed"].split('/')[-1].split('.')[0] + ".nii.gz")
        t_img_label_mv = sitk.ReadImage(args.label_dir + "\\"  + self.json_pairs[index]["moving"].split('/')[-1].split('.')[0] + ".nii.gz")
        t_img_fx_msk = sitk.ReadImage(args.mask_dir + "\\"  + self.json_pairs[index]["fixed"][1:].split('/')[2])
        t_img_mv_msk = sitk.ReadImage(args.mask_dir + "\\"  + self.json_pairs[index]["moving"][1:].split('/')[2])


        fixed_img = sitk.GetArrayFromImage(t_img_fx)[np.newaxis, ...]
        moving_img = sitk.GetArrayFromImage(t_img_mv)[np.newaxis, ...]
        fixed_label = sitk.GetArrayFromImage(t_img_label_fx)[np.newaxis, ...]
        moving_label = sitk.GetArrayFromImage(t_img_label_mv)[np.newaxis, ...]
        fixed_msk = sitk.GetArrayFromImage(t_img_fx_msk)[np.newaxis, ...]
        moving_msk = sitk.GetArrayFromImage(t_img_mv_msk)[np.newaxis, ...] 

        fixed_img = (fixed_img - fixed_img.mean()) / fixed_img.std()
        moving_img = (moving_img - moving_img.mean()) / moving_img.std()

        #decimate images to 1/4 of the original size
        fixed_img_dec = fixed_img[:, ::2, ::2, ::2]
        moving_img_dec = moving_img[:, ::2, ::2, ::2]
        fixed_label_dec = fixed_label[:, ::2, ::2, ::2]
        moving_label_dec = moving_label[:, ::2, ::2, ::2]
        fixed_msk_dec = fixed_msk[:, ::2, ::2, ::2]
        moving_msk_dec = moving_msk[:, ::2, ::2, ::2]

        #apply masks
        fixed_img_dec = apply_mask(fixed_img_dec, fixed_msk_dec)
        moving_img_dec = apply_mask(moving_img_dec, moving_msk_dec)
        moving_img_dec = apply_mask(moving_img_dec, fixed_msk_dec)
        fixed_label_dec = apply_mask(fixed_label_dec, fixed_msk_dec)
        moving_label_dec = apply_mask(moving_label_dec, moving_msk_dec)

        index_fx = self.json_pairs[index]["fixed"].split('/')[1:]
        index_mv = self.json_pairs[index]["moving"].split('/')[1:]

        # 
        return fixed_img_dec, moving_img_dec, fixed_label_dec, moving_label_dec, index_fx, index_mv