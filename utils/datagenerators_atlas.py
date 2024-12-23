import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import json
from utils.config import args
from utils.helper_functions import *


class AMS_Dataset(Data.Dataset):
    def __init__(self, json_path):
        #loading the json file
        self.json = json.load(open(json_path, 'r'))
        self.json_pairs = self.json["training_paired_images"]
        

    def __len__(self):
        # 
        return self.json["numPairedTraining"]


    def __getitem__(self, index):
        #load the fixed and moving images
        t_img_fx = sitk.ReadImage(args.train_dir + "/" + os.path.basename(self.json_pairs[index]["fixed"]))
        t_img_mv = sitk.ReadImage(args.train_dir + "/" + os.path.basename(self.json_pairs[index]["moving"]))

        #turn into a 4D tensor
        fixed_img = sitk.GetArrayFromImage(t_img_fx)[np.newaxis, ...]
        moving_img = sitk.GetArrayFromImage(t_img_mv)[np.newaxis, ...]

        if args.norm == 'meanstd':
            #normalising by subtracting the mean and dividing by the standard deviation
            fixed_img = (fixed_img - fixed_img.mean()) / fixed_img.std()
            moving_img = (moving_img - moving_img.mean()) / moving_img.std()
        elif args.norm == 'minmax':
            #normalising by min-max scaling
            min_fx = fixed_img.min()
            max_fx = fixed_img.max()
            fixed_img = (fixed_img-min_fx)/(max_fx-min_fx)

            min_mv = moving_img.min()
            max_mv = moving_img.max()
            moving_img = (moving_img-min_mv)/(max_mv-min_mv)
        else :
            pass
        
        #load and apply masks
        if args.mask_dir != "":
            t_img_fx_msk = sitk.ReadImage(args.mask_dir + "/"  + os.path.basename(self.json_pairs[index]["fixed"]))
            t_img_mv_msk = sitk.ReadImage(args.mask_dir + "/"  + os.path.basename(self.json_pairs[index]["moving"]))
            fixed_msk = sitk.GetArrayFromImage(t_img_fx_msk)[np.newaxis, ...]
            moving_msk = sitk.GetArrayFromImage(t_img_mv_msk)[np.newaxis, ...]

            fixed_img = apply_mask(fixed_img, fixed_msk)
            moving_img = apply_mask(moving_img, moving_msk)

            #uncomment to apply the fixed mask to the moving image
            #moving_img = apply_mask(moving_img, fixed_msk)

        #only used for printing the image names
        index_mv = os.path.basename(self.json_pairs[index]["fixed"])
        index_fx = os.path.basename(self.json_pairs[index]["moving"])
        # 
        return fixed_img, moving_img, index_mv, index_fx


class AMS_Dataset_val(Data.Dataset):
    def __init__(self, json_path):
        #loading the json file
        self.json = json.load(open(json_path, 'r'))
        self.json_pairs = self.json["registration_val"]
        

    def __len__(self):
        # 
        return self.json["numRegistration_val"]


    def __getitem__(self, index):
        #load the fixed and moving images
        t_img_fx = sitk.ReadImage(args.train_dir + "/"  + os.path.basename(self.json_pairs[index]["fixed"]))
        t_img_mv = sitk.ReadImage(args.train_dir + "/"  + os.path.basename(self.json_pairs[index]["moving"]))
        t_img_label_fx = sitk.ReadImage(args.label_dir + "/"  + os.path.basename(self.json_pairs[index]["fixed"]))
        t_img_label_mv = sitk.ReadImage(args.label_dir + "/"  + os.path.basename(self.json_pairs[index]["moving"]))


        #turn into a 4D tensor
        fixed_img = sitk.GetArrayFromImage(t_img_fx)[np.newaxis, ...]
        moving_img = sitk.GetArrayFromImage(t_img_mv)[np.newaxis, ...]
        fixed_label = sitk.GetArrayFromImage(t_img_label_fx)[np.newaxis, ...]
        moving_label = sitk.GetArrayFromImage(t_img_label_mv)[np.newaxis, ...]

        if args.norm == 'meanstd':
            #normalising by subtracting the mean and dividing by the standard deviation
            fixed_img = (fixed_img - fixed_img.mean()) / fixed_img.std()
            moving_img = (moving_img - moving_img.mean()) / moving_img.std()
            
        elif args.norm == 'minmax':
            #normalising by min-max scaling
            min_fx = fixed_img.min()
            max_fx = fixed_img.max()
            fixed_img = (fixed_img-min_fx)/(max_fx-min_fx)

            min_mv = moving_img.min()
            max_mv = moving_img.max()
            moving_img = (moving_img-min_mv)/(max_mv-min_mv)
        else :
            pass



        #load and apply masks
        if args.mask_dir != "":
            t_img_fx_msk = sitk.ReadImage(args.mask_dir + "/"  + os.path.basename(self.json_pairs[index]["fixed"]))
            t_img_mv_msk = sitk.ReadImage(args.mask_dir + "/"  + os.path.basename(self.json_pairs[index]["moving"]))

            fixed_msk = sitk.GetArrayFromImage(t_img_fx_msk)[np.newaxis, ...]
            moving_msk = sitk.GetArrayFromImage(t_img_mv_msk)[np.newaxis, ...] 

            fixed_img = apply_mask(fixed_img, fixed_msk)
            moving_img = apply_mask(moving_img, moving_msk)
            moving_img = apply_mask(moving_img, fixed_msk)
            fixed_label = apply_mask(fixed_label, fixed_msk)
            moving_label = apply_mask(moving_label, moving_msk)

        #image paths
        index_fx = self.json_pairs[index]["fixed"]
        index_mv = self.json_pairs[index]["moving"]
        # 
        return fixed_img, moving_img, fixed_label, moving_label, index_fx, index_mv