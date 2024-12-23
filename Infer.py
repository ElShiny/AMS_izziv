# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
# internal imports
from utils import losses
from utils.config import args
from utils.datagenerators_atlas import AMS_Dataset_val
from Models.STN import SpatialTransformer
from natsort import natsorted
from utils.helper_functions import *


from Models.TransMatch import TransMatch

import time
import warnings
warnings.filterwarnings("ignore")

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def train():

    # gpu setting
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    print("Image size: ", args.image_size)


    # load the model
    net = TransMatch(args).to(device)
    best_model = torch.load(args.model_dir)['state_dict']
    net.load_state_dict(best_model)

    STN = SpatialTransformer(tuple(args.image_size)).to(device)
    STN_label = SpatialTransformer(tuple(args.image_size), mode="nearest").to(device)

    net.train()
    STN.train()

    # Get all the names of the training data
    DSV = AMS_Dataset_val(json_path=args.dataset_cfg)
    print("Number of validating images: ", len(DSV))
    DLV = Data.DataLoader(DSV, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    input_validation_all = iter(DLV)

    net.eval()
    STN.eval()
    STN_label.eval()

    TIME = []
    with torch.no_grad():
        for input_fixed, input_moving, fixed_label, input_label, name1, name2 in input_validation_all:
            #prepare a name for the output files
            fixedName = extract_id_from_filename(str(name1)[2:-3])
            movingName = extract_id_from_filename(str(name2)[2:-3])
            tmpName = f"disp_{fixedName}_{movingName}"

            #read the reference image
            f_img = sitk.ReadImage(args.train_dir + "/" + os.path.basename(str(name2)[2:-3]))
            #print(args.train_dir + "\\" + str(name2[1])[2:-3])

            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()
            input_label = input_label.to(device).float()

            if args.downsample is not False:
                input_moving = downsample_image(input_moving, target_size=tuple(args.image_size))
                input_fixed = downsample_image(input_fixed, target_size=tuple(args.image_size))
                fixed_label = downsample_image(fixed_label, target_size=tuple(args.image_size))
                input_label = downsample_image(input_label, target_size=tuple(args.image_size))

            # get flow and warped image
            start = time.time()
            pred_flow = net(input_moving, input_fixed)
            pred_img = STN(input_moving, pred_flow)
            TIME.append(time.time() - start)
            pred_label = STN_label(input_label, pred_flow)

            save_image(pred_img, f_img, tmpName + '_warped.nii.gz')
            save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, tmpName + ".nii.gz")
            save_image(pred_label, f_img, tmpName + "_warped_label.nii.gz")
            del input_moving, input_label
            print('ok')
        print(np.mean(TIME))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
