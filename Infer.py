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
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join('./Result', name))


def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
            63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
            163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def train():

    # 创建需要的文件夹并指定gpu
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    #get image size
    img_name = os.listdir(args.train_dir)[0]
    t_img = sitk.ReadImage(args.train_dir+ "\\" + img_name)
    t_arr = sitk.GetArrayFromImage(t_img)
    t_arr = np.pad(t_arr, ((16, 16), (0, 0), (16, 16)), 'constant', constant_values=0)
    t_arr_test = t_arr[::2, ::2, ::2]
    vol_size = t_arr_test[:, :, :].shape
    print(vol_size)

    # 创建配准网络（net）和STN
    net = TransMatch(args).to(device)
    best_model = torch.load('./experiments/1212firstrunorigincode/ams_mrezadsc0.0593epoch021.pth.tar')['state_dict']
    net.load_state_dict(best_model)
    #C:\Users\Matej\Documents\GitHub\AMS_izziv\experiments\1212firstrunorigincode\dsc0.6351epoch012.pth.tar


    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    # UNet.train()
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

            #retarded fix
            #print(str(name1[1])[2:-10])
            tmpName = str(name2[1])[2:-10]
            #f_img = sitk.ReadImage(args.train_dir + "\\" + str(name1[0])[2:-3] + "\\" + str(name1[1])[2:-3])
            f_img = sitk.ReadImage(args.train_dir + "\\" + str(name2[1])[2:-3])
            print(args.train_dir + "\\" + str(name2[1])[2:-3])

            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()
            input_label = input_label.to(device).float()

            # 获得配准后的图像和label
            start = time.time()
            pred_flow = net(input_moving, input_fixed)
            pred_img = STN(input_moving, pred_flow)
            TIME.append(time.time() - start)
            pred_label = STN_label(input_label, pred_flow)

            #tmpName = name1  # please check the tmpName when you run by yourself
            # print(tmpName)
            save_image(pred_img, f_img, tmpName + '_warpped.nii.gz')
            save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, tmpName + "_flow.nii.gz")
            save_image(pred_label, f_img, tmpName + "_label.nii.gz")
            del input_moving, input_label
            print('ok')
        print(np.mean(TIME))


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    model_lists = natsorted(glob.glob(save_dir+  '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
    torch.save(state, save_dir+filename)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
