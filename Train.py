# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from utils import losses
from utils.config import args
from utils.datagenerators_atlas import AMS_Dataset, AMS_Dataset_val
from Models.STN import SpatialTransformer
from natsort import natsorted
from utils.helper_functions import *

import matplotlib.pyplot as plt
import wandb

warnings.filterwarnings("ignore")

from Models.TransMatch import TransMatch

wandb.init(
    # set the wandb project where this run will be logged
    project="AMS_iziv",

    # track hyperparameters and run metadata
    config={
    "architecture": "TransMatch",
    "dataset": "ThoraxCBCT",
    "epochs": args.n_iter,
    "learning_rate": args.lr,
    "image_size": args.image_size,

    }
)

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


def compute_label_dice(gt, pred):
    # list of classes to calculate
    cls_lst = [1,2,3,4,5,6,7,8,10,11]
    # cls_lst = [182]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)



def train():
    make_dirs()
    life = 42
    torch.manual_seed(life)
    #torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())

    # configure net
    net = TransMatch(args).to(device)

    iterEpoch = 1

    STN = SpatialTransformer(args.image_size).to(device)
    STN_label = SpatialTransformer(args.image_size, mode="nearest").to(device)
    net.train()
    STN.train()

    opt = Adam(net.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # Get all the names of the training data
    #train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = AMS_Dataset(json_path=args.dataset_cfg)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    DSV = AMS_Dataset_val(json_path=args.dataset_cfg)
    print("Number of validating images: ", len(DSV))
    DLV = Data.DataLoader(DSV, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    a = 0


    

    # Working loop
    for i in range(iterEpoch, args.n_iter + 1):
        
        # Generate the moving images and convert them to tensors.
        net.train()
        STN.train()
        print('epoch:', i)
        input_images_all = iter(DL)
        input_validation_all = iter(DLV)
        pair = 0

        
        # Training Loop
        for input_fixed, input_moving, fixed_name, moving_name in input_images_all:
            
            # To cuda gpu
            #print("fixed: ", fixed_name, "moving: ", moving_name)
            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()

            # downsample if needed
            if args.downsample is not False:
                input_moving = downsample_image(input_moving, target_size=tuple(args.image_size))
                input_fixed = downsample_image(input_fixed, target_size=tuple(args.image_size))

            # Run the data through the model to produce warp and flow field
            flow_m2f = net(input_fixed, input_moving)
            m2f = STN(input_fixed, flow_m2f)

            # Calculate loss
            sim_loss = sim_loss_fn(m2f, input_moving)
            grad_loss = grad_loss_fn(flow_m2f)
            # zero_loss = zero_loss_fn(flow_m2f, zero)
            loss = sim_loss + args.alpha * grad_loss #  + zero_loss

            # Backwards and optimise
            opt.zero_grad()
            loss.backward()
            opt.step()

            # inverse fixed image and moving image
            flow_m2f = net(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)

            # Calculate loss
            sim_loss = sim_loss_fn(m2f, input_fixed)
            grad_loss = grad_loss_fn(flow_m2f)
            # zero_loss = zero_loss_fn(flow_m2f, zero)
            loss = sim_loss + args.alpha * grad_loss #  + zero_loss
            
            wandb.log({"sim_loss": sim_loss, "grad_loss": grad_loss, "loss": loss})

            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            pair += 1

            
        # switch to evaluation mode
        net.eval()
        STN.eval()
        STN_label.eval()

        

        DSC = []
        # Validation Loop
        for input_fixed, input_moving, moving_label, fixed_label, name1, name2 in input_validation_all:
            
            # To cuda gpu
            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()
            fixed_label = fixed_label.to(device).float()
            moving_label = moving_label.to(device).float()

            if args.downsample is not False:
                input_moving = downsample_image(input_moving, target_size=tuple(args.image_size))
                input_fixed = downsample_image(input_fixed, target_size=tuple(args.image_size))
                fixed_label = downsample_image(fixed_label, target_size=tuple(args.image_size))
                moving_label = downsample_image(moving_label, target_size=tuple(args.image_size))

            # Run the data through the model to produce warp and flow field
            pred_flow = net(input_fixed, input_moving)
            pred_img = STN(input_fixed, pred_flow)
            pred_label = STN_label(fixed_label, pred_flow)
            
            # Partial results visualization - FOR DEBUGGING
            if (i % 10 == 1) & (a == False) & (args.partial_results == True):

                pred_flow_1 = net(input_moving, input_fixed)
                pred_img_1 = STN(input_moving, pred_flow_1)
                pred_label_1 = STN_label(moving_label, pred_flow)
                a = True
                plt.figure(1)
                plt.subplot(2, 2, 1)
                plt.imshow(input_fixed[0, 0, :, :, 50].cpu().detach().numpy(), cmap='gray')
                plt.subplot(2, 2, 2)
                plt.imshow(pred_img_1[0, 0, :, :, 50].cpu().detach().numpy(), cmap='gray')
                plt.subplot(2, 2, 3)
                plt.imshow(pred_flow_1[0, 0, :, :, 50].cpu().detach().numpy(), cmap='gray')
                plt.subplot(2, 2, 4)
                plt.imshow(pred_label_1[0, 0, :, :, 50].cpu().detach().numpy(), cmap='gray')
                plt.show()
                del pred_flow_1, pred_img_1, pred_label_1


            dice = compute_label_dice(moving_label[0, 0, ...].cpu().detach().numpy(), pred_label[0, 0, ...].cpu().detach().numpy())
            #print("{0}" .format(dice))
            DSC.append(dice)

            del pred_flow, pred_img, pred_label, input_moving
        a = False

        print(np.mean(DSC), np.std(DSC))
        wandb.log({"mean_DSC": np.mean(DSC), "std_DSC": np.std(DSC)})

        if i % args.n_save_iter == 0:
            save_checkpoint({
                'epoch': i+1,
                'state_dict': net.state_dict(),
                'optimizer': opt.state_dict(),
                }, save_dir=args.model_dir, filename='dsc{:.4f}epoch{:0>3d}.pth.tar'.format(np.mean(DSC), i+1))
            print("Model saved")

    #f.close()
    wandb.finish()

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    #model_lists = natsorted(glob.glob(save_dir+  '*'))
    #while len(model_lists) > max_model_num:
    #    os.remove(model_lists[0])
    #    model_lists = natsorted(glob.glob(save_dir + '*'))
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    torch.save(state, save_dir+filename) 

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
