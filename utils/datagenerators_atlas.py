import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''

x_offset = 30
y_offset = 30
z_offset = 30
x_size = 96
y_size = 96
z_size = 96

class Dataset(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        t_img = sitk.ReadImage(self.files[index])

        cropper = sitk.CropImageFilter()
        cropper.SetLowerBoundaryCropSize([0 + x_offset, 0 + y_offset, 0 + z_offset])
        cropper.SetUpperBoundaryCropSize([160 - x_size - x_offset, 192 - y_size - y_offset, 160 - z_size - z_offset])
        t_img = cropper.Execute(t_img)

        img_arr = sitk.GetArrayFromImage(t_img)[np.newaxis, ...]
        index = self.files[index][59:61]
        # 返回值自动转换为torch的tensor类型
        return img_arr, index
