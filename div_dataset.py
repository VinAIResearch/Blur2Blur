import os
import shutil
from os.path import join as osp

from tqdm import tqdm


dataset = "REDS"
res_path = "/home/dangpb1/Research/CamSpecDeblurring/blur2blur/results/b2b_reds-gopro_1Dnet_286crop256_onlyREDS_perc/train_deb_100"

if dataset == "REDS":
    train_path = "/home/dangpb1/Research/datasets/REDS/train/train_blur_deblur"
    val_path = "/home/dangpb1/Research/datasets/REDS/val/val_blur_deblur"

elif dataset == "RSBlur":
    train_path = ""
    val_path = ""

train_list = sorted(os.listdir(train_path))
val_list = sorted(os.listdir(val_path))

os.makedirs(osp(res_path, "train_blur_deblur"), exist_ok=True)
os.makedirs(osp(res_path, "val_blur_deblur"), exist_ok=True)

for each in tqdm(train_list):
    shutil.copyfile(osp(res_path, "images", each), osp(res_path, "train_blur_deblur", each))

for each in tqdm(val_list):
    shutil.copyfile(osp(res_path, "images", each), osp(res_path, "val_blur_deblur", each))
