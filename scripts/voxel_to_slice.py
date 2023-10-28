import os
import sys
from tqdm import tqdm
import shutil
import nibabel
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import hflip, vflip, rotate

if __name__ == "__main__":
    slice_path = "/mnt/ito/diffusion-anomaly/data/brats/training/000001/brats_train_001_flair_080_w.nii.gz"
    nib_slice = nibabel.load(slice_path)
    original_nib_img_tensor = torch.tensor(nib_slice.get_fdata())
    print("shape: ", original_nib_img_tensor.shape)
    print("max: ", torch.max(original_nib_img_tensor))
    print("min: ", torch.min(original_nib_img_tensor))
    print("")
    max_val = torch.max(original_nib_img_tensor)
    min_val = torch.min(original_nib_img_tensor)
    # original_nib_img_tensor = (original_nib_img_tensor - min_val) / (max_val - min_val)
    save_image(original_nib_img_tensor, "original_brats_train_001_flair_080_w.png")

    voxel_path = "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii"
    nib_voxel = nibabel.load(voxel_path)
    nib_voxel_tensor = torch.tensor(nib_voxel.get_fdata())
    print("shape: ", nib_voxel_tensor.shape)

    # 首側の80枚と頭頂側の26枚を除去
    pad_voxel = nib_voxel_tensor[:, :, 80:-26]
    # pad_voxel = nib_voxel_tensor[:, :, 26:-80]
    nib_voxel_to_slice = nib_voxel_tensor[:, :, 80]
    nib_voxel_to_slice = hflip(nib_voxel_to_slice)
    nib_voxel_to_slice = torch.rot90(nib_voxel_to_slice, 1, [0, 1])

    print("max: ", torch.max(nib_voxel_to_slice))
    print("min: ", torch.min(nib_voxel_to_slice))
    max_val = torch.max(nib_voxel_to_slice)
    min_val = torch.min(nib_voxel_to_slice)

    gray_without_min = nib_voxel_to_slice[nib_voxel_to_slice != min_val]
    second_min_val = torch.min(gray_without_min)
    np_voxel_to_slice = nib_voxel_to_slice.numpy()
    np_second_min_val = second_min_val.numpy()
    second_min_indices = np.where(np_voxel_to_slice == np_second_min_val)
    print(f"second_min_val: {second_min_val}")
    print(f"coordinate: {second_min_indices}")

    # created_normalized_slice = nib_img_tensor_slice
    created_normalized_slice = (nib_voxel_to_slice - second_min_val) / (
        max_val - second_min_val
    )
    print("max: ", torch.max(created_normalized_slice))
    print("min: ", torch.min(created_normalized_slice))
    # created_normalized_slice[44, 133] = torch.tensor([255])
    save_image(created_normalized_slice, "created_brats_train_001_flair_080_w.png")
    sys.exit()

    if torch.all(torch.eq(original_nib_img_tensor, created_normalized_slice)):
        print("SAME")
    else:
        print("NOT SAME")

    # with tqdm(os.walk(DATA_DIR)) as pbar:
    #     for root, dirs, files in pbar:
    #         print(root)
    #         print(dirs)
    #         print(files)
    #         # number = dirs[0].split("_")[-1]
    #         # print(number)
    #         sys.exit()
    # if not dirs:
    #     files.sort()
    #     files[0], files[1] = files[1], files[0]  # flair.nii.gzをfilesリストの先頭に
    #     datapoint = dict()
    #     # extract all files as channels
    #     concat_data = []
    #     for f in files:
    #         # seqtype = f.split("_")[3].split(".")[0]
    #         # 各Nifti画像を読み込んで結合
    #         data_path = os.path.join(root, f)
    #         nib_img = nibabel.load(data_path)
    #         nib_img_tensor = torch.tensor(nib_img.get_fdata())
    #         padding_img = torch.zeros(256, 256, 155)
    #         padding_img[8:-8, 8:-8, :] = nib_img_tensor  # padding
    #         padding_img = padding_img[:, :, 50:-26]  # 上26枚と下80枚を除去
    #         # データの正規化
    #         # for slice in range(padding_img.shape[2]):
    #         #     min_val = padding_img[:, :, slice].min()
    #         #     max_val = padding_img[:, :, slice].max()
    #         #     print("min: ", min_val, " max: ", max_val)
    #         #     normalized_slice = (padding_img[:, :, slice] - min_val) / (max_val - min_val + 1e-8)
    #         #     padding_img[:, :, slice] = normalized_slice
    #         concat_data.append(padding_img)
    #     concat_data = torch.stack(concat_data)
    #     with tqdm(range(concat_data.shape[3]), leave=False) as pbar2:
    #         for slice in pbar2:
    #             # print("slice_data: ", concat_data[:, :, :, slice].shape)
    #             # print(f.split('.')[0][0:20])
    #             pbar2.set_description(f"{f.split('.')[0][0:20]}")
    #             os.makedirs(
    #                 f"/media/user/ボリューム/brats_imgs/train_nonormalize/{f.split('.')[0][0:20]}/",
    #                 exist_ok=True,
    #             )
    #             torch.save(
    #                 concat_data[:, :, :, slice],
    #                 f"/media/user/ボリューム/brats_imgs/train_nonormalize/{f.split('.')[0][0:20]}/slice_{str(slice).zfill(3)}.pt",
    #             )
