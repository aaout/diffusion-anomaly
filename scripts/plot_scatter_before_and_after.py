import os
import sys
from PIL import Image
import nibabel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# CHANNEL = "flair"
# CHANNEL = "t1"
# CHANNEL = "t1ce"
CHANNEL = "flair"
SLICE = 80

# load sample image
sample_img_path = f"/media/user/ボリューム2/brats_imgs/train/BraTS20_Training_006/slice_081/BraTS20_Training_006_081_seg.nii.gz"
# sample_img_path = f"/mnt/ito/diffusion-anomaly/data/brats/testing/000247/brats_train_006_{CHANNEL}_081_w.nii.gz"
nib_slice = nibabel.load(sample_img_path)
original_nib_img_tensor = nib_slice.get_fdata()

# load BraTS voxel
# voxel_path = f"/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_{CHANNEL}.nii.gz"
# # voxel_path = f"/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_006/BraTS20_Training_006_{CHANNEL}.nii"
# nib_voxel = nibabel.load(voxel_path)
# nib_voxel_tensor = nib_voxel.get_fdata()
# pad_voxel_slice = nib_voxel_tensor[:, :, SLICE]
# # pad_voxel_slice = nib_voxel_tensor[:, :, 81]
# flipped_image_data = np.fliplr(pad_voxel_slice)
# rotated_image_data = np.rot90(flipped_image_data, k=1)
print("max: ", np.max(original_nib_img_tensor))
print("min: ", np.min(original_nib_img_tensor))
# print("")
# print("max: ", np.max(rotated_image_data))
# print("min: ", np.min(rotated_image_data))

plt.imshow(original_nib_img_tensor, cmap='gray')
plt.show()
plt.savefig("original_nib_img_seg.png")
sys.exit()

# 同じ座標のピクセル値を取得してリストに格納する
pixel_values_original = []
pixel_values_created = []

for y in range(nib_slice.shape[0]):
    for x in range(nib_slice.shape[1]):
        pixel_original = original_nib_img_tensor[x, y]
        pixel_created = rotated_image_data[x, y]
        pixel_values_original.append(pixel_original)
        pixel_values_created.append(pixel_created)

# 散布図をプロットする
# plt.figure(figsize=(8, 8))
# plt.scatter(pixel_values_created, pixel_values_original, marker='.')
# plt.show()
# plt.savefig("scatter.png")

scatter_data_to_excel = {'original': pixel_values_original, 'created': pixel_values_created}
df = pd.DataFrame(scatter_data_to_excel)
excel_filename = f'BraTS20_Training_001_{CHANNEL}_{SLICE}.xlsx'
# excel_filename = f'BraTS20_Training_006_{CHANNEL}_81w.xlsx'
df.to_excel(excel_filename, index=False, engine='openpyxl')
