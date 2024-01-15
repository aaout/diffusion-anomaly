import torch
import torch.nn
import numpy as np
import os
import os.path
import sys
import nibabel
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    subjet_id = "334"
    # slice_id = 0
    target_dir = "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_ValidationData/BraTS20_Validation_336/BraTS20_Training_336_t2.nii"

    diff_voxel = nibabel.load(f"{target_dir}")
    diff_voxel_tensor = diff_voxel.get_fdata()

    print(diff_voxel_tensor.shape)
    print("max: ", diff_voxel_tensor.max())
    diff_slice = diff_voxel_tensor[:, :, 80]
    fliplr_slice = np.fliplr(diff_slice)
    rotated_slice = np.rot90(fliplr_slice, k=1)

    plt.imshow(rotated_slice, cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(
        "BraTS20_Training_336_t2.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
