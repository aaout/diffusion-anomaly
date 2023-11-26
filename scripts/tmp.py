import torch
import torch.nn
import numpy as np
import os
import os.path
import sys
import nibabel
from scipy import ndimage
import matplotlib.pyplot as plt


if __name__ == "__main__":
    diff_voxel = nibabel.load(
        "/media/user/ボリューム/out/bin_voxel_and_label/334/diff_voxel.nii"
    )
    diff_voxel_tensor = diff_voxel.get_fdata()
    print(diff_voxel_tensor.shape)
    print("max: ", diff_voxel_tensor.max())
    plt.imshow(diff_voxel_tensor[10, :, :])
    plt.show()
    plt.savefig("diff_voxel.png")
    plt.close()

    otsubin_diff_voxel = nibabel.load(
        "/media/user/ボリューム/out/bin_voxel_and_label/334/otsubin_diff_voxel.nii"
    )
    otsubin_diff_voxel_tensor = otsubin_diff_voxel.get_fdata()
    print(otsubin_diff_voxel_tensor.shape)
    print("max: ", otsubin_diff_voxel_tensor.max())
    plt.imshow(otsubin_diff_voxel_tensor[10, :, :])
    plt.show()
    plt.savefig("otsubin_diff_voxel_tensor.png")
    plt.close()

    bin_label_voxel = nibabel.load(
        "/media/user/ボリューム/out/bin_voxel_and_label/334/bin_label_voxel.nii"
    )
    bin_label_voxel_tensor = bin_label_voxel.get_fdata()
    print(bin_label_voxel_tensor.shape)
    print("max: ", bin_label_voxel_tensor.max())
    plt.imshow(bin_label_voxel_tensor[10, :, :])
    plt.show()
    plt.savefig("bin_label_voxel_tensor.png")
    plt.close()
