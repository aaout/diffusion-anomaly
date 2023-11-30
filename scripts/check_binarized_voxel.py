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
    subjet_id = "365"
    slice_id = 0
    save_dir = f"/mnt/ito/diffusion-anomaly/out/binarized_img/{subjet_id}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    diff_voxel = nibabel.load(
        f"/media/user/ボリューム/out/bin_voxel_and_label/{subjet_id}/diff_voxel.nii"
    )
    diff_voxel_tensor = diff_voxel.get_fdata()
    print(diff_voxel_tensor.shape)
    print("max: ", diff_voxel_tensor.max())
    heatmap = plt.imshow(
        diff_voxel_tensor[slice_id, :, :],
        cmap="jet",
        interpolation="nearest",
        vmax=255,
        vmin=0,
    )
    plt.axis("off")
    plt.show()
    plt.savefig(
        f"{save_dir}/{str(slice_id+80).zfill(2)}_diffmap.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    otsubin_diff_voxel = nibabel.load(
        f"/media/user/ボリューム/out/bin_voxel_and_label/{subjet_id}/otsubin_diff_voxel.nii"
    )
    otsubin_diff_voxel_tensor = otsubin_diff_voxel.get_fdata()
    print(otsubin_diff_voxel_tensor.shape)
    print("max: ", otsubin_diff_voxel_tensor.max())
    plt.imshow(otsubin_diff_voxel_tensor[slice_id, :, :], cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(
        f"{save_dir}/{str(slice_id+80).zfill(2)}_diffmap_otsubin.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    bin_label_voxel = nibabel.load(
        f"/media/user/ボリューム/out/bin_voxel_and_label/{subjet_id}/bin_label_voxel.nii"
    )
    bin_label_voxel_tensor = bin_label_voxel.get_fdata()
    print(bin_label_voxel_tensor.shape)
    print("max: ", bin_label_voxel_tensor.max())
    plt.imshow(bin_label_voxel_tensor[slice_id, :, :], cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(
        f"{save_dir}/{str(slice_id+80).zfill(2)}_label_bin.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
