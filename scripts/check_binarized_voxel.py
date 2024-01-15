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
    subjet_id = "338"
    # slice_id = 0
    target_dir = f"/media/user/ボリューム/out/bin_voxel_and_label_080-128_nonclassifier_ddim/{subjet_id}"
    save_dir = f"/mnt/ito/diffusion-anomaly/out/binarized_img_080-128_nonclassifier_ddim/{subjet_id}"
    # target_dir = f"/media/user/ボリューム/out/bin_voxel_and_label/{subjet_id}"
    # save_dir = f"/mnt/ito/diffusion-anomaly/out/binarized_img/{subjet_id}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    # TODO: voxelのヒストグラムを確認し、Otsuの手法を適応するかしないかを決める
    diff_voxel = nibabel.load(f"{target_dir}/diff_voxel.nii")
    diff_voxel_tensor = diff_voxel.get_fdata()
    bin_label_voxel = nibabel.load(f"{target_dir}/bin_label_voxel.nii")
    bin_label_voxel_tensor = bin_label_voxel.get_fdata()
    otsubin_diff_voxel = nibabel.load(f"{target_dir}/otsubin_diff_voxel.nii")
    otsubin_diff_voxel_tensor = otsubin_diff_voxel.get_fdata()

    print(diff_voxel_tensor.shape)
    print("max: ", diff_voxel_tensor.max())
    for slice_id in range(diff_voxel_tensor.shape[0]):
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
            f"{save_dir}/{str(slice_id).zfill(3)}_diffmap.png",
            # f"{save_dir}/{str(slice_id+80).zfill(2)}_diffmap.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        plt.imshow(otsubin_diff_voxel_tensor[slice_id, :, :], cmap="gray")
        plt.axis("off")
        plt.show()
        plt.savefig(
            f"{save_dir}/{str(slice_id).zfill(3)}_diffmap_otsubin.png",
            # f"{save_dir}/{str(slice_id+80).zfill(2)}_diffmap_otsubin.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        plt.imshow(bin_label_voxel_tensor[slice_id, :, :], cmap="gray")
        plt.axis("off")
        plt.show()
        plt.savefig(
            f"{save_dir}/{str(slice_id).zfill(3)}_label_bin.png",
            # f"{save_dir}/{str(slice_id+80).zfill(2)}_label_bin.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
