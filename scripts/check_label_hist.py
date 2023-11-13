import os
import sys

from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from PIL import Image
import nibabel

if __name__ == "__main__":
    NUMBER = "006"
    CHANNEL = "flair"
    # CHANNEL = "t1"
    # CHANNEL = "t1ce"
    # CHANNEL = "t2"
    # voxel_path = f"/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData/BraTS20_Training_{NUMBER}/BraTS20_Training_{NUMBER}_{CHANNEL}.nii"
    # voxel_path = "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_ValidationData/BraTS20_Validation_342/BraTS20_Training_342_seg.nii"
    # voxel_path = "/media/user/ボリューム/brats_imgs/test/BraTS20_Training_349/slice_099/BraTS20_Training_349_flair_099.nii.gz"
    # # voxel_path = "/mnt/ito/diffusion-anomaly/data/brats/test_labels/000247-label.nii.gz"
    # nib_voxel = nibabel.load(voxel_path)
    # np_voxel = nib_voxel.get_fdata()
    # # np_voxel = np_voxel[:, :, 80]
    # norm_image = cv2.normalize(
    #     np_voxel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    # )
    # norm_image = norm_image.astype(np.uint8)
    # thres_value, binarized_input_image = cv2.threshold(
    #     norm_image, 50, 255, cv2.THRESH_BINARY
    # )
    # padded_mask = np.zeros((256, 256), dtype=np.uint8)
    # padded_mask[8:-8, 8:-8] = norm_image

    padded_mask = cv2.imread(
        "/mnt/ito/diffusion-anomaly/out/sample_data_and_heatmap/349/099_input_label.png",
        cv2.IMREAD_GRAYSCALE,
    )

    # padded_mask = torch.load(
    #     "/mnt/ito/diffusion-anomaly/out/sample_data_and_heatmap/349/099_input_label.pt"
    # )
    # padded_mask = padded_mask.numpy()

    # ピクセル値のヒストグラムを表示
    plt.figure(figsize=(8, 6))
    plt.hist(padded_mask.ravel(), bins=100, color="b", alpha=0.7, rwidth=0.8)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.ylim(0, 2000)
    plt.show()
    plt.savefig("check_label_hist.jpg")
    plt.close()

    plt.imshow(padded_mask, cmap="gray")
    plt.axis("off")
    plt.savefig(
        "label_image.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
