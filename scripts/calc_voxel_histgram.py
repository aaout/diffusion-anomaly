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
    voxel_path = f"/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData/BraTS20_Training_{NUMBER}/BraTS20_Training_{NUMBER}_{CHANNEL}.nii"
    nib_voxel = nibabel.load(voxel_path)
    np_voxel = nib_voxel.get_fdata()

    # 各スライスの背景を除くピクセル値の合計を格納するリスト
    total_pixel_values = []

    for slice_number in range(np_voxel.shape[2]):
        slice_data = np_voxel[:, :, slice_number]
        pixel_values = slice_data.ravel()
        non_zero_pixel_values = pixel_values[pixel_values != 0]
        total_pixel_values.extend(non_zero_pixel_values)  # ヒストグラム用の変数に追加

    # ピクセル値の最大、最小、クリップ点が上位何%にあるかを表示
    # clip_upper = 378
    # clip_lower = 60
    # 上位1%と下位1%の位置を計算
    sorted_lists = sorted(total_pixel_values)
    total_count = len(sorted_lists)
    top_1_percent_index = int(0.99 * total_count)
    bottom_1_percent_index = int(0.01 * total_count)
    # 上位1%の数値を取得
    top_1_percent_values = sorted_lists[top_1_percent_index]
    # 下位1%の数値を取得
    bottom_1_percent_values = sorted_lists[bottom_1_percent_index]

    # 結果を表示
    print(f"上位1%の数値とインデックス: {top_1_percent_values}")

    print(f"下位1%の数値とインデックス: {bottom_1_percent_values}")
    # upper_position = sorted_lists.index(clip_upper) + 1
    # lower_position = sorted_lists.index(clip_lower) + 1
    # upper_percentile = (upper_position / total_count) * 100
    # lower_percentile = (lower_position / total_count) * 100
    # print(f"{NUMBER}_{CHANNEL}")
    # print("sum of pixels: ", total_count)
    # print(f"max: {sorted_lists[-1]}")
    # print(f"min: {sorted_lists[0]}")
    # print(f"upper_percentile: {clip_upper} {upper_percentile:.4f}%")
    # print(f"lower_percentile: {clip_lower} {lower_percentile:.4f}%")

    # ピクセル値のヒストグラムを表示
    # plt.figure(figsize=(8, 6))
    # plt.hist(total_pixel_values, bins=100, color='b', alpha=0.7, rwidth=0.8)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()
    # os.makedirs("/mnt/ito/diffusion-anomaly/out/hist/", exist_ok=True)
    # plt.savefig(f"/mnt/ito/diffusion-anomaly/out/hist/BraTS20_Training_{NUMBER}_{CHANNEL}.jpg")
