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
    filename_list = []
    SUBJETC_ID = "341"
    diffmap_dirs = "/media/user/ボリューム/out/sample_data_and_heatmap_000-154/"
    save_dir = "/mnt/ito/diffusion-anomaly/out/diffmap_histgram_000-154/"

    subject_dirs = sorted(os.listdir(diffmap_dirs))
    for subject_id in subject_dirs:
        if subject_id == SUBJETC_ID:
            diff_all_voxel_list = []
            label_voxel_list = []
            subject_path = os.path.join(diffmap_dirs, subject_id)
            slice_list = sorted(os.listdir(subject_path))
            print(f"ID: {subject_id}")
            filename_list.append(subject_id)
            for slice_id in slice_list:
                slice_path = os.path.join(subject_path, slice_id)
                target_file_list = sorted(os.listdir(slice_path))
                # 4チャネル差分合計のファイル名を取得してnumpy形式に変換
                diff_all_pt = [f for f in target_file_list if f.endswith("diff_all.pt")]
                diff_all_pt_path = os.path.join(slice_path, diff_all_pt[0])
                diff_all_pt_data = torch.load(diff_all_pt_path)
                diff_all_voxel_list.append(diff_all_pt_data)
            diff_all_voxel = torch.stack(diff_all_voxel_list, dim=0)
            diff_all_voxel_np = diff_all_voxel.cpu().numpy()

            # ピクセル値のヒストグラムを表示
            os.makedirs(save_dir, exist_ok=True)
            plt.figure(figsize=(8, 6))
            plt.hist(
                diff_all_voxel_np.ravel(), bins=100, color="b", alpha=0.7, rwidth=0.8
            )
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
            plt.savefig(f"{save_dir}diff_all_voxel_hist_{SUBJETC_ID}.jpg")
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.hist(
                diff_all_voxel_np.ravel(), bins=100, color="b", alpha=0.7, rwidth=0.8
            )
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            # plt.xlim(-0.05, 2)
            plt.ylim(0, 5000)
            plt.tight_layout()
            plt.show()
            plt.savefig(f"{save_dir}diff_all_voxel_hist_{SUBJETC_ID}_ylim.jpg")
            plt.close()
            sys.exit()
    # os.makedirs("/mnt/ito/diffusion-anomaly/out/hist/", exist_ok=True)
    # plt.savefig(
    #     f"/mnt/ito/diffusion-anomaly/out/hist/BraTS20_Training_{NUMBER}_{CHANNEL}.jpg"
    # )

    # NUMBER = "006"
    # CHANNEL = "flair"
    # # CHANNEL = "t1"
    # # CHANNEL = "t1ce"
    # # CHANNEL = "t2"
    # voxel_path = f"/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData/BraTS20_Training_{NUMBER}/BraTS20_Training_{NUMBER}_{CHANNEL}.nii"
    # nib_voxel = nibabel.load(voxel_path)
    # np_voxel = nib_voxel.get_fdata()

    # # 各スライスの背景を除くピクセル値の合計を格納するリスト
    # total_pixel_values = []

    # for slice_number in range(np_voxel.shape[2]):
    #     slice_data = np_voxel[:, :, slice_number]
    #     pixel_values = slice_data.ravel()
    #     non_zero_pixel_values = pixel_values[pixel_values != 0]
    #     total_pixel_values.extend(non_zero_pixel_values)  # ヒストグラム用の変数に追加

    # # ピクセル値の最大、最小、クリップ点が上位何%にあるかを表示
    # # clip_upper = 378
    # # clip_lower = 60
    # # 上位1%と下位1%の位置を計算
    # sorted_lists = sorted(total_pixel_values)
    # total_count = len(sorted_lists)
    # top_1_percent_index = int(0.99 * total_count)
    # bottom_1_percent_index = int(0.01 * total_count)
    # # 上位1%の数値を取得
    # top_1_percent_values = sorted_lists[top_1_percent_index]
    # # 下位1%の数値を取得
    # bottom_1_percent_values = sorted_lists[bottom_1_percent_index]

    # # 結果を表示
    # print(f"上位1%の数値とインデックス: {top_1_percent_values}")

    # print(f"下位1%の数値とインデックス: {bottom_1_percent_values}")
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
