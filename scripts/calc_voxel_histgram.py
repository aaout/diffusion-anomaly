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
    """
    4チャネルの入出力差分データのヒストグラムを表示
    """
    filename_list = []
    SUBJETC_ID = "006"
    diffmap_dirs = (
        "/media/user/ボリューム/out/sample_data_and_heatmap_080-154_train_abnormal/"
    )
    save_dir = "/media/user/ボリューム/out/bin_voxel_and_label_080-154_train_abnormal"

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
            norm_diff_all_voxel_np = cv2.normalize(
                diff_all_voxel_np,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            )
            norm_diff_all_voxel_np = norm_diff_all_voxel_np.astype(np.uint8)

            # ピクセル値のヒストグラムを表示
            os.makedirs(save_dir, exist_ok=True)
            plt.figure(figsize=(8, 6))
            plt.hist(
                norm_diff_all_voxel_np.ravel(),
                bins=256,
                color="b",
                alpha=0.7,
                rwidth=0.8,
            )
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
            plt.savefig(f"{save_dir}diff_all_voxel_hist_{SUBJETC_ID}.jpg")
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.hist(
                norm_diff_all_voxel_np.ravel(),
                bins=256,
                color="b",
                alpha=0.7,
                rwidth=0.8,
            )
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            # plt.xlim(-0.05, 2)
            plt.ylim(0, 1000)
            plt.tight_layout()
            plt.show()
            plt.savefig(f"{save_dir}diff_all_voxel_hist_{SUBJETC_ID}_ylim.jpg")
            plt.close()
            sys.exit()
