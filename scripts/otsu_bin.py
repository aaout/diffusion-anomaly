import os
import sys
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from skimage.filters import threshold_otsu
import nibabel as nib


# DICE係数の計算
def dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient between two binary arrays.

    Args:
        y_true (numpy.ndarray): Ground truth binary array.
        y_pred (numpy.ndarray): Predicted binary array.

    Returns:
        float: Dice coefficient between y_true and y_pred.
    """
    intersection = np.sum(y_true & y_pred)
    return (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)


if __name__ == "__main__":
    dice_list = []
    auroc_list = []
    filename_list = []
    thresh_value_list = []
    # diffmap_dirs = "/media/user/ボリューム/out/nonclassifier_sample_data_and_heatmap_080-128"
    # save_dir = "/media/user/ボリューム/out/bin_voxel_and_label_080-128_nonclassifier"
    diffmap_dirs = "/media/user/ボリューム/out/sample_data_and_heatmap"
    save_dir = "/media/user/ボリューム/out/bin_voxel_and_label"

    subject_dirs = sorted(os.listdir(diffmap_dirs))
    for subject_id in subject_dirs:
        if subject_id != "334":
            continue
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

            # ラベルデータのファイル名を取得してnumpy形式に変換
            label_pt = [f for f in target_file_list if f.endswith("label.pt")]
            label_pt_path = os.path.join(slice_path, label_pt[0])
            label_pt_data = torch.load(label_pt_path)
            label_np_data = label_pt_data.cpu().numpy()
            norm_label_np = cv2.normalize(
                label_np_data,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            )
            norm_label_np = norm_label_np.astype(np.uint8)
            label_thres_value, bin_norm_label = cv2.threshold(
                norm_label_np, 50, 255, cv2.THRESH_BINARY
            )
            label_voxel_list.append(torch.tensor(bin_norm_label))

        diff_all_voxel = torch.stack(diff_all_voxel_list, dim=0)
        diff_all_voxel_np = diff_all_voxel.cpu().numpy()
        norm_diff_all_np = cv2.normalize(
            diff_all_voxel_np,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )
        norm_diff_all_np = norm_diff_all_np.astype(np.uint8)
        # plt.imshow(norm_diff_all_np[0, :, :], cmap="gray")
        heatmap = plt.imshow(
            norm_diff_all_np[0, :, :],
            cmap="jet",
            interpolation="nearest",
            # vmax=1,
            # vmin=-1,
        )
        plt.colorbar(heatmap)
        plt.axis("off")
        plt.savefig(
            "diff_heatmap.jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        thresh_value = 5
        thres, bin_diff_all = cv2.threshold(
            norm_diff_all_np, thresh_value, 255, cv2.THRESH_BINARY
        )
        print(f"Otsu threshold: {thres}")
        thresh_value_list.append(thres)

        # 二値化したヒートマップをNIfTI形式で保存
        plt.imshow(bin_diff_all[0, :, :], cmap="gray")
        plt.axis("off")
        plt.savefig(
            f"bin{thresh_value}.jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
