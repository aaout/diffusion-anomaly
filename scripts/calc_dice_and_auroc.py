import os
import sys
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from skimage.filters import threshold_otsu


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
    diffmap_dirs = "/media/user/ボリューム/out/sample_data_and_heatmap/"

    subject_dirs = sorted(os.listdir(diffmap_dirs))
    for subject_id in subject_dirs:
        if subject_id == "341":
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
        # Otsu thresholdingにより二値化
        diff_all_thresh = threshold_otsu(norm_diff_all_np)
        _, bin_diff_all = cv2.threshold(
            norm_diff_all_np, diff_all_thresh, 255, cv2.THRESH_BINARY
        )
        thresh_value_list.append(diff_all_thresh.astype(np.int64))

        label_voxel = torch.stack(label_voxel_list, dim=0)
        label_voxel_np = label_voxel.cpu().numpy()
        norm_label_voxel_np = cv2.normalize(
            label_voxel_np,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )
        norm_label_voxel_np = norm_label_voxel_np.astype(np.uint8)

        # ピクセル値のヒストグラムを表示
        # plt.figure(figsize=(8, 6))
        # plt.hist(
        #     bin_diff_all.ravel(),
        #     bins=100,
        #     color="b",
        #     alpha=0.7,
        #     rwidth=0.8,
        # )
        # plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.tight_layout()
        # plt.ylim(0, 2000)
        # plt.show()
        # plt.savefig("diff_all_voxel_hist.jpg")
        # plt.close()

        # # ピクセル値のヒストグラムを表示
        # plt.figure(figsize=(8, 6))
        # plt.hist(
        #     norm_label_voxel_np.ravel(),
        #     bins=100,
        #     color="b",
        #     alpha=0.7,
        #     rwidth=0.8,
        # )
        # plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.tight_layout()
        # plt.ylim(0, 2000)
        # plt.show()
        # plt.savefig("label_voxel_list_hist.jpg")
        # plt.close()

        # ラベル画像と二値化された画像のDICE係数を計算
        dice = dice_coefficient(norm_label_voxel_np, bin_diff_all)
        dice_list.append(dice)
        print(f"DICE: {dice}")

        # AUROCの計算
        flat_truth = norm_label_voxel_np.flatten()
        flat_pred = bin_diff_all.flatten()
        auroc = roc_auc_score(flat_truth, flat_pred)
        auroc_list.append(auroc)
        print(f"AUROC: {auroc}")
        print("")

    # # DICEとAUROCの値をJSON形式で保存
    dice_and_auroc_json = {}
    dice_avg = np.mean(dice_list)
    auroc_avg = np.mean(auroc_list)
    print(f"DICE average: {dice_avg}")
    print(f"AUROC average: {auroc_avg}")
    dice_and_auroc_json["average"] = {"dice": dice_avg, "auroc": auroc_avg}
    for fname, dice, auroc in zip(filename_list, dice_list, auroc_list):
        dice_and_auroc_json[fname] = {
            "dice": dice,
            "auroc": auroc,
        }

    with open("/mnt/ito/diffusion-anomaly/out/dice_and_auroc.json", "w") as f:
        json.dump(dice_and_auroc_json, f, indent=4)
