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
    diffmap_dirs = "/media/user/ボリューム/out/sample_data_and_heatmap"
    save_dir = "/media/user/ボリューム/out/bin_voxel_and_label"
    # diffmap_dirs = "/media/user/ボリューム/out/sample_data_and_heatmap_000-154"
    # save_dir = "/media/user/ボリューム/out/diffmap_and_bin_voxel_000-154"

    subject_dirs = sorted(os.listdir(diffmap_dirs))
    for subject_id in subject_dirs:
        if subject_id != "341":
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

        # # TODO: ヒートマップ出力を別ファイルに分割
        # # ヒートマップをNIfTI形式で保存
        # os.makedirs(f"{save_dir}/{subject_id}", exist_ok=True)
        # norm_diff_all_nii = nib.Nifti1Image(norm_diff_all_np, affine=np.eye(4))
        # norm_diff_all_nii.to_filename(f"{save_dir}/{subject_id}/diff_voxel.nii")
        # norm_diff_all_np = norm_diff_all_np.astype(np.uint8)

        # # Otsu thresholdingにより二値化
        diff_all_thresh = threshold_otsu(norm_diff_all_np)
        thres, bin_diff_all = cv2.threshold(
            norm_diff_all_np, diff_all_thresh, 255, cv2.THRESH_BINARY
        )
        print(f"Otsu threshold: {thres/256}")

        # # 二値化したヒートマップをNIfTI形式で保存
        # bin_diff_all_nii = nib.Nifti1Image(bin_diff_all, affine=np.eye(4))
        # bin_diff_all_nii.to_filename(f"{save_dir}/{subject_id}/otsubin_diff_voxel.nii")
        # thresh_value_list.append(diff_all_thresh.astype(np.int64))

        # label_voxel = torch.stack(label_voxel_list, dim=0)
        # label_voxel_np = label_voxel.cpu().numpy()
        # norm_label_voxel_np = cv2.normalize(
        #     label_voxel_np,
        #     None,
        #     alpha=0,
        #     beta=255,
        #     norm_type=cv2.NORM_MINMAX,
        # )
        # # 二値化したラベルデータをNIfTI形式で保存
        # norm_label_voxel_nii = nib.Nifti1Image(norm_label_voxel_np, affine=np.eye(4))
        # norm_label_voxel_nii.to_filename(f"{save_dir}/{subject_id}/bin_label_voxel.nii")
        # norm_label_voxel_np = norm_label_voxel_np.astype(np.uint8)

        # # ラベル画像と二値化された画像のDICE係数を計算
        # dice = dice_coefficient(norm_label_voxel_np, bin_diff_all)
        # dice_list.append(dice)
        # print(f"DICE: {dice}")

        # # AUROCの計算
        # flat_truth = norm_label_voxel_np.flatten()
        # flat_pred = norm_diff_all_np.flatten()
        # auroc = roc_auc_score(flat_truth, flat_pred)
        # auroc_list.append(auroc)
        # print(f"AUROC: {auroc}")
        # print("")

        # ROCカーブとROCの計算
        # flat_truth = flat_truth / 255.0
        # flat_pred = flat_pred / 255.0
        # fpr, tpr, thresholds = roc_curve(flat_truth, flat_pred)
        # print("fpr: ", fpr)
        # print("tpr: ", tpr)
        # print("thresholds: ", thresholds)
        # roc_auc = auc(fpr, tpr)
        # print(roc_auc)
        # sys.exit()

    # # DICEとAUROCの値をJSON形式で保存
    # dice_and_auroc_json = {}
    # dice_avg = np.mean(dice_list)
    # auroc_avg = np.mean(auroc_list)
    # print(f"DICE average: {dice_avg}")
    # print(f"AUROC average: {auroc_avg}")
    # dice_and_auroc_json["average"] = {"dice": dice_avg, "auroc": auroc_avg}
    # for fname, dice, auroc in zip(filename_list, dice_list, auroc_list):
    #     dice_and_auroc_json[fname] = {
    #         "dice": dice,
    #         "auroc": auroc,
    #     }

    # with open("/mnt/ito/diffusion-anomaly/out/dice_and_auroc.json", "w") as f:
    #     json.dump(dice_and_auroc_json, f, indent=4)
