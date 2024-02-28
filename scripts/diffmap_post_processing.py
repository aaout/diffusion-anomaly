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


def calculate_sensitivity(y_true, y_pred):
    """
    Calculate the sensitivity (true positive rate).

    Args:
        y_true (numpy.ndarray): Ground truth binary array.
        y_pred (numpy.ndarray): Predicted binary array.

    Returns:
    float: The sensitivity value
    """
    tp = np.sum(y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    if tp + fn == 0:
        return 0  # Avoid division by zero
    return tp / (tp + fn)


def calculate_specificity(y_true, y_pred):
    """
    Calculate the specificity (true negative rate).

    Args:
        y_true (numpy.ndarray): Ground truth binary array.
        y_pred (numpy.ndarray): Predicted binary array.

    Returns:
    float: The specificity value
    """
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    if tn + fp == 0:
        return 0  # Avoid division by zero
    return tn / (tn + fp)


if __name__ == "__main__":
    diffmap_dirs = (
        "/media/user/ボリューム/out/sample_data_and_heatmap_080-154_train_abnormal"
    )
    save_dir = "/media/user/ボリューム/out/bin_voxel_and_label_080-154_train_abnormal_post_processing"

    dice_list = []
    sensitivity_list = []
    specificity_list = []
    auroc_list = []
    filename_list = []
    thresh_value_list = []

    subject_dirs = sorted(os.listdir(diffmap_dirs))
    for subject_id in subject_dirs:
        if subject_id == "341":
            continue
        diff_all_voxel_list = []
        label_voxel_list = []
        diff_t1_voxel_list = []
        diff_t1ce_voxel_list = []
        diff_t2_voxel_list = []
        diff_flair_voxel_list = []
        subject_path = os.path.join(diffmap_dirs, subject_id)
        slice_list = sorted(os.listdir(subject_path))
        print(f"ID: {subject_id}")
        filename_list.append(subject_id)
        for slice_id in slice_list:
            slice_path = os.path.join(subject_path, slice_id)
            target_file_list = sorted(os.listdir(slice_path))

            label_pt = [f for f in target_file_list if f.endswith("label.pt")]
            label_pt_path = os.path.join(slice_path, label_pt[0])
            label_pt_data = torch.load(label_pt_path)
            label_np_data = label_pt_data.cpu().numpy()
            background = torch.zeros(256, 256)
            background[8:-8, 8:-8] = label_pt_data
            label_np_data = background.cpu().numpy()
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

            # t1の入出力差分は正の値を0にし, 負の値の絶対値をとる
            diff_t1_pt = [f for f in target_file_list if f.endswith("diff_t1.pt")]
            diff_t1_pt_path = os.path.join(slice_path, diff_t1_pt[0])
            diff_t1_pt_data = torch.load(diff_t1_pt_path)
            diff_t1_pt_clip = torch.clamp(diff_t1_pt_data, None, 0)
            diff_t1_pt_clip = torch.abs(diff_t1_pt_clip)
            # t1_np = diff_t1_pt_clip.cpu().numpy()
            # heatmap = plt.imshow(
            #     t1_np,
            #     cmap="bwr",
            #     interpolation="nearest",
            #     vmax=1,
            #     vmin=-1,
            # )
            # plt.axis("off")
            # plt.show()
            # plt.savefig(
            #     "diffmap_t1.jpg",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # plt.close()

            # t1ceの入出力差分は絶対値をとる
            diff_t1ce_pt = [f for f in target_file_list if f.endswith("diff_t1ce.pt")]
            diff_t1ce_pt_path = os.path.join(slice_path, diff_t1ce_pt[0])
            diff_t1ce_pt_data = torch.load(diff_t1ce_pt_path)
            diff_t1ce_pt_clip = torch.abs(diff_t1ce_pt_data)
            # diff_t1ce_pt_clip = torch.clamp(diff_t1ce_pt_data, 0, None)
            # t1ce_np = diff_t1ce_pt_clip.cpu().numpy()
            # heatmap = plt.imshow(
            #     t1ce_np,
            #     cmap="bwr",
            #     interpolation="nearest",
            #     vmax=1,
            #     vmin=-1,
            # )
            # plt.axis("off")
            # plt.show()
            # plt.savefig(
            #     "diffmap_t1ce.jpg",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # plt.close()

            diff_t2_pt = [f for f in target_file_list if f.endswith("diff_t2.pt")]
            diff_t2_pt_path = os.path.join(slice_path, diff_t2_pt[0])
            diff_t2_pt_data = torch.load(diff_t2_pt_path)
            diff_t2_pt_clip = torch.clamp(diff_t2_pt_data, 0, None)
            # t2_np = diff_t2_pt_clip.cpu().numpy()
            # heatmap = plt.imshow(
            #     t2_np,
            #     cmap="bwr",
            #     interpolation="nearest",
            #     vmax=1,
            #     vmin=-1,
            # )
            # plt.axis("off")
            # plt.show()
            # plt.savefig(
            #     "diffmap_t2.jpg",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # plt.close()

            diff_flair_pt = [f for f in target_file_list if f.endswith("diff_flair.pt")]
            diff_flair_pt_path = os.path.join(slice_path, diff_flair_pt[0])
            diff_flair_pt_data = torch.load(diff_flair_pt_path)
            diff_flair_pt_clip = torch.clamp(diff_flair_pt_data, 0, None)
            # flair_np = diff_flair_pt_clip.cpu().numpy()
            # heatmap = plt.imshow(
            #     flair_np,
            #     cmap="bwr",
            #     interpolation="nearest",
            #     vmax=1,
            #     vmin=-1,
            # )
            # plt.axis("off")
            # plt.show()
            # plt.savefig(
            #     "diffmap_flair.jpg",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # plt.close()
            # sys.exit()

            diff_all_pt_clip = (
                diff_t1_pt_clip
                + diff_t1ce_pt_clip
                + diff_t2_pt_clip
                + diff_flair_pt_clip
            )
            diff_all_voxel_list.append(diff_all_pt_clip)

        diff_all_voxel = torch.stack(diff_all_voxel_list, dim=0)
        diff_all_voxel_np = diff_all_voxel.cpu().numpy()
        norm_diff_all_np = cv2.normalize(
            diff_all_voxel_np,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )

        os.makedirs(f"{save_dir}/{subject_id}", exist_ok=True)
        norm_diff_all_nii = nib.Nifti1Image(norm_diff_all_np, affine=np.eye(4))
        norm_diff_all_nii.to_filename(f"{save_dir}/{subject_id}/diff_voxel.nii")
        norm_diff_all_np = norm_diff_all_np.astype(np.uint8)

        # Otsu thresholdingにより二値化
        diff_all_thresh = threshold_otsu(norm_diff_all_np)
        thres, bin_diff_all = cv2.threshold(
            norm_diff_all_np, diff_all_thresh, 255, cv2.THRESH_BINARY
        )
        print(f"Otsu threshold: {thres}")
        thresh_value_list.append(thres)

        # 二値化したセグメンテーション結果をNIfTI形式で保存
        bin_diff_all_nii = nib.Nifti1Image(bin_diff_all, affine=np.eye(4))
        bin_diff_all_nii.to_filename(f"{save_dir}/{subject_id}/otsubin_diff_voxel.nii")

        label_voxel = torch.stack(label_voxel_list, dim=0)
        label_voxel_np = label_voxel.cpu().numpy()
        norm_label_voxel_np = cv2.normalize(
            label_voxel_np,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )
        # 二値化したラベルデータをNIfTI形式で保存
        norm_label_voxel_nii = nib.Nifti1Image(norm_label_voxel_np, affine=np.eye(4))
        norm_label_voxel_nii.to_filename(f"{save_dir}/{subject_id}/bin_label_voxel.nii")
        norm_label_voxel_np = norm_label_voxel_np.astype(np.uint8)

        # ラベル画像と二値化された画像のDICE係数を計算
        dice = dice_coefficient(norm_label_voxel_np, bin_diff_all)
        sensitivity = calculate_sensitivity(norm_label_voxel_np, bin_diff_all)
        specificity = calculate_specificity(norm_label_voxel_np, bin_diff_all)
        dice_list.append(dice)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        print(f"DICE: {dice}")
        print(f"sensitivity: {sensitivity}")
        print(f"specificity: {specificity}")

        # AUROCの計算
        flat_truth = norm_label_voxel_np.flatten()
        flat_pred = norm_diff_all_np.flatten()
        auroc = roc_auc_score(flat_truth, flat_pred)
        auroc_list.append(auroc)
        print(f"AUROC: {auroc}")
        print("")

    # DICEとAUROCの値をJSON形式で保存
    dice_and_auroc_json = {}
    dice_avg = np.mean(dice_list)
    threshold_avg = np.mean(thresh_value_list)
    sensitivity_avg = np.mean(sensitivity_list)
    specificity_avg = np.mean(specificity_list)
    auroc_avg = np.mean(auroc_list)
    print(f"DICE average: {dice_avg}")
    print(f"sensitivity average: {sensitivity_avg}")
    print(f"specificity average: {specificity_avg}")
    print(f"AUROC average: {auroc_avg}")
    print(f"threshold average: {threshold_avg}")

    dice_and_auroc_json["average"] = {
        "dice": dice_avg,
        "senstivity": sensitivity_avg,
        "specificity": specificity_avg,
        "auroc": auroc_avg,
        "threshold": threshold_avg,
    }
    for fname, dice, sensi, speci, auroc, thres_value in zip(
        filename_list,
        dice_list,
        sensitivity_list,
        specificity_list,
        auroc_list,
        thresh_value_list,
    ):
        dice_and_auroc_json[fname] = {
            "dice": dice,
            "sensitivity": sensi,
            "specificity": speci,
            "auroc": auroc,
            "threshold": thres_value,
        }

    with open("/mnt/ito/diffusion-anomaly/out/dice_and_auroc_ReLU.json", "w") as f:
        json.dump(dice_and_auroc_json, f, indent=4)
