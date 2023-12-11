import os
import sys
import json

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import nibabel as nib

sys.path.append("/mnt/ito/diffusion-anomaly/")
from guided_diffusion.bratsloader import BRATSDataset
from segment_anything import SamPredictor, sam_model_registry


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


def make_points_promt(mask, num_point):
    """マスク画像からpoints promtを作成する関数

    Args:
        mask: 二値化したマスク画像 [256, 256, 3]
        num_point: points promtの数

    Returns:
        points_promt: points promt [num_point, 2]
        example: [[150, 220], [140, 250], [160, 200]]
    """
    # マスク画像からランダムにnum_pointの座標を取得
    h, w, _ = mask.shape
    points = []
    for i in range(num_point):
        while True:
            x = np.random.randint(w)
            y = np.random.randint(h)
            if mask[y, x, :].any():
                points.append([x, y])
                break
    points = np.array(points)
    return points


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1, 1, 1, 0.8])
        # color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def save_figure(image, save_path, mask=None, input_points=None, point_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    if mask is not None:
        show_mask(mask, plt.gca())
    if input_points is not None:
        show_points(input_points, point_labels, plt.gca())
    plt.axis("off")
    plt.show()
    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


if __name__ == "__main__":
    # 結果の保存先フォルダ名
    save_dir = "/media/user/ボリューム/out/segment_diffmap_sam/"
    # SAMで使用するモデルタイプの指定
    sam_checkpoint = (
        "/mnt/ito/diffusion-anomaly/segment_anything/models/sam_vit_h_4b8939.pth"
    )
    # sam_checkpoint = "models/sam_vit_b_01ec64.pth"
    # sam_checkpoint = "models/sam_vit_l_0b3195.pth"
    model_type = "vit_h"
    device = "cuda"
    num_point = 5

    # SAMのロード
    print(f"Loading SAM model {sam_checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    print("SAM loaded")

    # 入力画像の読み込み
    filename_list = []
    dice_mean_list = []
    dice_var_list = []
    diffmap_dirs = "/media/user/ボリューム/out/bin_voxel_and_label/"
    subject_dirs = sorted(os.listdir(diffmap_dirs))
    for subject_id in subject_dirs:
        print("=====================================")
        print(f"ID: {subject_id}")
        filename_list.append(subject_id)
        os.makedirs(f"{save_dir}/{subject_id}", exist_ok=True)
        subject_path = os.path.join(diffmap_dirs, subject_id)
        target_file_list = sorted(os.listdir(subject_path))

        # diffmapデータの読み込み
        diff_voxel_filename = [
            f
            for f in target_file_list
            if f.endswith("diff_voxel.nii") and f.startswith("diff_voxel")
        ]
        diff_voxel_path = os.path.join(subject_path, diff_voxel_filename[0])
        diff_voxel_np = nib.load(diff_voxel_path).get_fdata()
        norm_diff_voxel_np = cv2.normalize(
            diff_voxel_np,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )
        norm_diff_voxel_np = norm_diff_voxel_np.astype(np.uint8)

        # labelデータの読み込み
        label_voxel_filename = [
            f for f in target_file_list if f.startswith("bin_label_voxel.nii")
        ]
        label_voxel_path = os.path.join(subject_path, label_voxel_filename[0])
        label_voxel_np = nib.load(label_voxel_path).get_fdata()
        norm_label_voxel_np = cv2.normalize(
            label_voxel_np,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )
        norm_label_voxel_np = norm_label_voxel_np.astype(np.uint8)

        dice_list = []
        for slice_id in range(norm_diff_voxel_np.shape[0]):
            print("slice_id: ", slice_id + 80)
            diff_slice_np = norm_diff_voxel_np[slice_id, :, :]
            diff_slice_np = cv2.cvtColor(diff_slice_np, cv2.COLOR_BGR2RGB)
            label_slice_np_original = norm_label_voxel_np[slice_id, :, :]
            label_slice_np = cv2.cvtColor(label_slice_np_original, cv2.COLOR_BGR2RGB)

            # diffmapが真っ黒, つまり正常スライスと判定された場合はスキップ
            if label_slice_np.max() == 0:
                continue

            # SAMへのpoints prompt作成
            input_points = make_points_promt(label_slice_np, num_point)
            input_labels = np.ones(num_point)

            # SAMによるセグメンテーション
            predictor.set_image(diff_slice_np)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            pred_mask = masks[0].astype(np.uint8)
            norm_pred_mask = cv2.normalize(
                pred_mask,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            )

            # DICE係数計算
            dice = dice_coefficient(label_slice_np_original, norm_pred_mask)
            dice_list.append(dice)
            print(dice)

            # セグメンテーション結果の保存
            save_figure(
                diff_slice_np,
                f"{save_dir}/{subject_id}/{slice_id + 80}_input_diffmap.jpg",
            )
            save_figure(
                label_slice_np,
                f"{save_dir}/{subject_id}/{slice_id + 80}_lable_slice.jpg",
            )
            save_figure(
                diff_slice_np,
                f"{save_dir}/{subject_id}/{slice_id + 80}_input_diffmap_and_points.jpg",
                input_points=input_points,
                point_labels=input_labels,
            )
            save_figure(
                masks[0],
                f"{save_dir}/{subject_id}/{slice_id + 80}_pred_slice.jpg",
            )

        dice_list_np = np.array(dice_list)
        dice_mean_list.append(np.mean(dice_list_np))
        dice_var_list.append(np.var(dice_list_np))
        print(np.mean(dice_list_np))
        print(np.var(dice_list_np))

    dice_mean_list_np = np.array(dice_mean_list)
    print("mean DICE: ", np.mean(dice_mean_list_np))

    segment_diffmap_json = {}
    for fname, dice_mean, dice_var in zip(filename_list, dice_mean_list, dice_var_list):
        segment_diffmap_json[fname] = {"dice_mean": dice_mean, "dice_var": dice_var}
    with open("/mnt/ito/diffusion-anomaly/out/segment_diffmap.json", "w") as f:
        json.dump(segment_diffmap_json, f, indent=4)
