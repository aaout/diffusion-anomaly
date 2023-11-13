import os
import sys
import json
import torch
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# DICE係数の計算
def dice_coefficient(y_true, y_pred):
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
        subject_path = os.path.join(diffmap_dirs, subject_id)
        slice_list = sorted(os.listdir(subject_path))
        for slice_id in slice_list:
            slice_path = os.path.join(subject_path, slice_id)
            target_file_list = sorted(os.listdir(slice_path))
            # 対象の被験者名とスライス名を取得
            target_id = f"{subject_id}_{slice_id}"
            print(f"ID: {target_id}")

            # 4チャネル差分合計のファイル名を取得してnumpy形式に変換
            diff_all_pt = [f for f in target_file_list if f.endswith("diff_all.pt")]
            diff_all_pt_path = os.path.join(slice_path, diff_all_pt[0])
            diff_all_pt_data = torch.load(diff_all_pt_path)
            diff_all_np_data = diff_all_pt_data.cpu().numpy()
            norm_diff_all_np = cv2.normalize(
                diff_all_np_data,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            )
            norm_diff_all_np = norm_diff_all_np.astype(np.uint8)
            # 入出力差分をOtsu手法により二値化
            thresh_value, bin_norm_diff_all = cv2.threshold(
                norm_diff_all_np, 0, 255, cv2.THRESH_OTSU
            )
            thresh_value_list.append(thresh_value)

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
            # print("label_np_data.shape: ", label_np_data.shape)
            # print("label_np_data.type: ", label_np_data.dtype)

            # # 二値化された入力画像を保存
            # plt.imshow(bin_norm_diff_all, cmap="gray")
            # plt.axis("off")
            # plt.savefig(
            #     "label_image.jpg",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # plt.close()

            # # ptデータのピクセル値のヒストグラムを表示
            # plt.figure(figsize=(8, 6))
            # plt.hist(
            #     bin_norm_diff_all.ravel(),
            #     bins=100,
            #     color="b",
            #     alpha=0.7,
            #     rwidth=0.8,
            # )
            # plt.grid(axis="y", linestyle="--", alpha=0.7)
            # plt.tight_layout()
            # plt.ylim(0, 2000)
            # plt.show()
            # plt.savefig("label_np_hist.jpg")
            # plt.close()
            # sys.exit()

            # ラベル画像と二値化された画像のDICE係数を計算
            dice = dice_coefficient(bin_norm_label, bin_norm_diff_all)
            dice_list.append(dice)
            print(f"DICE: {dice}")

            # AUROCの計算
            flat_truth = bin_norm_label.flatten()
            flat_pred = bin_norm_diff_all.flatten()
            try:
                auroc = roc_auc_score(flat_truth, flat_pred)
                auroc_list.append(auroc)
                print(f"AUROC: {auroc}")
                print("")
            except ValueError:
                print("AUROC cannot be calculated.")
                print("")

    # # DICEとAUROCの値をJSON形式で保存
    dice_and_auroc_json = {}
    dice_avg = np.mean(dice_list)
    auroc_avg = np.mean(auroc_list)
    print(f"DICE average: {dice_avg}")
    print(f"AUROC average: {auroc_avg}")
    dice_and_auroc_json["average"] = {"dice": dice_avg, "auroc": auroc_avg}
    for fname, dice, auroc, thresh in zip(
        filename_list, dice_list, auroc_list, thresh_value_list
    ):
        dice_and_auroc_json[fname] = {"dice": dice, "auroc": auroc, "threshold": thresh}

    with open("/mnt/ito/diffusion-anomaly/out/dice_and_auroc.json", "w") as f:
        json.dump(dice_and_auroc_json, f, indent=4)
