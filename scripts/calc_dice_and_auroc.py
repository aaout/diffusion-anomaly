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
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)


if __name__ == '__main__':
    dice_list = []
    auroc_list = []
    filename_list = []
    thresh_value_list = []
    diffmap_dirs = "/media/user/ボリューム2/out/diffmap_pt/"

    for root, dirs, files in os.walk(diffmap_dirs):
        if not dirs:
            # ファイル名を取得
            diff_all_file = [f for f in files if f.endswith('all.pt')]
            diff_all_file_path = os.path.join(root, diff_all_file[0])
            slice_id = diff_all_file_path.split("/")[7]
            subject_path = diff_all_file_path.split("/")[:7]
            subject_path_concat = "/".join(subject_path)
            ground_truth_mask_path = subject_path_concat + "/" + slice_id + "_label.jpg"
            ground_truth_mask_path = ground_truth_mask_path.replace("diffmap_pt", "abnormal_sample_and_heatmap")
            subject_id = subject_path[-1] + "_" + slice_id
            filename_list.append(subject_id)
            print(f"ID: {subject_id}")

            # 入出力差分のテンソルデータを読み込み, NumPy配列に変換
            tensor_data = torch.load(diff_all_file_path)
            numpy_data = tensor_data.cpu().numpy()

            # ptデータのピクセル値のヒストグラムを表示
            plt.figure(figsize=(8, 6))
            plt.hist(numpy_data.ravel() , bins=100, color='b', alpha=0.7, rwidth=0.8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.ylim(0, 2000)
            plt.show()
            os.makedirs(
                f"{diffmap_dirs}/{subject_path[-1]}/{slice_id}",
                exist_ok=True,
            )
            plt.savefig(f"{diffmap_dirs}/{subject_path[-1]}/{slice_id}/diffmap_hist_pt.jpg")
            plt.close()

            # pt wo normalize
            norm_image = cv2.normalize(numpy_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            norm_image = norm_image.astype(np.uint8)

            # 入出力差分をOtsu手法により二値化
            thresh_value, binarized_input_image = cv2.threshold(norm_image, 0, 255, cv2.THRESH_OTSU)
            thresh_value_list.append(thresh_value)

            # 二値化された入力画像を保存
            plt.imshow(binarized_input_image, cmap='gray')
            plt.axis('off')
            plt.savefig(f'{diffmap_dirs}/{subject_path[-1]}/{slice_id}/binarized_diffmap_image.jpg', bbox_inches='tight', pad_inches=0)
            plt.close()            

            # DICEやAUROCを計算するため, ラベルデータを読み込む
            ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth_mask is None:
                print(f'ground_truth_mask is None: {ground_truth_mask_path}')
                sys.exit()
            else:
                ground_truth_mask = cv2.normalize(ground_truth_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                _, binarized_ground_truth_mask = cv2.threshold(ground_truth_mask, 50, 255, cv2.THRESH_BINARY)
                binarized_ground_truth_mask = binarized_ground_truth_mask.astype(np.uint8)
                padded_mask = np.zeros((256, 256), dtype=np.uint8)
                padded_mask[8:-8, 8:-8] = binarized_ground_truth_mask

                plt.imshow(padded_mask, cmap='gray')
                plt.axis('off')
                plt.savefig(f'{diffmap_dirs}/{subject_path[-1]}/{slice_id}/binarized_label_image.jpg', bbox_inches='tight', pad_inches=0)
                plt.close()

                # 真実のマスクと二値化された画像のDICE係数を計算
                dice = dice_coefficient(padded_mask, binarized_input_image)
                dice_list.append(dice)
                print(f"DICE: {dice}")

                # AUROCの計算
                flat_truth = padded_mask.flatten()
                flat_pred = binarized_input_image.flatten()
                auroc = roc_auc_score(flat_truth, flat_pred)
                auroc_list.append(auroc)
                print(f"AUROC: {auroc}")
                print("")

    # DICEとAUROCの値をJSON形式で保存
    dice_and_auroc_json = {}
    dice_avg = np.mean(dice_list)
    auroc_avg = np.mean(auroc_list)
    print(f"DICE average: {dice_avg}")
    print(f"AUROC average: {auroc_avg}")
    dice_and_auroc_json["average"] = {"dice": dice_avg, "auroc": auroc_avg}
    for fname, dice, auroc, thresh in zip(filename_list, dice_list, auroc_list, thresh_value_list):
        dice_and_auroc_json[fname] = {"dice": dice, "auroc": auroc, "threshold": thresh}

    with open("/mnt/ito/diffusion-anomaly/out/dice_and_auroc.json", "w") as f:
        json.dump(dice_and_auroc_json, f, indent=4)
