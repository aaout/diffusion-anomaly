import sys
import torch
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# DICE係数の計算
def dice_coefficient(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    intersection = np.sum(y_true_flat & y_pred_flat)
    print(f"intersection: {intersection}")
    print(f"y_true_flat: {np.sum(y_true_flat)}")
    print(f"y_pred_flat: {np.sum(y_pred_flat)}")
    print(f"DICE: {(2. * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat))}")
    return (2.0 * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat))


# TODO: 変数名のリファクタリング
# TODO: 各IDに対して, 閾値と混同行列の計算と保存(dice関数の中で算出)
# TODO: ヒストグラムに閾値ラインを追加した上で画像の保存


if __name__ == "__main__":
    SUBJECT_ID = "335"
    SICE_ID = "100"
    print(f"ID: {SUBJECT_ID}_{SICE_ID}")
    # 入出力差分のテンソルデータを読み込み, NumPy配列に変換
    tensor_data = torch.load(
        f"/media/user/ボリューム2/out/diffmap_pt/{SUBJECT_ID}/{SICE_ID}/diff_all.pt"
    )
    numpy_data = tensor_data.cpu().numpy()

    # ptデータのピクセル値のヒストグラムを表示
    plt.figure(figsize=(8, 6))
    plt.hist(numpy_data.ravel(), bins=100, color="b", alpha=0.7, rwidth=0.8)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.ylim(0, 2000)
    plt.show()
    plt.savefig("diffmap_hist_pt.jpg")
    plt.close()

    norm_image = cv2.normalize(
        numpy_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    norm_image = norm_image.astype(np.uint8)

    # numpyデータに変換し, ピクセル値を0-255に正規化した後のヒストグラムを表示
    plt.figure(figsize=(8, 6))
    plt.hist(norm_image.ravel(), bins=100, color="b", alpha=0.7, rwidth=0.8)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.ylim(0, 2000)
    plt.show()
    plt.savefig("diffmap_hist_normed.jpg")
    plt.close()

    # 入出力差分をOtsu手法により二値化
    thresh_value, binarized_input_image = cv2.threshold(
        norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    count_input = np.sum(binarized_input_image == 255)
    print(f"count_input: {count_input}")
    print(f"thresh_value: {thresh_value}")

    # 二値化された入力画像を保存
    plt.imshow(binarized_input_image, cmap="gray")
    plt.axis("off")
    plt.savefig("binarized_input_image.jpg", bbox_inches="tight", pad_inches=0)
    plt.close()

    # DICEやAUROCを計算するため, ラベルデータを読み込む
    ground_truth_mask = cv2.imread(
        f"/media/user/ボリューム2/out/abnormal_sample_and_heatmap/{SUBJECT_ID}/{SICE_ID}_label.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    ground_truth_mask = cv2.normalize(
        ground_truth_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    _, binarized_ground_truth_mask = cv2.threshold(
        ground_truth_mask, 50, 255, cv2.THRESH_BINARY
    )
    binarized_ground_truth_mask = binarized_ground_truth_mask.astype(np.uint8)
    count_label = np.sum(binarized_ground_truth_mask == 255)
    print(f"count_label: {count_label}")
    plt.figure(figsize=(8, 6))
    plt.hist(
        binarized_ground_truth_mask.ravel(), bins=100, color="b", alpha=0.7, rwidth=0.8
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.ylim(0, 1000)
    plt.show()
    plt.savefig("ground_truth_mask_hist.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # ラベル画像のパディング
    padded_label_image = np.zeros((256, 256), dtype=np.uint8)
    padded_label_image[8:-8, 8:-8] = binarized_ground_truth_mask

    plt.imshow(padded_label_image, cmap="gray")
    plt.axis("off")
    plt.savefig("binarized_ground_truth_mask.jpg", bbox_inches="tight", pad_inches=0)
    plt.close()

    # 真実のマスクと二値化された画像のDICE係数を計算
    dice = dice_coefficient(padded_label_image, binarized_input_image)
    print(f"DICE coefficient: {dice}")

    # AUROCの計算
    flat_truth = padded_label_image.flatten()
    flat_pred = binarized_input_image.flatten()
    auroc = roc_auc_score(flat_truth, flat_pred)
    print(f"AUROC: {auroc}")
