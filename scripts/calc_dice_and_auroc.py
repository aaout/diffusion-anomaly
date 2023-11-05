import sys
import torch
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# DICE係数の計算
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


if __name__ == '__main__':
    # 入出力差分のテンソルデータを読み込み, NumPy配列に変換
    tensor_data = torch.load('/media/user/ボリューム2/out/diffmap_pt/348/080/diff_all.pt')
    numpy_data = tensor_data.cpu().numpy()
    norm_image = cv2.normalize(numpy_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_image = norm_image.astype(np.uint8)

    # 入出力差分をOtsu手法により二値化
    _, binarized_image = cv2.threshold(norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plt.imshow(binarized_image, cmap='gray')
    # plt.axis('off')
    # plt.savefig('norm_input_image.png', bbox_inches='tight', pad_inches=0)
    # plt.close()

    # DICEやAUROCを計算するため, ラベルデータを読み込む
    ground_truth_mask = cv2.imread('/media/user/ボリューム2/out/abnormal_sample_and_heatmap/348/080_label.jpg', cv2.IMREAD_GRAYSCALE)
    ground_truth_mask = (ground_truth_mask > 30).astype(np.uint8)
    # plt.figure(figsize=(8, 6))
    # plt.hist(ground_truth_mask.ravel(), bins=100, color='b', alpha=0.7, rwidth=0.8)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.ylim(0, 1000)
    # plt.show()
    # plt.savefig('ground_truth_mask_hist.png', bbox_inches='tight', pad_inches=0)
    # plt.close()

    padded_mask = np.zeros((256, 256), dtype=np.uint8)
    padded_mask[8:-8, 8:-8] = ground_truth_mask
    # plt.imshow(padded_mask, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ground_truth_mask.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    # sys.exit()

    # 真実のマスクと二値化された画像のDICE係数を計算
    dice = dice_coefficient(padded_mask, binarized_image)
    print(f'DICE coefficient: {dice}')

    # AUROCの計算
    flat_truth = padded_mask.flatten()
    flat_pred = binarized_image.flatten()
    auroc = roc_auc_score(flat_truth, flat_pred)
    print(f'AUROC: {auroc}')
