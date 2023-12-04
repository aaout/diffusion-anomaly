import matplotlib.pyplot as plt
import argparse
import os
import sys
import json

import numpy as np
import cv2
import torch

sys.path.append("/mnt/ito/diffusion-anomaly/")
from guided_diffusion.bratsloader import BRATSDataset
from segment_anything import SamPredictor, sam_model_registry


# TODO: マスク画像からpoints promtを作成する関数を作成
def make_points_promt(mask, num_point):
    """マスク画像からpoints promtを作成する関数

    Args:
        mask: マスク画像 [1, 1, 240, 240]
        num_point: points promtの数

    Returns:
        points_promt: points promt [1, 2, 240, 240]
    """
    # マスク画像を2値化
    mask = mask.squeeze()
    mask = mask > 0.5
    # マスク画像からランダムにnum_pointの座標を取得
    h, w = mask.shape
    points = []
    for i in range(num_point):
        while True:
            x = np.random.randint(w)
            y = np.random.randint(h)
            if mask[y, x]:
                points.append([x, y])
                break
    points = np.array(points)
    points = points.reshape(1, 2, -1)
    points_promt = torch.from_numpy(points).float()
    return points_promt


def main():
    data_dir = "/media/user/ボリューム/brats_imgs/train/"
    batch_size = 1
    model_type = "vit_h"
    device = "cuda"
    num_point = 1

    # SAMモデル構築
    sam_checkpoint = (
        "/mnt/ito/diffusion-anomaly/segment_anything/models/sam_vit_h_4b8939.pth"
    )
    print(f"Loading SAM model {sam_checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    print("Loading dataset...")
    ds = BRATSDataset(data_dir, test_flag=False)
    datal = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    for i, (images, out_dict, weak_label, label, number) in enumerate(datal):
        # print(f"image: {images.shape}")
        # print(f"out_dict: {out_dict}")
        # print(f"weak_label: {weak_label}")
        # print(f"label: {label.shape}")
        # print(f"number: {number}")

        for image in images[0]:
            input_image = image.numpy()
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            print(input_image.shape)
            predictor.set_image(input_image)
            sys.exit()


if __name__ == "__main__":
    main()
