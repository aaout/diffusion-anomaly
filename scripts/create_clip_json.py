import os
import sys
import json
from tqdm import tqdm
import shutil
import nibabel
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
from PIL import Image


def create_clip_json(voxel_data):
    """各被験者の各チャネルのボクセルデータに対し、clip点(上位1%と下位1%)の値を返す関数
    jsonファイルは/data/clip_train.json, /data/clip_test.jsonに保存
    作成したjsonファイルはscripts/create_dataset.pyで使用

    Args:
        voxel_data (numpy.ndarray [240, 240, 155]): 各被験者の各チャネルのボクセルデータ

    Returns:
        upper_clip (int): 上位1%の値
        lower_clip (int): 下位1%の値
    """
    total_pixel_values = []  # 各スライスの背景を除くピクセル値の合計を格納するリスト
    # 上位1%と下位1%の値を取得
    for slice_number in range(voxel_data.shape[2]):
        np_slice = voxel_data[:, :, slice_number]
        pixel_values = np_slice.ravel()
        non_zero_pixel_values = pixel_values[pixel_values != 0]  # 背景を削除
        total_pixel_values.extend(non_zero_pixel_values)  # ヒストグラム用の変数に追加
    sorted_lists = sorted(total_pixel_values)
    total_count = len(sorted_lists)
    top_1_percent_index = int(0.99 * total_count)
    bottom_1_percent_index = int(0.01 * total_count)
    upper_clip = sorted_lists[top_1_percent_index]
    lower_clip = sorted_lists[bottom_1_percent_index]

    return upper_clip, lower_clip


if __name__ == "__main__":
    FLAG = "test"  # train or test
    if FLAG == "train":
        DATA_DIR = "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData"
    elif FLAG == "test":
        DATA_DIR = "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_ValidationData"
    else:
        print("FLAG must be train or test")
        sys.exit()

    file_list = []
    upper_clip_list = []
    lower_clip_list = []
    with tqdm(os.walk(DATA_DIR)) as pbar:
        for root, dirs, files in pbar:
            # print(root)
            # print(dirs)
            # print(files)
            # number = dirs[0].split("_")[-1]
            # print(number)
            if not dirs:
                # 各被験者の各チャネル[seg, flair, t1, t1ce, t2]をfilesとして読み込む
                files = sorted(files)
                for f in files:
                    data_path = os.path.join(root, f)
                    nib_voxel = nibabel.load(data_path)
                    np_voxel = nib_voxel.get_fdata()
                    # np_norm_voxel = norm_func(np_voxel)

                    upper_clip, lower_clip = create_clip_json(np_voxel)
                    file_list.append(f)
                    upper_clip_list.append(upper_clip)
                    lower_clip_list.append(lower_clip)

    clip_json = {}
    for fname, cmax, cmin in zip(file_list, upper_clip_list, lower_clip_list):
        clip_json[fname] = {"upper_clip": cmax, "lower_clip": cmin}

    with open(f"/mnt/ito/diffusion-anomaly/data/clip_{FLAG}.json", "w") as f:
        json.dump(clip_json, f, indent=4)
