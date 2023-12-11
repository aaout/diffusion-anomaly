import os
import sys
from tqdm import tqdm
import shutil
import nibabel
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
from PIL import Image


def norm_func(voxel_data):
    """各被験者の各チャネルのボクセルデータに対し、上位1%を1, 下位1%を0, その他を線形に正規化する関数

    Args:
        voxel_data (numpy.ndarray): 各被験者の各チャネルのボクセルデータ[240, 240, 155]

    Returns:
        normalized_data (numpy.ndarray): 0-1に正規化した後のボクセルデータ[240, 240, 155]
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
    max_value = sorted_lists[-1]
    min_value = sorted_lists[0]
    top_1_percent_index = int(0.99 * total_count)
    bottom_1_percent_index = int(0.01 * total_count)
    upper_clip = sorted_lists[top_1_percent_index]
    lower_clip = sorted_lists[bottom_1_percent_index]
    # print(f"upper clip: {upper_clip}")
    # print(f"lower clip: {lower_clip}")

    # upper_clip以上を1, lower_clip以下を0, その他は最大値をupper_clip, 最小値をlower_clipとして線形に正規化
    normalized_data = np.clip(voxel_data, lower_clip, upper_clip)
    normalized_data = (normalized_data - lower_clip) / (upper_clip - lower_clip)
    # print("max: ", np.max(normalized_data))
    # print("min: ", np.min(normalized_data))

    return normalized_data


# TODO: maxで割るのではなく、4で割る
def norm_seg_func(seg_slice_data):
    max_value = np.max(seg_slice_data)
    min_value = np.min(seg_slice_data)
    if max_value == 0:
        normalized_seg_slice_data = seg_slice_data
    else:
        normalized_seg_slice_data = (seg_slice_data - min_value) / (
            max_value - min_value
        )
    return normalized_seg_slice_data


if __name__ == "__main__":
    FLAG = "test"  # train or test
    save_dir = "/media/user/ボリューム/brats_imgs_000-154/"  # 全スライスを利用する場合
    # save_dir = "/media/user/ボリューム/brats_imgs_080-128/"  # 首側の80枚と頭頂側の26枚を削除する場合
    if FLAG == "train":
        DATA_DIR = "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData"
    elif FLAG == "test":
        DATA_DIR = "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_ValidationData"
    else:
        print("FLAG must be train or test")
        sys.exit()

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
                    # seg.niiはnorm_funcを適用せず正規化
                    if f.endswith("seg.nii"):
                        data_path = os.path.join(root, f)
                        nib_voxel = nibabel.load(data_path)
                        np_voxel = nib_voxel.get_fdata()
                        for slice_i in range(np_voxel.shape[2]):
                            # brats_imgs_000-154: 全てのスライスを使用
                            # brats_imgs_080-128: 首側の80枚と頭頂側の26枚を削除
                            # if slice_i < 80 or slice_i >= 129:
                            #     continue
                            np_slice = np_voxel[:, :, slice_i]
                            np_norm_slice = norm_seg_func(np_slice)
                            # スライスデータの左右反転と90度回転
                            flipped_image_data = np.fliplr(np_norm_slice)
                            rotated_image_data = np.rot90(flipped_image_data, k=1)

                            # labelデータ保存
                            # testデータのlabelsだけ別のディレクトリに保存
                            if FLAG == "test":
                                os.makedirs(
                                    f"{save_dir}/test_labels/{f.split('.')[0][0:20]}/slice_{str(slice_i).zfill(3)}/",
                                    exist_ok=True,
                                )
                                nib_slice = nibabel.Nifti1Image(
                                    rotated_image_data, affine=np.eye(4)
                                )
                                nibabel.save(
                                    nib_slice,
                                    f"{save_dir}/test_labels/{f.split('.')[0][0:20]}/slice_{str(slice_i).zfill(3)}/{f.split('.')[0][0:20]}_{f.split('.')[0].split('_')[-1]}_{str(slice_i).zfill(3)}.nii.gz",
                                )
                            else:  # trainのlabelデータ保存
                                os.makedirs(
                                    f"{save_dir}/{FLAG}/{f.split('.')[0][0:20]}/slice_{str(slice_i).zfill(3)}/",
                                    exist_ok=True,
                                )
                                nib_slice = nibabel.Nifti1Image(
                                    rotated_image_data, affine=np.eye(4)
                                )
                                nibabel.save(
                                    nib_slice,
                                    f"{save_dir}/{FLAG}/{f.split('.')[0][0:20]}/slice_{str(slice_i).zfill(3)}/{f.split('.')[0][0:20]}_{f.split('.')[0].split('_')[-1]}_{str(slice_i).zfill(3)}.nii.gz",
                                )
                    else:  # norm_funcを適用して正規化した後, 4チャネルの入力データを保存
                        data_path = os.path.join(root, f)
                        nib_voxel = nibabel.load(data_path)
                        np_voxel = nib_voxel.get_fdata()
                        np_norm_voxel = norm_func(np_voxel)

                        with tqdm(range(np_norm_voxel.shape[2]), leave=False) as pbar2:
                            pbar2.set_description(f"{f.split('.')[0][0:20]}")
                            print("\n")
                            for slice_i in pbar2:
                                # 首側の80枚と頭頂側の26枚は使用しない
                                # if slice_i < 80 or slice_i >= 129:
                                #     continue
                                np_norm_slice = np_norm_voxel[:, :, slice_i]
                                # スライスデータの左右反転と90度回転
                                flipped_image_data = np.fliplr(np_norm_slice)
                                rotated_image_data = np.rot90(flipped_image_data, k=1)

                                os.makedirs(
                                    f"{save_dir}/{FLAG}/{f.split('.')[0][0:20]}/slice_{str(slice_i).zfill(3)}/",
                                    exist_ok=True,
                                )
                                nib_slice = nibabel.Nifti1Image(
                                    rotated_image_data, affine=np.eye(4)
                                )
                                nibabel.save(
                                    nib_slice,
                                    f"{save_dir}/{FLAG}/{f.split('.')[0][0:20]}/slice_{str(slice_i).zfill(3)}/{f.split('.')[0][0:20]}_{f.split('.')[0].split('_')[-1]}_{str(slice_i).zfill(3)}.nii.gz",
                                )
