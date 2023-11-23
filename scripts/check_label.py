import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import nibabel


def check_label(imgs_path, mode):
    abnormal_slice_dict = {}
    for subject_id in sorted(os.listdir(imgs_path)):
        print("")
        print(f"ID: {subject_id}")
        abnormal_slice_list = []
        subject_path = os.path.join(imgs_path, subject_id)
        for slice_id in sorted(os.listdir(subject_path)):
            slice_path = os.path.join(subject_path, slice_id)
            seg_slice_path = [
                seg_nii for seg_nii in os.listdir(slice_path) if "seg" in seg_nii
            ]
            seg_slice_absolute_path = os.path.join(slice_path, seg_slice_path[0])
            seg_slice = nibabel.load(seg_slice_absolute_path)
            seg_slice_tensor = torch.tensor(seg_slice.get_fdata())
            if seg_slice_tensor.max() > 0:
                print(f"abnormal: {slice_id}")
                abnormal_slice_list.append(slice_id.split("_")[-1])
        abnormal_slice_dict[subject_id] = abnormal_slice_list

    with open(
        f"/mnt/ito/diffusion-anomaly/out/abnormal_slice_dict_{mode}.json", "w"
    ) as f:
        json.dump(abnormal_slice_dict, f)


if __name__ == "__main__":
    brats_train_path = "/media/user/ボリューム/brats_imgs/train/"
    brats_test_path = "/media/user/ボリューム/brats_imgs/test_labels/"
    check_label(brats_train_path, "train")
    check_label(brats_test_path, "test")
