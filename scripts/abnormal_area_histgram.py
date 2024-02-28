import os
import sys

from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from PIL import Image
import nibabel

if __name__ == "__main__":
    channel = "t1"
    diffmap_dirs = (
        "/media/user/ボリューム/out/sample_data_and_heatmap_080-154_train_abnormal/001/081"
    )

    channel_list = sorted(os.listdir(diffmap_dirs))

    # label_pt = [f for f in channel_list if f.endswith("label.pt")]
    # label_pt_path = os.path.join(diffmap_dirs, label_pt[0])
    # label_pt_data = torch.load(label_pt_path)
    # background = torch.zeros(256, 256)
    # background[8:-8, 8:-8] = label_pt_data
    # label_np_data = background.cpu().numpy()
    # plt.figure(figsize=(8, 6))
    # plt.hist(
    #     label_np_data.ravel(),
    #     bins=100,
    #     color="b",
    #     alpha=0.7,
    #     rwidth=0.8,
    # )
    # plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.ylim(0, 1000)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("hist_label.jpg")
    # plt.close()

    t1_pt = [f for f in channel_list if f.endswith(f"diff_{channel}.pt")]
    t1_pt_path = os.path.join(diffmap_dirs, t1_pt[0])
    t1_pt_data = torch.load(t1_pt_path)
    t1_np_data = t1_pt_data.cpu().numpy()

    t1_np_func = np.where(t1_np_data < 0, abs(t1_np_data), 0)
    # t1_np_func = np.abs(t1_np_data)
    # t1_np_func = np.maximum(0, t1_np_data)

    heatmap_t1 = plt.imshow(
        t1_np_func, cmap="bwr", interpolation="nearest", vmax=1, vmin=-1
    )
    # plt.colorbar(heatmap_t1)
    # plt.imshow(label_np_data, cmap="bwr")
    plt.axis("off")
    plt.savefig(
        f"{channel}_abs.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    sys.exit()
    # print(t1_np_data.shape)
    # print(t1_np_data.max())
    # print(t1_np_data.min())
    # plt.figure(figsize=(8, 6))
    # plt.hist(
    #     t1_np_data.ravel(),
    #     bins=100,
    #     color="b",
    #     alpha=0.7,
    #     rwidth=0.8,
    # )
    # plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.ylim(0, 1000)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f"hist_{channel}_input.jpg")
    # plt.close()

    t1_pt_sample = [f for f in channel_list if f.endswith(f"sample_{channel}.pt")]
    t1_pt_sample_path = os.path.join(diffmap_dirs, t1_pt_sample[0])
    t1_pt_sample_data = torch.load(t1_pt_sample_path)
    t1_np_sample_data = t1_pt_sample_data.cpu().numpy()

    # print(t1_np_sample_data.shape)
    # print(t1_np_sample_data.max())
    # print(t1_np_sample_data.min())
    # plt.figure(figsize=(8, 6))
    # plt.hist(
    #     t1_np_sample_data.ravel(),
    #     bins=100,
    #     color="b",
    #     alpha=0.7,
    #     rwidth=0.8,
    # )
    # plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.ylim(0, 1000)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f"hist_{channel}_sample.jpg")
    # plt.close()

    input_abnormal_area = t1_np_data[label_np_data == 1]
    mask = label_np_data == 1
    segmented_region = np.zeros_like(t1_np_data)
    segmented_region[mask] = t1_np_data[mask]

    plt.imshow(segmented_region, cmap="gray")
    plt.axis("off")
    plt.savefig(
        f"input_abnormal_area_{channel}.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(
        input_abnormal_area.ravel(),
        bins=80,
        color="b",
        alpha=0.7,
        rwidth=0.8,
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(-0.1, 1)
    plt.ylim(0, 1000)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"hist_{channel}_input_normal_area.jpg")
    # plt.savefig(f"hist_{channel}_input_abnormal_area.jpg")
    plt.close()

    mask = label_np_data == 1
    segmented_region = np.zeros_like(t1_np_sample_data)
    segmented_region[mask] = t1_np_sample_data[mask]
    sample_abnormal_area = t1_np_sample_data[label_np_data == 1]

    plt.imshow(segmented_region, cmap="gray")
    plt.axis("off")
    plt.savefig(
        f"sample_abnormal_area_{channel}.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(
        sample_abnormal_area.ravel(),
        bins=80,
        color="b",
        alpha=0.7,
        rwidth=0.8,
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(-0.1, 1)
    plt.ylim(0, 1000)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"hist_{channel}_sample_normal_area.jpg")
    # plt.savefig(f"hist_{channel}_sample_abnormal_area.jpg")
    plt.close()

    print(f"{channel}")
    print(f"input mean: {input_abnormal_area.mean()}")
    print(f"input std: {input_abnormal_area.std()}")
    print(f"sample mean: {sample_abnormal_area.mean()}")
    print(f"sample std: {sample_abnormal_area.std()}")
