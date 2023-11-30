import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from segment_anything import SamPredictor, sam_model_registry


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


if __name__ == "__main__":
    # SAMで使用するモデルタイプの指定
    sam_checkpoint = (
        "/mnt/ito/diffusion-anomaly/segment_anything/models/sam_vit_h_4b8939.pth"
    )
    # sam_checkpoint = "models/sam_vit_b_01ec64.pth"
    # sam_checkpoint = "models/sam_vit_l_0b3195.pth"
    model_type = "vit_h"
    device = "cuda"
    num_point = 1

    # 入力画像の読み込み
    # image = cv2.imread(
    #     "/mnt/ito/diffusion-anomaly/out/binarized_img/334/80_diffmap.png"
    # )
    image = cv2.imread(
        "/mnt/ito/diffusion-anomaly/out/binarized_img/365/80_diffmap.png"
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # SAMへ画像を入力し, モデルを構築
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # promptの指定
    if num_point == 1:
        # 1 point
        input_point = np.array([[150, 220]])
        input_label = np.array([1])
    elif num_point == 3:
        # 3 point
        input_point = np.array([[150, 220], [140, 250], [160, 200]])
        input_label = np.array([1, 1, 1])
    elif num_point == 5:
        # 5 point
        input_point = np.array(
            [[150, 220], [140, 250], [160, 200], [140, 200], [130, 210]]
        )
        input_label = np.array([1, 1, 1, 1, 1])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis("on")
    plt.show()
    plt.savefig(
        f"output/input_{num_point}points.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )

    # 1pointをpromptとして入力して推論(ambiguous)
    # multimask_output=Trueを指定することで3つのマスクを出力
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    print("masks: ", masks.shape)
    print("scores: ", scores)
    # print("logits: ", logits)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    # plt.title(f"Score: {scores[0]:.3f}", fontsize=18)
    plt.axis("off")
    plt.show()
    plt.savefig(
        f"output/pred_{num_point}point.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(masks[0], cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(
        f"output/pred_{num_point}point_mask.jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    # # 2point+mask(モデルからの出力)をpromptとして入力して推論
    # # multimask_output=Falseを指定することで特定のマスクを出力
    # input_point = np.array([[500, 375], [1125, 625]])
    # input_label = np.array([1, 1])
    # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    # print("mask_input: ", mask_input.shape)

    # masks, _, _ = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     mask_input=mask_input[None, :, :],
    #     multimask_output=False,
    # )

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis("off")
    # plt.show()
    # plt.savefig("output/pred_with_2point_and_mask.jpg")

    # # pointにlabelを指定可能
    # # "1"とラベル付けされた点はセグメンテーションし, "0"とラベル付けされた点を除くようにセグメンテーション
    # # input_point = np.array([[500, 375], [1125, 625]])
    # # input_label = np.array([1, 0])
    # # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    # # masks, _, _ = predictor.predict(
    # #     point_coords=input_point,
    # #     point_labels=input_label,
    # #     mask_input=mask_input[None, :, :],
    # #     multimask_output=False,
    # # )

    # # bboxでpromptを指定するケース
    # # input_box = np.array([425, 600, 700, 875])
    # # masks, _, _ = predictor.predict(
    # #     point_coords=None,
    # #     point_labels=None,
    # #     box=input_box[None, :],
    # #     multimask_output=False,
    # # )

    # # 複数のbboxでpromptを指定するケース
    # # input_boxes = torch.tensor(
    # #     [
    # #         [75, 275, 1725, 850],
    # #         [425, 600, 700, 875],
    # #         [1375, 550, 1650, 800],
    # #         [1240, 675, 1400, 750],
    # #     ],
    # #     device=predictor.device,
    # # )
    # # transformed_boxes = predictor.transform.apply_boxes_torch(
    # #     input_boxes, image.shape[:2]
    # # )
    # # masks, _, _ = predictor.predict_torch(
    # #     point_coords=None,
    # #     point_labels=None,
    # #     boxes=transformed_boxes,
    # #     multimask_output=False,
    # # )
    # # masks.shape  # (batch_size) x (num_predicted_masks_per_input) x H x W
    # # plt.figure(figsize=(10, 10))
    # # plt.imshow(image)
    # # for mask in masks:
    # #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # # for box in input_boxes:
    # #     show_box(box.cpu().numpy(), plt.gca())
    # # plt.axis("off")
    # # plt.show()
