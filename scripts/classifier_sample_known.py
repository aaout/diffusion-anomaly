"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
import sys
import json

sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import torch.nn.functional as F
import numpy as np
import cv2
import torch as th
import torch.distributed as dist
import torchvision.utils as vutils
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# 0: 異常データのみ入力
# 1: 正常データのみ入力
SAMPLE_MODE = 1
FOLDER_NAME = "sample_data_and_heatmap"


def norm_func(slice_data):
    slice_data_cpu = slice_data.cpu().numpy()
    max_slice_data = slice_data_cpu.max()
    min_slice_data = slice_data_cpu.min()
    # normalized_data = np.clip(slice_data_cpu, lower_clip, upper_clip)
    normalized_slice = (slice_data_cpu - min_slice_data) / (
        max_slice_data - min_slice_data
    )
    return th.tensor(normalized_slice)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.dataset == "brats":
        ds = BRATSDataset(args.data_dir, test_flag=True)
        datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
        # dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(dist_util.load_state_dict(args.classifier_path))

    print("loaded classifier")
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print("pmodel", p1, "pclass", p2)

    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a = th.autograd.grad(selected.sum(), x_in)[0]
            return a, a * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    for img in datal:
        model_kwargs = {}
        # img[0] (input_img): input test data [1, 4, 256, 256]
        # img[1]: label dict {'y': tensor([1])} 1->diseased, 0->healthy
        # img[2] (normal_or_abnormal): weak label 1->diseased, 0->healthy
        # img[3] (label_img): label img data [1, 1, 240, 240]
        # img[4] (file_name): file name tuple: ('BraTS20_Training_349_t1_099.nii.gz',)
        print("")
        print(img[4][0])
        print("weakly label: ", img[1])

        # 入力ファイルに対応するclip点をjsonファイルから獲得
        file_name = img[4][0]
        split_filename_list = file_name.split("_")
        subject_name = "_".join(split_filename_list[:-2])
        # subject_name_flair = subject_name + "_flair.nii"
        # subject_name_t1 = subject_name + "_t1.nii"
        # subject_name_t1ce = subject_name + "_t1ce.nii"
        # subject_name_t2 = subject_name + "_t2.nii"
        # upper_clip_flair, lower_clip_flair = refer_clip_point(subject_name_flair)
        # upper_clip_t1, lower_clip_t1 = refer_clip_point(subject_name_t1)
        # upper_clip_t1ce, lower_clip_t1ce = refer_clip_point(subject_name_t1ce)
        # upper_clip_t2, lower_clip_t2 = refer_clip_point(subject_name_t2)
        # print("upper_clip_flair: ", upper_clip_flair)
        # print("lower_clip_flair: ", lower_clip_flair)
        # print("upper_clip_flair: ", upper_clip_t1)
        # print("lower_clip_flair: ", lower_clip_t1)
        # print("upper_clip_flair: ", upper_clip_t1ce)
        # print("lower_clip_flair: ", lower_clip_t1ce)
        # print("upper_clip_flair: ", upper_clip_t2)
        # print("lower_clip_flair: ", lower_clip_t2)
        # sys.exit()

        # 複数の情報を持つ入力データimgを別々の変数に格納
        subject_number = file_name.split("_")[2]
        slice_num = file_name.split(".")[0].split("_")[-1]
        input_img = img[0].cpu()
        input_img_t1 = input_img[0, 0, ...]
        input_img_t1ce = input_img[0, 1, ...]
        input_img_t2 = input_img[0, 2, ...]
        input_img_flair = input_img[0, 3, ...]
        input_img_label = img[3][0, ...].cpu()
        normal_or_abnormal = img[2]

        # 特定の被験者からサンプリングを開始する場合
        # if subject_number in [f"{i:03}" for i in range(334, 355)]:
        #     print(f"skip {subject_number}")
        #     continue

        # 正常(0) or 異常データ(1)のみに対してサンプリング
        # if normal_or_abnormal == SAMPLE_MODE:
        #     continue  # take only diseased images as input

        if args.class_cond:
            classes = th.randint(
                low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
            )
            # 正常条件付け
            model_kwargs["y"] = classes
            # print("y", model_kwargs["y"])
        sample_fn = (
            diffusion.p_sample_loop_known
            if not args.use_ddim
            else diffusion.ddim_sample_loop_known
        )
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()
        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, args.image_size, args.image_size),
            img,
            org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level,
        )
        end.record()
        th.cuda.synchronize()
        th.cuda.current_stream().synchronize()

        print("time for 1000", start.elapsed_time(end))

        # sampleしたデータも分かりやすいように変数名変更
        sample_img_t1 = sample[0, 0, ...].cpu()
        sample_img_t1ce = sample[0, 1, ...].cpu()
        sample_img_t2 = sample[0, 2, ...].cpu()
        sample_img_flair = sample[0, 3, ...].cpu()

        if args.dataset == "brats":
            # 入力, 生成, 差分データを保存するフォルダを作成
            os.makedirs(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/",
                exist_ok=True,
            )

            # TODO: 保存する関数を作成
            # XXX: 画像を保存する際は以下の手順を踏む
            # 1. tensor型をnumpy型に変換
            # 2. cv2.normalizeでピクセル値を0~255に正規化
            # 3. astype(np.uint8)で型をuint8に変換
            # 4. plt.imshowで画像を表示した後, plt.savefigで画像を保存
            # 5. plt.closeで画像を閉じる

            # ここから入力データの保存
            # 入力データのt1チャネルを保存
            th.save(
                input_img_t1,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_t1.pt",
            )
            input_img_t1_np = input_img_t1.numpy()
            norm_input_img_t1 = cv2.normalize(
                input_img_t1_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_input_img_t1 = norm_input_img_t1.astype(np.uint8)
            plt.imshow(norm_input_img_t1, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_t1.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # 入力データのt1ceチャネルを保存
            th.save(
                input_img_t1ce,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_t1ce.pt",
            )
            input_img_t1ce_np = input_img_t1ce.numpy()
            norm_input_img_t1ce = cv2.normalize(
                input_img_t1ce_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_input_img_t1ce = norm_input_img_t1ce.astype(np.uint8)
            plt.imshow(norm_input_img_t1ce, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_t1ce.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # 入力データのt2チャネルを保存
            th.save(
                input_img_t2,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_t2.pt",
            )
            input_img_t2_np = input_img_t2.numpy()
            norm_input_img_t2 = cv2.normalize(
                input_img_t2_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_input_img_t2 = norm_input_img_t2.astype(np.uint8)
            plt.imshow(norm_input_img_t2, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_t2.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # 入力データのflairチャネルを保存
            th.save(
                input_img_flair,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_flair.pt",
            )
            input_img_flair_np = input_img_flair.numpy()
            norm_input_img_flair = cv2.normalize(
                input_img_flair_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_input_img_flair = norm_input_img_flair.astype(np.uint8)
            plt.imshow(norm_input_img_flair, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_flair.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # 入力データのlabelチャネルを保存
            th.save(
                input_img_label,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_label.pt",
            )
            input_img_label_np = input_img_label.numpy()
            norm_input_img_label = cv2.normalize(
                input_img_label_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_input_img_label = norm_input_img_label.astype(np.uint8)
            plt.imshow(norm_input_img_label, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_input_label.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # ここから生成データの保存
            # 生成データのt1チャネルを保存
            th.save(
                sample_img_t1,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_sample_t1.pt",
            )
            sample_img_t1_np = sample_img_t1.numpy()
            norm_sample_img_t1 = cv2.normalize(
                sample_img_t1_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_sample_img_t1 = norm_sample_img_t1.astype(np.uint8)
            plt.imshow(norm_sample_img_t1, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_sample_t1.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # 生成データのt1ceチャネルを保存
            th.save(
                sample_img_t1ce,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_sample_t1ce.pt",
            )
            sample_img_t1ce_np = sample_img_t1ce.numpy()
            norm_sample_img_t1ce = cv2.normalize(
                sample_img_t1ce_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_sample_img_t1ce = norm_sample_img_t1ce.astype(np.uint8)
            plt.imshow(norm_sample_img_t1ce, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_sample_t1ce.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # 生成データのt2チャネルを保存
            th.save(
                sample_img_t2,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_sample_t2.pt",
            )
            sample_img_t2_np = sample_img_t2.numpy()
            norm_sample_img_t2 = cv2.normalize(
                sample_img_t2_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_sample_img_t2 = norm_sample_img_t2.astype(np.uint8)
            plt.imshow(norm_sample_img_t2, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_sample_t2.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # 生成データのflairチャネルを保存
            th.save(
                sample_img_flair,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_sample_flair.pt",
            )
            sample_img_flair_np = sample_img_flair.numpy()
            norm_sample_img_flair = cv2.normalize(
                sample_img_flair_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            norm_sample_img_flair = norm_sample_img_flair.astype(np.uint8)
            plt.imshow(norm_sample_img_flair, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_sample_flair.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # ここから差分データの保存
            # 差分データのt1チャネルを保存
            diff_t1 = input_img_t1 - sample_img_t1
            th.save(
                diff_t1,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_diff_t1.pt",
            )
            # 入力データ(input_img)とsampleしたデータ(sample_img)の差分をヒートマップとして保存
            heatmap_t1 = plt.imshow(
                diff_t1, cmap="bwr", interpolation="nearest", vmax=1, vmin=-1
            )
            plt.colorbar(heatmap_t1)
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_heatmap_t1.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            diff_t1ce = input_img_t1ce - sample_img_t1ce
            th.save(
                diff_t1ce,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_diff_t1ce.pt",
            )
            heatmap_t1ce = plt.imshow(
                diff_t1ce, cmap="bwr", interpolation="nearest", vmax=1, vmin=-1
            )
            plt.colorbar(heatmap_t1ce)
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_heatmap_t1ce.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            diff_t2 = input_img_t2 - sample_img_t2
            th.save(
                diff_t2,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_diff_t2.pt",
            )
            heatmap_t2 = plt.imshow(
                diff_t2, cmap="bwr", interpolation="nearest", vmax=1, vmin=-1
            )
            plt.colorbar(heatmap_t2)
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_heatmap_t2.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            diff_flair = input_img_flair - sample_img_flair
            th.save(
                diff_flair,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_diff_flair.pt",
            )
            heatmap_flair = plt.imshow(
                diff_flair.cpu(), cmap="bwr", interpolation="nearest", vmax=1, vmin=-1
            )
            plt.colorbar(heatmap_flair)
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_heatmap_flair.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            diff_all = abs(org[0, :4, ...].cpu() - sample[0, ...].cpu()).sum(dim=0)
            th.save(
                diff_all,
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_diff_all.pt",
            )
            heatmap_all = plt.imshow(
                diff_all.cpu(), cmap="jet", interpolation="nearest"
            )
            plt.colorbar(heatmap_all)
            plt.savefig(
                f"/media/user/ボリューム/out/{FOLDER_NAME}/{subject_number}/{slice_num}/{subject_number}_{slice_num}_heatmap_all.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset="brats",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
