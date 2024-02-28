# 分類器のテストを行うスクリプト
# 詳細はscripts/classifier_valid.pyを参照

MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 49"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --save_interval 2000 --iterations 30000"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"

# script loggers_classifier_valid.txt
python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass020000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass004000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass006000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass008000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass010000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass012000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass014000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass016000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass018000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass020000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass022000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass024000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass026000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass028000.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS

# python3 scripts/classifier_valid.py --data_dir /media/user/ボリューム/brats_imgs_080-128/train/ --val_data_dir /media/user/ボリューム/brats_imgs_080-128/test/ --dataset brats --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass029999.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS
