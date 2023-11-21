# 拡散モデルのテストを行うスクリプト
# 詳細はscripts/image_valid.pyを参照

MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr_anneal_steps 80000 --log_interval 100 --save_interval 2500"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"
python3 scripts/image_valid.py --data_dir /media/user/ボリューム/brats_imgs/train --dataset brats $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
