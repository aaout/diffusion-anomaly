# 学習済みの分類器と拡散モデルを用いて条件付き生成を行うスクリプト
# classifier_train.shとimage_train.shを実行した後に実行する
# 詳細はscripts/classifier_sample_known.pyを参照

MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"
python3 scripts/classifier_sample_known.py  --data_dir /media/user/ボリューム2/brats_imgs/test  --model_path /mnt/ito/diffusion-anomaly/out/20231027_trained_diffusion_models/brats2update062500.pt --classifier_path /mnt/ito/diffusion-anomaly/out/20231027_trained_classifier_models/modelbratsclass020000.pt --dataset brats --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS
