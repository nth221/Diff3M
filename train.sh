MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 3"

ADDITIONAL_FLAGS="--one_class True --exp_name train_model --demo_cat_num 3 --ehr_cat_num 0"

torchrun --nproc_per_node=8 --master_port=34797 ./main/train_diff3M.py --data_dir '{dataset_root_path}' --dataset 'mimic' --resume_checkpoint '{pretrained_model_path}' $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ADDITIONAL_FLAGS
