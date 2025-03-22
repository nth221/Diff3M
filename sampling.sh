MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"

ADDITIONAL_FLAGS="--one_class True --latent_control False --exp_name test_model"

torchrun --nproc_per_node=8 --master_port=34700 ./main/sampling_diff3M.py --noise_level 400 --data_dir '{dataset_root_path}' --model_path '{trained_model_path}' --dataset 'mimic' $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $ADDITIONAL_FLAGS
