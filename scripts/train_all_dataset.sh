#!/bin/bash

# Training on full dataset (all cameras included)

cd "$(dirname "$0")/.."
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
accelerate launch  --gpu_ids 0 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=20000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=150 \
  --output_dir="/path/to/your/output_directory" \
  --validation_epochs=1 \
  --dataloader_num_workers=4 \
  --mixed_precision="fp16" \
  --cache_dir="/path/to/your/cache_directory"\
  --val_image_folder="val_image" \
  --single_timestep=1000 \
  --train_data_dir="/path/to/your/processed_dataset" \
  --dataset="gehler" \
  --fold 'all' \
  --checkpointing_steps 1000 \
  --crop_prob 0.75 \
  --crop_size 0.5 \
  --color_aug_prob 1.0 \