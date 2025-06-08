#!/bin/bash

# Inference with leave-one-camera-out

cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --dataset_type gehler \
    --camera_name Canon1D \
    --fold exclude \
    --dataset_dir /path/to/your/processed_dataset \
    --cache_dir /path/to/your/cache_directory \
    --output_dir /path/to/your/output_directory \
    --model_path /path/to/your/trained_model \
    --color_checker_path /path/to/your/color_checker.jpg \
    --batch_size 20