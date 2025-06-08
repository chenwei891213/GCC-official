#!/bin/bash

# Inference on full dataset (all cameras included)

cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --dataset_type gehler \
    --dataset_dir /path/to/your/processed_dataset \
    --cache_dir /path/to/your/cache_directory \
    --output_dir /path/to/your/output_directory \
    --model_path /path/to/your/trained_model \
    --color_checker_path color_chart.jpg \
    --batch_size 20