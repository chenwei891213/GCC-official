# Data Preprocessing Guide

Processes black level (darkness level) and saturation level correction for color constancy datasets. Extracts and saves metadata in JSON format with illuminant information.

## Supported Datasets

- [NUS-8 Dataset](http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html)
- [Gehler-Shi Dataset](https://www2.cs.sfu.ca/~colour/data/shi_gehler/)

## Dataset Format

### NUS-8 Dataset
```
dataset_root/
├── Camera1DsMkIII/
│   ├── PNG/
│   │   └── *.png
│   └── ground_truth/
│       └── Camera1DsMkIII_gt.mat
├── Canon600D/
│   ├── PNG/
│   │   └── *.png
│   └── ground_truth/
│       └── Canon600D_gt.mat
└── ... (other cameras)
```

### Gehler Dataset
```
dataset_root/
├── images/
│   └── *.png
├── coordinates/
│   └── *_macbeth.txt
└── ground_truth.mat
```

## Usage

To preprocess the datasets

```bash
# NUS-8 Dataset
python process_dataset.py \
    --dataset_type NUS8 \
    --dataset_root /path/to/nus8_dataset \
    --output_root /path/to/processed_nus8

# Gehler Dataset  
python process_dataset.py \
    --dataset_type Gehler \
    --dataset_root /path/to/gehler_dataset \
    --output_root /path/to/processed_gehler
```

## Output Structure

```
output_root/
├── images/        # Processed data
├── mask/          # Binary masks for ColorChecker locations
└── all_cameras_meta.json   # Metadata for all processed images
```