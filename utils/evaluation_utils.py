import os
import json
from pathlib import Path
from functools import partial
from statistics import mean, median

import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader


# ===============================
# Image Processing Functions
# ===============================

def gamma_correction(image, gamma=2.2):
    """
    Apply gamma correction to an image with automatic bit-depth detection
    
    Args:
        image: Input image (PIL Image or numpy array)
        gamma: Gamma correction value (default: 2.2)
    
    Returns:
        Gamma-corrected image as uint8 array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Determine max value based on bit depth
    if image.dtype == np.uint16:
        max_val = 65535.0
    elif image.dtype == np.uint8:
        max_val = 255.0
    else:
        raise ValueError(f"Unsupported image data type: {image.dtype}")
    
    # Make a copy to avoid modifying the original
    img_array = image.astype(np.float32)
    
    # Apply gamma correction
    img_array = img_array / max_val
    img_corrected = np.power(img_array, 1.0/gamma)
    img_corrected = np.clip(img_corrected * 255.0, 0, 255).astype(np.uint8)
    
    return img_corrected


def apply_awb_raw(image, rgb_illuminant):
    """
    Apply automatic white balance to RAW image.
    
    Args:
        image: Input image already processed with black/saturation level normalization 
               and scaled to [0, 65535] range
        rgb_illuminant: RGB illuminant values for white balance correction
    
    Returns:
        Balanced image in uint8 format
    """
    illuminant = np.array(rgb_illuminant, dtype=float)
    
    # Calculate white balance gains using green channel as reference
    gains = illuminant[1] / illuminant
    
    # Apply white balance correction
    balanced = image.astype(float)
    for i in range(3):
        balanced[:,:,i] = balanced[:,:,i] * gains[i]
    
    # Clip to valid range to prevent overflow
    balanced = np.clip(balanced, 0, 65535)

    normalized = balanced / 65535.0
    gamma_corrected = np.power(normalized, 1/2.2)
    balanced = (gamma_corrected * 255).astype(np.uint8)
    
    return balanced


# ===============================
# Color Checker Functions
# ===============================

def load_color_checker(path, width=180, height=135):
    """Load and resize color checker"""
    color_checker = cv2.imread(path)
    if color_checker is None:
        raise ValueError(f"Could not load color checker from {path}")
    return cv2.resize(color_checker, (width, height))


def get_safe_position(center, rect_width, rect_height, image_size, padding=3):
    """Get safe position for placing a color checker"""
    x, y = center
    padded_width = rect_width + padding * 2
    padded_height = rect_height + padding * 2
    
    x1 = x - padded_width // 2
    y1 = y - padded_height // 2
    x2 = x1 + padded_width
    y2 = y1 + padded_height
    
    # Ensure position stays within image bounds
    if x1 < 0:
        x1 = 0
        x2 = padded_width
    if x2 > image_size[1]:
        x2 = image_size[1]
        x1 = image_size[1] - padded_width
    if y1 < 0:
        y1 = 0
        y2 = padded_height
    if y2 > image_size[0]:
        y2 = image_size[0]
        y1 = image_size[0] - padded_height
    
    mask_coords = (x1, y1, x2, y2)
    checker_coords = (x1 + padding, y1 + padding, x2 - padding, y2 - padding)
    
    return mask_coords, checker_coords


# ===============================
# Illuminant Estimation Functions
# Adapted from colour-science library: https://github.com/colour-science/colour-checker-detection/blob/develop/colour_checker_detection/detection/common.py#L998
# ===============================

def _create_swatch_masks(width, height, swatches_h, swatches_v, samples):
    """Helper function to create swatch masks"""
    samples_half = max(samples / 2, 1)
    masks = []
    offset_h = width / swatches_h / 2
    offset_v = height / swatches_v / 2
    
    for j in np.linspace(offset_v, height - offset_v, swatches_v):
        for i in np.linspace(offset_h, width - offset_h, swatches_h):
            masks.append(np.array([
                j - samples_half,
                j + samples_half,
                i - samples_half,
                i + samples_half,
            ], dtype=np.int32))
    
    return np.array(masks, dtype=np.int32)


def _extract_swatch_colors(image, masks):
    """Helper function to extract swatch colors from masks"""
    return np.array([
        np.mean(image[mask[0]:mask[1], mask[2]:mask[3], ...], axis=(0, 1))
        for mask in masks
    ], dtype=np.float32)


def estimate_illuminant(image: np.ndarray, checker_pos: tuple) -> np.ndarray:
    """
    Estimate illuminant color from a ColorChecker Classic chart in the image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.
    checker_pos : tuple
        Bounding box of the color checker in (x1, y1, x2, y2) format.

    Returns
    -------
    np.ndarray
        Estimated RGB illuminant from the gray patch.
    """
    # Unpack bounding box
    x1, y1, x2, y2 = checker_pos

    # Define the 4 corners of the detected checker
    coords_pixel = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
    ], dtype=np.float32)

    # Standard working size
    working_width = int(x2 - x1)
    working_height = int(y2 - y1)
    samples = int(working_width / 15)  # adjustable sample size

    # Destination rectangle (warped top-down view)
    rectangle = np.array([
        [0, 0],
        [working_width, 0],
        [working_width, working_height],
        [0, working_height],
    ], dtype=np.float32)

    # Perspective warp
    M = cv2.getPerspectiveTransform(coords_pixel, rectangle)
    warped = cv2.warpPerspective(image, M, (working_width, working_height), flags=cv2.INTER_CUBIC)

    # Generate swatch masks and extract RGB colors
    masks = _create_swatch_masks(working_width, working_height, 6, 4, samples)
    colours = _extract_swatch_colors(warped, masks)

    # Use 20th patch (index 19) which corresponds to a gray patch
    estimated_illuminant = colours[19]

    return estimated_illuminant


# ===============================
# Error Calculation Functions
# ===============================

def calculate_angular_error(true_ill, est_ill): 
    """Calculate angular error between two illuminants"""
    # Normalize vectors
    true_normalized = true_ill / (np.linalg.norm(true_ill) + 1e-8)
    est_normalized = est_ill / (np.linalg.norm(est_ill) + 1e-8)
    
    # Calculate dot product
    cos_theta = np.clip(np.dot(true_normalized, est_normalized), -1.0, 1.0)
    
    # Convert to angle
    return np.arccos(cos_theta) * 180 / np.pi


def calculate_statistics(angular_errors):
    """Calculate statistics from angular errors"""
    values = [data['error'] for data in angular_errors.values()]
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Calculate quartiles
    q1 = np.percentile(values, 25)
    q2 = median(values)  # Median
    q3 = np.percentile(values, 75)
    
    # Calculate mean
    mean_value = mean(values)
    
    # Calculate tri-mean
    tri_mean = (q1 + 2 * q2 + q3) / 4
    
    # Calculate best 25% and worst 25%
    best_25_percent = sorted_values[:max(1, n // 4)]
    worst_25_percent = sorted_values[-max(1, n // 4):]
    
    
    stats = {
        "Mean": mean_value,
        "Median": q2,
        "Tri-mean": tri_mean,
        "Best 25% (mean)": mean(best_25_percent),
        "Worst 25% (mean)": mean(worst_25_percent),
    }
    
    # Print statistics
    for name, value in stats.items():
        print(f"{name}: {value:.2f}")
    
    return stats


# ===============================
# Dataset Processing Functions
# ===============================

def preprocess(examples, color_checker=None, dataset_dir=None):
    """
    Dataset transformation function for training
    
    Args:
        examples: Dataset samples
        color_checker: Color checker image
        dataset_dir: Directory containing the dataset
    
    Returns:
        Processed samples with both gamma-corrected and raw images
    """
    # Get images and related data
    images = examples["image"]
    masks = examples["mask"] if "mask" in examples else None
    illuminants = examples["illuminant"]
    orig_image_names = examples["image_name"]
    cc_coords = examples.get("cc_coords", None)
    
    # Containers for processed data
    processed_images = []
    processed_masks = []
    raw_images = []  # Store pre-gamma images for AWB
    prompts = []
    checker_positions = []
    illuminant_list = []
    image_name_list = []
    
    for i, (image, illuminant, image_name) in enumerate(zip(images, illuminants, orig_image_names)):
        image_path = os.path.join(dataset_dir, "images", image_name)
        image_np = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        
        # Resize to 512x512 (before gamma correction)
        if image_np.shape[:2] != (512, 512):
            image_np_resized = cv2.resize(image_np, (512, 512))
        else:
            image_np_resized = image_np.copy()
        
        # Store raw image (pre-gamma) for AWB
        raw_images.append(image_np_resized)
        
        # Apply gamma correction for model input
        image_gamma = gamma_correction(image_np_resized)
        
        # Calculate image center based on actual dimensions
        center = (image_gamma.shape[1] // 2, image_gamma.shape[0] // 2)  # (width/2, height/2)
        mask_pos, checker_pos = get_safe_position(
            center, color_checker.shape[1], color_checker.shape[0], 
            image_gamma.shape[:2], padding=3
        )
        
        # Create new mask
        new_mask = np.zeros((512, 512), dtype=np.uint8)
        new_mask[mask_pos[1]:mask_pos[3], mask_pos[0]:mask_pos[2]] = 255
        
        # Place color checker on gamma-corrected image
        image_with_checker = image_gamma.copy()
        image_with_checker[checker_pos[1]:checker_pos[3], checker_pos[0]:checker_pos[2]] = color_checker
        
        # Convert back to PIL image (post-gamma for model input)
        image_pil = Image.fromarray(cv2.cvtColor(image_with_checker, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(new_mask)
        
        processed_images.append(image_pil)
        processed_masks.append(mask_pil)
        checker_positions.append(checker_pos)
        image_name_list.append(image_name)
        illuminant_list.append(illuminant)

        # Add prompt
        prompts.append("a scene with a color checker that accurately reflects the ambient lighting of the scene.")
    
    # Update processed samples
    examples["image"] = processed_images  # Post-gamma images for model
    examples["raw_image"] = raw_images    # Pre-gamma images for AWB
    examples["mask"] = processed_masks
    examples["prompt"] = prompts
    examples["cc_coords"] = checker_positions
    examples["image_name"] = image_name_list
    examples["gt_illuminant"] = illuminant_list
    
    return examples


def collate_fn(batch):
    """Batch collation function"""
    images = []
    raw_images = []
    masks = []
    prompts = []
    cc_coords = []
    image_names = []
    gt_illuminants = []
    
    for item in batch:
        images.append(item["image"])
        masks.append(item["mask"])
        prompts.append(item["prompt"])
        cc_coords.append(item["cc_coords"])
        image_names.append(item["image_name"])
        gt_illuminants.append(item["gt_illuminant"])
        raw_images.append(item["raw_image"])
    
    return images, raw_images, masks, prompts, cc_coords, image_names, gt_illuminants


def process_batch(dataset, pipeline, color_checker, dataset_dir, output_dir, batch_size=4, dataset_type="nus8"):
    """Process a dataset in batches using DataLoader"""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset = dataset["train"].with_transform(
        partial(preprocess, color_checker=color_checker, dataset_dir=dataset_dir)
    )

    dataloader = DataLoader(dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    
    angular_errors = {}
        
    for batch in dataloader:
        images, raw_images, masks, prompts, cc_coords, image_names, gt_illuminants = batch

        output_images = pipeline.run_inference(images, masks, prompts)
        
        for idx, output_image in enumerate(output_images):
            image_name = image_names[idx]
            gt_illuminant = np.array(gt_illuminants[idx])
            checker_pos = cc_coords[idx]
            # output_path = os.path.join(output_dir, f"output_{image_name}")
            # output_image_bgr = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            
            # # Save output image
            # cv2.imwrite(str(output_path), output_image_bgr)
            
            # Process output image to get estimated illuminant
            output_array = gamma_correction(output_image, 1/2.2)
            output_array = (output_array / 257).astype(np.float32)
            
            # Estimate illuminant from output
            estimated_illuminant = estimate_illuminant(
                output_array, checker_pos
            )
            
            # Apply white balance to raw input image using estimated illuminant
            raw_image = raw_images[idx]  # Get corresponding raw image
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
            # Apply AWB and get uint8 result directly
            wb_raw_image = apply_awb_raw(raw_image, estimated_illuminant)
            
            # # Save white balanced raw image
            wb_raw_path = os.path.join(output_dir, f"wb_raw_{image_name}")
            wb_raw_image = cv2.cvtColor(wb_raw_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
            cv2.imwrite(str(wb_raw_path), wb_raw_image)
            
            # Calculate angular error
            # angular_error = calculate_angular_error(gt_illuminant, estimated_illuminant)
            
            # Save results
            angular_errors[image_name] = {
                'estimated': estimated_illuminant.tolist(),
                'ground_truth': gt_illuminant.tolist()
            }
            
            print(f"Processed {image_name}")

    return angular_errors


# ===============================
# Results Management Functions
# ===============================

def save_results(angular_errors, output_dir, filename_prefix):
    """Save results to files with consistent naming"""
    # Save angular errors
    angular_errors_path = os.path.join(output_dir, f"{filename_prefix}_angular_errors.json")
    with open(angular_errors_path, 'w') as f:
        json.dump(angular_errors, f, indent=4)
    
    # Save statistics
    # with open(os.path.join(output_dir, "statistics.json"), 'w') as f:
    #     json.dump(statistics, f, indent=4)