import argparse
import sys
import os
from datetime import datetime
from dataset import NUS8Dataset, GehlerDataset
from utils import (
    process_batch, calculate_statistics, 
    load_color_checker, save_results
)
from pipelines.inpaint_pipeline import DiffusionInpaintPipeline

def get_output_filename(dataset_type, camera_name, fold):
    """
    Generate descriptive output filename based on training/testing strategy
    
    Args:
        dataset_type: "nus8" or "gehler"
        camera_name: Camera name or None for cross-dataset evaluation
        fold: Fold number, "all", or "exclude"
        
    Returns:
        Descriptive filename string
        
    Examples:
        "eval_nus8_from_gehler"                      # camera_name=None, cross-dataset eval
        "nus8_train_exclude_SonyA57_test_SonyA57"    # camera_name="SonyA57", fold="exclude"  
        "gehler_train_fold12_test_fold0_Canon1D"     # camera_name="Canon1D", fold=0
    """
    
    if camera_name is None:
        # Cross-dataset evaluation: model trained on other dataset, evaluated on this one
        other_dataset = "gehler" if dataset_type.lower() == "nus8" else "nus8"
        return f"eval_{dataset_type.lower()}_from_{other_dataset}"
        
    # Single dataset experiments
    parts = [dataset_type.lower()]
        
    if fold == "exclude":
        # Model trained excluding specific camera, evaluated on that camera
        parts.extend([f"train_exclude_{camera_name}", f"test_{camera_name}"])
        
    elif isinstance(fold, int):
        # Test on specific fold, train on other folds for specific camera
        other_folds = [str(i) for i in range(3) if i != fold]  # assuming 3-fold CV
        train_folds = "".join(other_folds)
        parts.extend([f"train_fold{train_folds}", f"test_fold{fold}", camera_name])
        
    elif fold == "all":
        # Test on all folds for specific camera
        parts.extend([f"train_all", f"test_all", camera_name])
    
    return "_".join(parts)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Run inference on specified dataset')
    
    # Add dataset type and camera args
    parser.add_argument('--dataset_type', type=str, default="nus8", 
                        choices=["nus8", "gehler"],
                        help='Type of dataset ("nus8" or "gehler")')
    parser.add_argument('--camera_name', type=str, default=None,
                        help='Camera to use ("SonyA57", "Canon1D", "Canon5D", etc.)')
    parser.add_argument('--fold', type=str, default="all",
                        help='Fold to use (0, 1, 2, or "all", "exclude")')
    
    # Updated path arguments for new dataset structure
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Base directory containing dataset with images/ and masks/ subdirectories')
    parser.add_argument('--cache_dir', type=str, required=True, 
                        help='Directory for caching processed data')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model')
    parser.add_argument('--color_checker_path', type=str, required=True, 
                        help='Path to the color checker image')
    parser.add_argument('--batch_size', type=int, default=20, 
                        help='Batch size for processing')
    
    # Optional arguments for output control
    parser.add_argument('--output_prefix', type=str, default="",
                        help='Custom prefix for output filename')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Automatically construct metadata path
    metadata_path = os.path.join(args.dataset_dir, "all_cameras_meta.json")
    
    # Verify metadata file exists
    if not os.path.exists(metadata_path):
        print(f"❌ Error: Metadata file not found at {metadata_path}")
        print(f"   Please ensure all_cameras_meta.json exists in {args.dataset_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try to convert fold to int if it's a number
    try:
        fold = int(args.fold)
    except ValueError:
        fold = args.fold
    
    # Generate descriptive output filename
    output_filename = get_output_filename(args.dataset_type, args.camera_name, fold)
    
    # Add custom prefix if provided
    if args.output_prefix:
        output_filename = f"{args.output_prefix}_{output_filename}"
    
    print("=" * 60)
    print(f"📁 Output file: {output_filename}")
    print(f"📄 Metadata file: {metadata_path}")
    print("=" * 60)
    # Load appropriate dataset with updated parameters
    if args.dataset_type.lower() == "nus8":
        print(f"📊 Loading NUS8 dataset...")
        dataset = NUS8Dataset(
            dataset_dir=args.dataset_dir,
            metadata_path=metadata_path,
            type="test",
            cache_dir=args.cache_dir,
            camera_name=args.camera_name,
            folds=fold
        ).dataset
    else:  # gehler
        print(f"📊 Loading Gehler dataset...")
        dataset = GehlerDataset(
            dataset_dir=args.dataset_dir,
            metadata_path=metadata_path,
            type="test",
            cache_dir=args.cache_dir,
            camera_name=args.camera_name,
            folds=fold
        ).dataset
    
    print(f"✅ Dataset loaded: {len(dataset['train'])} samples")
    
    # Load model
    print(f"🤖 Loading model from: {args.model_path}")
    pipeline = DiffusionInpaintPipeline(args.model_path)
    
    # Load color checker
    print(f"🎨 Loading color checker from: {args.color_checker_path}")
    color_checker = load_color_checker(args.color_checker_path)
    
    # Process in batches
    print(f"🚀 Starting inference with batch size: {args.batch_size}")
    angular_errors = process_batch(
        dataset, pipeline, color_checker, args.dataset_dir,
        args.output_dir, args.batch_size, args.dataset_type
    )
    # statistics = calculate_statistics(angular_errors)
    # Save results
    print(f"💾 Saving results to: {output_filename}")
    save_results(angular_errors, args.output_dir, output_filename)
    
    print("✅ Inference completed successfully!")