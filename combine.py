import json
import os
import numpy as np
from statistics import mean, median
import glob
import argparse
from utils import calculate_angular_error, calculate_statistics

def load_and_process_json_files(folder_path):
    """Load all JSON files from folder and calculate angular errors"""
    all_angular_errors = {}
    
    # Get all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return None
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each image's data
            for image_name, image_data in data.items():
                if 'estimated' in image_data and 'ground_truth' in image_data:
                    estimated = np.array(image_data['estimated'])
                    ground_truth = np.array(image_data['ground_truth'])
                    
                    # Calculate angular error
                    angular_error = calculate_angular_error(ground_truth, estimated)
                    
                    # Store result
                    unique_key = f"{os.path.basename(json_file)}_{image_name}"
                    all_angular_errors[unique_key] = {
                        'error': angular_error
                    }
                    
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
            continue
    
    return all_angular_errors

def main():
    parser = argparse.ArgumentParser(description='Calculate angular errors from JSON files')
    parser.add_argument('--folder_path', help='Path to folder containing JSON files')
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder_path):
        print(f"Folder {args.folder_path} does not exist")
        return
    
    print(f"Processing folder: {args.folder_path}")
    
    # Load and process all JSON files
    angular_errors = load_and_process_json_files(args.folder_path)
    
    if angular_errors is None or len(angular_errors) == 0:
        print("No valid data found")
        return
    
    print(f"Successfully processed {len(angular_errors)} images")
    
    # Calculate and display statistics
    calculate_statistics(angular_errors)

if __name__ == "__main__":
    main()