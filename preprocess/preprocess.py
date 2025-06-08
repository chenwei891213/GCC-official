import cv2
import pickle as pickle
import scipy.io
import numpy as np
import os
import sys
import random
from PIL import Image
import argparse
import json

class NUS8Dataset:
    def __init__(self, dataset_root, output_root):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.camera_names = [
            'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 'NikonD5200',
            'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57'
        ]
        self.all_meta_data = []
        
    def setup_directories(self):
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
            
        # Create common output directories
        self.mask_output_dir = os.path.join(self.output_root, 'mask')
        self.img_output_dir = os.path.join(self.output_root, 'images')
        
        if not os.path.exists(self.mask_output_dir):
            os.makedirs(self.mask_output_dir)
        if not os.path.exists(self.img_output_dir):
            os.makedirs(self.img_output_dir)
            
    def process_camera(self, camera_name):
        ground_truth = scipy.io.loadmat(os.path.join(self.dataset_root, camera_name, "ground_truth", f"{camera_name}_gt.mat"))
        filenames = sorted(os.listdir(os.path.join(self.dataset_root, camera_name, "PNG")))
        
        # Get all the metadata
        illuminants = ground_truth['groundtruth_illuminants']
        darkness_level = ground_truth['darkness_level']
        saturation_level = ground_truth['saturation_level']
        cc_coords = ground_truth['CC_coords']
        
        # Normalize illuminants
        illuminants = illuminants / np.linalg.norm(illuminants, axis=1)[..., np.newaxis]
        
        for idx, file in enumerate(filenames):
            print(f"Processing {camera_name} file: {file} ({idx+1}/{len(filenames)})")
            
            # Create individual meta data entry for each image
            meta_entry = {
                'camera_name': camera_name,
                'illuminant': illuminants[idx].tolist(),
                'darkness_level': darkness_level.tolist(),
                'saturation_level': saturation_level.tolist(),
                'cc_coord': cc_coords[idx].tolist(),
                'filename': os.path.basename(file),
            }
            
            # Add to flat list
            self.all_meta_data.append(meta_entry)
            
            # Process and save the image
            self.process_image(camera_name, file, darkness_level, cc_coords, idx)
    
    def process_image(self, camera_name, file, darkness_level, cc_coords, index):
        # Read raw image
        file_path = os.path.join(self.dataset_root, camera_name, "PNG", file)
        raw = np.array(cv2.imread(file_path, -1), dtype='float32')
        raw = np.maximum(raw - darkness_level, [0, 0, 0])

        # Process image
        img = (raw/raw.max() * 65535.0).astype(np.uint16)
        img = np.clip(img, 0, 65535).astype(np.uint16)
        index = int(file.split('_')[1].split('.')[0]) - 1
        
        # Get cc_coords for current image
        cc_coord = cc_coords[index]
        
        # Create mask from cc_coords
        y1, y2, x1, x2 = cc_coord
        mask = np.zeros(raw.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        cv2.imwrite(os.path.join(self.mask_output_dir, file), mask)
        cv2.imwrite(os.path.join(self.img_output_dir, file), img)
    
    def save_metadata(self):
        with open(os.path.join(self.output_root, 'all_cameras_meta.json'), 'w') as f:
            json.dump(self.all_meta_data, f, indent=4)
    
    def process_dataset(self):
        print("Start processing")
        self.setup_directories()
        
        for camera_name in self.camera_names:
            self.process_camera(camera_name)
            
        self.save_metadata()
        print("Finish processing NUS8 dataset")

class GehlerDataset:
    def __init__(self, dataset_root, output_root):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.all_meta_data = []
        
    def get_directory(self):
        return self.dataset_root + '/'
        
    def get_img_directory(self):
        return self.dataset_root
        
    def setup_directories(self):
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
            
        # Create common output directories
        self.mask_output_dir = os.path.join(self.output_root, 'mask')
        self.img_output_dir = os.path.join(self.output_root, 'images')
        
        if not os.path.exists(self.mask_output_dir):
            os.makedirs(self.mask_output_dir)
        if not os.path.exists(self.img_output_dir):
            os.makedirs(self.img_output_dir)
    
    def get_mcc_coord(self, fn):
        # Note: relative coord
        with open(self.get_directory() + 'coordinates/' + fn.split('.')[0] + '_macbeth.txt', 'r') as f:
            lines = f.readlines()
            width, height = map(float, lines[0].split())
            scale_x = 1 / width
            scale_y = 1 / height
            
            # Read all four corner points
            lines = [lines[1], lines[2], lines[4], lines[3]]
            polygon = []
            for line in lines:
                line = line.strip().split()
                x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
                polygon.append((x, y))
            
            # Convert to numpy array for easier manipulation
            polygon = np.array(polygon, dtype='float32')
            
            # Calculate bounding box coordinates (min/max of x and y)
            # This ensures the original polygon area is fully contained
            x_min = np.min(polygon[:, 0])
            y_min = np.min(polygon[:, 1])
            x_max = np.max(polygon[:, 0])
            y_max = np.max(polygon[:, 1])
            
            # Convert to absolute coordinates for the image
            img_path = self.get_img_directory() + '/images/' + fn
            img = cv2.imread(img_path, -1)
            height, width = img.shape[:2]
            
            # Make sure coordinates are within image boundaries (0 to width/height)
            y1 = max(0, int(y_min * height))
            y2 = min(height, int(y_max * height))
            x1 = max(0, int(x_min * width))
            x2 = min(width, int(x_max * width))
            
            return [y1, y2, x1, x2] 
        
    def load_image(self, fn):
        file_path = self.get_img_directory() + '/images/' + fn
        raw = np.array(cv2.imread(file_path, -1), dtype='float32')
        
        # Set black level based on camera
        if fn.startswith('IMG'):
            # Canon 5D images
            black_level = 129
            camera_name = 'Canon5D'
        else:
            # Canon 1D images
            black_level = 1
            camera_name = 'Canon1D'
            
        raw = np.maximum(raw - black_level, [0, 0, 0])
        return raw, black_level, camera_name
    
    def create_mask(self, img_shape, cc_coord):
        # Create mask from cc_coords (y1, y2, x1, x2)
        height, width = img_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Unpack coordinates
        y1, y2, x1, x2 = cc_coord
        
        # Fill rectangle
        mask[y1:y2, x1:x2] = 255
        
        # Check if mask area is empty
        if np.sum(mask) == 0:
            print(f"Warning: Empty mask created with coordinates {cc_coord}")
            print(f"Image shape: {img_shape}")
        
        return mask
    
    def process_image(self, filename, illuminant):
        print(f"Processing file: {filename}")
        
        # Get MCC coordinates in NUS-8 format (y1, y2, x1, x2)
        cc_coord = self.get_mcc_coord(filename)
        
        # Load and process image
        raw, black_level, camera_name = self.load_image(filename)
        
        # Create mask from cc_coords
        mask = self.create_mask(raw.shape, cc_coord)
        
        # Process image
        img = (raw/raw.max() * 65535.0).astype(np.uint16)
        img = np.clip(img, 0, 65535).astype(np.uint16)
        
        # Create metadata entry
        meta_entry = {
            'filename': filename,
            'camera_name': camera_name,
            'illuminant': illuminant.tolist(),
            'darkness_level': black_level,
            'cc_coord': cc_coord
        }
        
        # Add to metadata list
        self.all_meta_data.append(meta_entry)
        
        # Save mask
        cv2.imwrite(os.path.join(self.mask_output_dir, filename), mask)
        cv2.imwrite(os.path.join(self.img_output_dir, filename), img)
    
    def save_metadata(self):
        with open(os.path.join(self.output_root, 'all_cameras_meta.json'), 'w') as f:
            json.dump(self.all_meta_data, f, indent=4)
    
    def process_dataset(self):
        print("Start processing Gehler dataset")
        self.setup_directories()
        
        # Load ground truth illuminants
        ground_truth = scipy.io.loadmat(self.get_directory() + 'ground_truth.mat')['real_rgb']
        ground_truth /= np.linalg.norm(ground_truth, axis=1)[..., np.newaxis]
        
        # Get all image filenames
        image_files = sorted(os.listdir(self.get_img_directory() + '/images'))
        
        # Process each image
        for idx, filename in enumerate(image_files):
            print(f"Processing image {idx+1}/{len(image_files)}")
            self.process_image(filename, ground_truth[idx])
            
        self.save_metadata()
        print("Finish processing Gehler dataset")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset.')
    parser.add_argument('--dataset_type', type=str, choices=['NUS8', 'Gehler'], required=True,
                        help='which dataset to process: NUS8 or Gehler')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='root directory of the dataset')
    parser.add_argument('--output_root', type=str, required=True,
                        help='root directory to save processed images')
    args = parser.parse_args()
    
    if args.dataset_type == 'NUS8':
        dataset = NUS8Dataset(args.dataset_root, args.output_root)
        dataset.process_dataset()
    elif args.dataset_type == 'Gehler':
        dataset = GehlerDataset(args.dataset_root, args.output_root)
        dataset.process_dataset()
