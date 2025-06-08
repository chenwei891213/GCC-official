import os
import random
import json
from typing import List, Dict, Any, Optional, Union
from .base_dataset import BaseAwbDataset

class GehlerDataset(BaseAwbDataset):
    """Gehler dataset implementation
    
    This dataset contains images from two camera models:
    - Canon 1D (filenames starting with "8D5U")
    - Canon 5D (filenames starting with "IMG_")
    
    Usage examples:
    1. Load all cameras: GehlerDataset(..., camera_name=None, folds="all")
    2. Use only Canon 1D: GehlerDataset(..., camera_name="Canon1D", folds=0)
    3. Use only Canon 5D: GehlerDataset(..., camera_name="Canon5D", folds=0)
    4. Exclude Canon 1D: GehlerDataset(..., camera_name="Canon1D", folds="exclude")
    """
    
    def __init__(
        self,
        dataset_dir: str,
        metadata_path: str,
        cache_dir: str = None,
        type: str = "train",
        camera_name: str = None,  # 'Canon1D' or 'Canon5D'
        folds: Union[int, str] = 0,
        num_folds: int = 3,
        seed: int = 42,
        prompt: str = "a scene with a color checker that accurately reflects the ambient lighting of the scene."
    ):
        """
        Initialize Gehler dataset
        
        Args:
            dataset_dir: Base directory containing dataset with images/ and masks/ subdirectories
            metadata_path: Path to metadata JSON file with ground truth data
            cache_dir: Directory for caching the processed dataset
            type: Dataset type, either "train" or "test"
            camera_name: Camera model to use or exclude. Options:
                        - None: Use both cameras
                        - "Canon1D": Use/exclude Canon 1D (8D5U prefix)
                        - "Canon5D": Use/exclude Canon 5D (IMG_ prefix)
            folds: Controls dataset splitting. Options:
                   - "all": Return data from all cameras/folds
                   - "exclude": Exclude data from camera specified by camera_name
                   - int (0,1,2): Use the specified fold index
            num_folds: Number of folds to split the data into
            seed: Random seed for reproducibility
        """
        # Mapping between camera names and filename prefixes
        self.camera_map = {
            "Canon1D": "8D5U",
            "Canon5D": "IMG_"
        }
        self.supported_cameras = list(self.camera_map.keys())
        
        super().__init__(
            dataset_dir, metadata_path, 
            cache_dir, type, camera_name, folds, num_folds, seed, prompt
        )
        self.dataset = self.prepare_dataset()
    
    def get_image_mask_pairs(self, camera_data) -> List[Dict[str, Any]]:
        """Get pairs of images and masks for Gehler dataset
        
        This method uses the metadata.json format to match images with their ground truth.
        """
        
        def _add_entry_to_list(data_list, entry):
            """Helper function to add entry to data list if files exist"""
            image_file = entry['filename']
            image_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(self.mask_dir, image_file)
            
            # if os.path.exists(image_path) and os.path.exists(mask_path):
            data_list.append({
                "image": image_path,
                "mask": mask_path,
                "text": self.prompt,
                "illuminant": entry['illuminant'],
                "cc_coords": entry['cc_coord'],
                "image_name": image_file,
            })
        
        # If no camera_name specified, return all data
        if self.camera_name is None:
            data_list = []
            for camera_name, entries in camera_data.items():
                for entry in entries:
                    _add_entry_to_list(data_list, entry)
            random.shuffle(data_list)
            return data_list
        
        # Handle cases with specified camera_name
        if self.camera_name not in self.supported_cameras:
            raise ValueError(f"Camera {self.camera_name} not supported. Choose from: {self.supported_cameras}")
        
        # Handle camera filtering logic
        if self.folds == "exclude":
            if self.type == "train":
                # Train mode: exclude specified camera data
                filtered_cameras = {cam: data for cam, data in camera_data.items() 
                                if cam != self.camera_name}
            else:  # test mode
                # Test mode: include only specified camera data
                filtered_cameras = {cam: data for cam, data in camera_data.items() 
                                if cam == self.camera_name} if self.camera_name in camera_data else {}
        else:
            # Other cases (including numeric folds): use only specified camera data
            filtered_cameras = {cam: data for cam, data in camera_data.items() 
                            if cam == self.camera_name} if self.camera_name in camera_data else {}
        
        # Build data list
        data_list = []
        for camera_name, entries in filtered_cameras.items():
            for entry in entries:
                _add_entry_to_list(data_list, entry)
        random.shuffle(data_list)
        return data_list
# dataset = GehlerDataset(
#     dataset_dir="/project2/stevech/GCC/Gehler",
#     metadata_path="/project2/stevech/GCC/Gehler/all_cameras_meta.json",
#     cache_dir="/project2/stevech/GCC/Gehler/cache",
#     type="test",
#     camera_name="Canon1D",  # or "Canon5D" or None for both
#     folds=0,
#     num_folds=3,
#     seed=42,
#     prompt="a scene with a color checker that accurately reflects the ambient lighting of the scene."
# )