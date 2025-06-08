import os
import json
import random
from typing import List, Dict, Any, Optional, Union
from .base_dataset import BaseAwbDataset

class NUS8Dataset(BaseAwbDataset):
    """NUS8 dataset implementation
    
    This dataset contains images from 8 different camera models:
    - Canon1DsMkIII
    - Canon600D
    - NikonD5200
    - FujifilmXM1
    - OlympusEPL6
    - SonyA57
    - PanasonicGX1
    - SamsungNX2000
    
    Usage examples:
    1. Load all cameras: NUS8Dataset(..., camera_name=None, folds="all")
    2. Exclude specific camera: NUS8Dataset(..., camera_name="SonyA57", folds="exclude")
    3. Use specific camera with 3-fold: NUS8Dataset(..., camera_name="SonyA57", folds=0)
    """
    
    def __init__(
        self,
        dataset_dir: str,
        metadata_path: str,
        cache_dir: str = None,
        type: str = "train",
        camera_name: Optional[str] = None,
        folds: Union[int, str] = 0,
        num_folds: int = 3,
        seed: int = 42,
        prompt: str = "a scene with a color checker that accurately reflects the ambient lighting of the scene."
    ):
        """
        Initialize NUS8 dataset
        
        Args:
            dataset_dir: Base directory containing dataset with images/ and masks/ subdirectories
            metadata_path: Path to metadata JSON file with ground truth data
            cache_dir: Directory for caching the processed dataset
            type: Dataset type, either "train" or "test"
            camera_name: Camera to include/exclude. Options:
                         - None: Use all cameras
                         - One of the 8 camera models: Use only that camera
            folds: Controls dataset splitting. Options:
                   - "all": Return data from all cameras/folds
                   - "exclude": Exclude data from camera_name
                   - int (0,1,2): Use the specified fold index for the
                     selected camera (or all cameras if camera_name is None)
            num_folds: Number of folds to split the data into
            seed: Random seed for reproducibility
        """
        self.supported_cameras = [
            'Canon1DsMkIII', "Canon600D", "NikonD5200", "FujifilmXM1",
            'OlympusEPL6', 'SonyA57', "PanasonicGX1", "SamsungNX2000"
        ]
        
        super().__init__(
            dataset_dir, metadata_path, 
            cache_dir, type, camera_name, folds, num_folds, seed, prompt
        )
        self.dataset = self.prepare_dataset()
    
    def get_image_mask_pairs(self, camera_data) -> List[Dict[str, Any]]:
        """Get pairs of images and masks for NUS8 dataset"""
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
        
    # def get_image_names(self):
    #     """Get all image names in the dataset"""
    #     if hasattr(self, 'dataset') and self.dataset:
    #         if isinstance(self.dataset, dict) and 'train' in self.dataset:
    #             return self.dataset["train"]["image_name"]
    #         return self.dataset["image_name"]
    #     return []


# dataset = NUS8Dataset(
#     dataset_dir="/project2/stevech/GCC/NUS8",
#     metadata_path="/project2/stevech/GCC/NUS8/all_cameras_meta.json",
#     cache_dir="/project2/stevech/GCC/NUS8/cache",
#     type="train",
#     # camera_name="SonyA57",
#     folds="all",
#     num_folds=3,
#     seed=42,
#     prompt="a scene with a color checker that accurately reflects the ambient lighting of the scene."
# )