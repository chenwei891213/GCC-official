from abc import ABC, abstractmethod
import os
import json
import random
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Union
from .utils import slice_list, save_dataset_json, set_seed
from datasets import load_dataset, Image, Features, Value, Sequence


class BaseAwbDataset(ABC):
    """Base class for auto white balance datasets"""

    def __init__(
        self,
        dataset_dir: str,
        metadata_path: str,
        cache_dir: Optional[str] = None,
        type: str = "train",
        camera_name: Optional[str] = None,
        folds: Union[int, str] = 0,
        num_folds: int = 3,
        seed: int = 42,
        prompt: str = "a scene with a color checker that accurately reflects the ambient lighting of the scene."
    ):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "images")
        self.mask_dir = os.path.join(dataset_dir, "mask")
        self.metadata_path = metadata_path
        self.cache_dir = cache_dir
        self.type = type
        self.camera_name = camera_name
        self.folds = folds
        self.num_folds = num_folds
        self.prompt = prompt

        set_seed(seed)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def load_ground_truth(self) -> Dict[str, Any]:
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        camera_data = {}
        for entry in metadata:
            camera_name = entry.get('camera_name')
            camera_data.setdefault(camera_name, []).append(entry)
        return camera_data

    @abstractmethod
    def get_image_mask_pairs(self, camera_data) -> List[Dict[str, Any]]:
        pass

    def prepare_dataset(self) -> Any:
        """Prepare dataset with caching and fold handling."""
        cache_path = self._get_cache_path()
        cache_hit = cache_path and os.path.exists(cache_path)

        # Determine camera usage description
        if self.camera_name is None:
            camera_info = "Using all cameras"
        elif self.folds == "exclude":
            camera_info = f"Excluding camera='{self.camera_name}'"
        else:
            camera_info = f"Using camera='{self.camera_name}'"

        fold_info = f"(fold={self.folds})"
        mode = self.type.upper()

        if cache_hit:
            print(f"[BaseAwbDataset] [{mode}] {camera_info} {fold_info} → Loading dataset from cache.")
            return self._load_dataset_from_json(cache_path)

        print(f"[BaseAwbDataset] [{mode}] {camera_info} {fold_info} → Building dataset from scratch.")

        # Build dataset
        data = self.get_image_mask_pairs(self.load_ground_truth())
        selected = self._split_folds(data)

        print(f"[BaseAwbDataset] → {len(selected)} samples selected.")

        if cache_path:
            save_dataset_json(selected, cache_path)
            return self._load_dataset_from_json(cache_path)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = os.path.join(tmp, self._get_cache_filename())
                save_dataset_json(selected, tmp_path)
                return self._load_dataset_from_json(tmp_path)


    def _split_folds(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.folds in ("all", "exclude"):
            return data_list
        splits = slice_list(data_list, [1] * self.num_folds)
        if self.type == "test":
            return splits[self.folds]
        else:
            return [item for i, split in enumerate(splits) if i != self.folds for item in split]

    def _get_cache_path(self) -> str:
        """Return the cache file path if cache_dir is set, otherwise None."""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, self._get_cache_filename())


    def _get_cache_filename(self) -> str:
        """Generate cache filename with explicit state representation"""
        
        # Handle different folding scenarios
        if self.camera_name is None:
            # No camera specified - use all cameras
            filename = f"{self.type}_all_cameras"
        elif self.folds == "exclude":
            # Exclude specific camera
            filename = f"{self.type}_exclude_{self.camera_name}"
        elif isinstance(self.folds, int):
            # Use specific camera with fold number
            filename = f"{self.type}_{self.camera_name}_{self.folds}fold"

        return f"{filename}.json"


    def _load_dataset_from_json(self, json_path: str) -> Any:
        features = self.get_features()
        dataset = load_dataset(
            'json',
            data_files={'train': json_path},
            features=features
        )
        print(
            f"[BaseAwbDataset] → Loaded {len(dataset['train'])} samples from '{json_path}'."
        )
        return dataset

    def get_features(self) -> Features:
        return Features({
            "image": Image(),
            "mask": Image(),
            "text": Value("string"),
            "illuminant": Sequence(Value("float64"), length=3),
            "cc_coords": Sequence(Value("float64"), length=4),
            "image_name": Value("string"),
        })