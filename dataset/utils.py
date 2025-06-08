import os
import random
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import numpy as np
import random
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def slice_list(lst, fractions):
    """
    Split a list according to specified proportions
    
    Args:
        lst: List to be split
        fractions: Weight for each portion
        
    Returns:
        List of split lists
    """
    sliced = []
    for i in range(len(fractions)):
        total_fraction = sum(fractions)
        start = int(round(1.0 * len(lst) * sum(fractions[:i]) / total_fraction))
        end = int(round(1.0 * len(lst) * sum(fractions[:i + 1]) / total_fraction))
        sliced.append(lst[start:end])
    return sliced

def save_dataset_json(data_list, output_path):
    """
    Save dataset to JSON file
    
    Args:
        data_list: Data list
        output_path: Output path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for item in data_list:
            f.write(json.dumps(item) + '\n')

def set_seed(seed=42):
    """
    Set random seed for reproducibility
        
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)