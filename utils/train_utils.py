import numpy as np
import random
from PIL import Image

def generate_color_matrix(color_strength=0.8, color_offdiag=0.2):
    """
    Generate a color transformation matrix
    
    Args:
        color_strength (float): Strength of change for diagonal elements
        color_offdiag (float): Strength of change for off-diagonal elements
        
    Returns:
        numpy.ndarray: 3x3 color transformation matrix
    """
    color_aug = np.zeros(shape=(3, 3))
    
    for i in range(3):
        color_aug[i, i] = 1 + np.random.random() * color_strength - 0.5 * color_strength

        for j in range(3):
            if i != j:
                color_aug[i, j] = (np.random.random() - 0.5) * color_offdiag
                
    return color_aug

class ColorGammaTransform:
    def __init__(self, color_prob=0.5, color_strength=0.1, color_offdiag=0.0, gamma=2.2):
        """
        PIL-based probabilistic color enhancement and gamma correction.
        
        Args:
            color_prob (float): Probability of applying color matrix transformation [0-1]
            color_strength (float): Strength of change for diagonal elements
            color_offdiag (float): Strength of change for off-diagonal elements
            gamma (float): Parameter for gamma correction
        """
        self.color_prob = color_prob
        self.color_strength = color_strength
        self.color_offdiag = color_offdiag
        self.gamma = gamma

    def __call__(self, img):
        """
        Apply color enhancement based on probability and gamma correction to the input PIL image.
        
        Args:
            img (PIL.Image): Input PIL image
        Returns:
            PIL.Image: Transformed PIL image
        """
        # Input validation - ensure we're working with PIL images
        if not isinstance(img, Image.Image):
            raise TypeError("ColorGammaTransform expected PIL.Image input")

        # Convert PIL image to numpy array (0-255) and normalize to [0-1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Check channels
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image, but got shape {img_array.shape}")
            
        # Decide whether to apply color transformation based on probability
        apply_color = random.random() < self.color_prob
        
        if apply_color and (self.color_strength != 0 or self.color_offdiag != 0):
            # Generate color enhancement matrix
            color_matrix = generate_color_matrix(self.color_strength, self.color_offdiag)
            
            # Apply color transformation
            h, w, c = img_array.shape
            img_reshaped = img_array.reshape(h * w, 3).T  # Transpose to get (3, H*W)
            img_enhanced = np.dot(color_matrix, img_reshaped)
            img_array = img_enhanced.T.reshape(h, w, 3)  # Transpose back and reshape
            
            # Clamp values after color enhancement
            img_array = np.clip(img_array, 0, 1)
        
        # Always apply gamma correction
        img_array = np.power(img_array, 1 / self.gamma)
        
        # Final clamp and convert back to uint8 (0-255)
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        
        # Convert back to PIL image
        transformed_img = Image.fromarray(img_array)
        
        return transformed_img

class MaskedAugmentation:
    def __init__(self):
        self.augmentations = [
            self.random_brightness,
            self.random_contrast,
            self.random_saturation,
            # self.random_noise,
            
        ]
    
    def apply_augmentation(self, image, mask, prob=1):
        """
        Apply random data augmentation to image
        
        Args:
            image: PIL Image
            mask: PIL Image (binary image, 0 for background, 1 for foreground)
            prob: Probability of applying each augmentation method
        
        Returns:
            Augmented PIL Image
        """
        # Convert to numpy array for easier processing
        img_array = np.array(image)
        mask_array = np.array(mask)
    
        aug_funcs = self.augmentations.copy()
        random.shuffle(aug_funcs)
        # Randomly select and apply augmentation methods
        for aug_func in aug_funcs:
            if random.random() < prob:
                img_array = aug_func(img_array, mask_array)
        
        return Image.fromarray(img_array)
    
    def random_brightness(self, img, mask, brightness_range=(0.8, 2.0)):
        """Randomly adjust brightness"""
        factor = random.uniform(*brightness_range)
        mask_region = mask > 0
        
        result = img.copy()
        # Convert PIL Image method to numpy operations
        result[mask_region] = np.clip(img[mask_region] * factor, 0, 255).astype(np.uint8)
        
        return result
    
    def random_saturation(self, img, mask, saturation_range=(0.8, 1.4)):
        """Randomly adjust saturation"""
        factor = random.uniform(*saturation_range)
        mask_region = mask > 0
        
        img_hsv = np.array(Image.fromarray(img).convert('HSV'))
        result = img_hsv.copy()
        
        s = result[:,:,1]
        s[mask_region] = np.clip(s[mask_region] * factor, 0, 255).astype(np.uint8)
        result[:,:,1] = s
        
        return np.array(Image.fromarray(result, mode='HSV').convert('RGB'))
    
    def random_contrast(self, img, mask, contrast_range=(0.8, 1.4)):
        """Randomly adjust contrast"""
        factor = random.uniform(*contrast_range)
        mask_region = mask > 0

        result = img.copy()
        pixels = img[mask_region] 
        mean = np.mean(pixels, axis=0)

        adjusted = (pixels - mean) * factor + mean
        result[mask_region] = np.clip(adjusted, 0, 255).astype(np.uint8)

        return result

    
    
class RandomCropInPolygon(object):
    """
    Randomly crop the image within the non-zero regions of the mask.
    
    Args:
        crop_size (float): Minimum size ratio (0-1) of the original image to be used for cropping
        p (float, optional): Probability of applying the transform. Default: 0.5
    """
    
    def __init__(self, crop_size=0.5, crop_prob=0.5):
        self.crop_size = crop_size
        self.crop_prob = crop_prob
        
    def __call__(self, image, mask):
        """
        Args:
            image (PIL.Image): Input image
            mask (PIL.Image): Binary mask image
            
        Returns:
            tuple: Cropped (image, mask) pair
        """
        if random.random() < self.crop_prob:
            # Calculate crop dimensions based on the original image size
            width, height = image.size
            crop_w = random.randint(int(width * self.crop_size), width)
            crop_h = random.randint(int(height * self.crop_size), height)
            
            return self._random_crop_in_polygon(image, mask, (crop_w, crop_h))
        
        return image, mask
    
    def _random_crop_in_polygon(self, image, mask, crop_size):
        """
        Randomly crop within the non-zero regions of the mask.
        
        Args:
            image (PIL.Image): Original image
            mask (PIL.Image): Binary mask image
            crop_size (tuple): Crop size (width, height)
            
        Returns:
            tuple: Cropped (image, mask) pair
        """
        crop_w, crop_h = crop_size
        img_w, img_h = image.size
        mask_np = np.array(mask)
        
        # Find indices of non-zero regions
        ys, xs = np.nonzero(mask_np)
        if len(xs) == 0 or len(ys) == 0:
            # If mask is all zeros, return the original image
            return image, mask
            
        # Find the minimum bounding box containing the mask
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # Ensure crop area is large enough to contain the mask
        crop_w = max(crop_w, x_max - x_min + 1)
        crop_h = max(crop_h, y_max - y_min + 1)
        crop_w = min(crop_w, img_w)
        crop_h = min(crop_h, img_h)
        
        # Calculate valid crop starting point range
        x_min_crop = max(0, x_max - crop_w)
        x_max_crop = min(x_min, img_w - crop_w)
        y_min_crop = max(0, y_max - crop_h)
        y_max_crop = min(y_min, img_h - crop_h)
        
        if x_min_crop > x_max_crop or y_min_crop > y_max_crop:
            # If range is invalid, use center crop as fallback
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            x0 = max(0, min(center_x - crop_w // 2, img_w - crop_w))
            y0 = max(0, min(center_y - crop_h // 2, img_h - crop_h))
        else:
            # Random crop within valid range
            x0 = random.randint(x_min_crop, x_max_crop)
            y0 = random.randint(y_min_crop, y_max_crop)
        
        # Crop the images
        x1, y1 = x0 + crop_w, y0 + crop_h
        image_cropped = image.crop((x0, y0, x1, y1))
        mask_cropped = mask.crop((x0, y0, x1, y1))
        
        return image_cropped, mask_cropped