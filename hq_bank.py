"""
HQ Image Bank for Stage-2 Prompt Learning

Provides a simple interface to sample high-quality images for training
prompts to distinguish between HQ and LQ image representations.
"""

import os
import glob
import torch
from PIL import Image
import torchvision.transforms as T


class HQBank:
    """
    Simple HQ image bank for prompt learning.
    
    Expected structure:
        root/
          class1/
            img1.jpg
            img2.jpg
            ...
          class2/
            img1.jpg
            ...
    """
    
    def __init__(self, root, image_size=224):
        """
        Initialize HQ image bank.
        
        Args:
            root: Root directory containing class subdirectories with images
            image_size: Target image size (default: 224)
        """
        self.root = root
        self.image_size = image_size
        
        # Find all images (supports .jpg, .jpeg, .png)
        self.paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.paths.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
        
        self.paths = sorted(self.paths)
        
        # Image transforms
        self.tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        
        if len(self.paths) == 0:
            print(f"Warning: No images found in {root}. Stage-2 prompt learning will be disabled.")
        else:
            print(f"HQ Bank initialized: {len(self.paths)} images found in {root}")
    
    def sample_batch(self, batch_size=16, device="cuda"):
        """
        Sample a batch of HQ images.
        
        Args:
            batch_size: Number of images to sample
            device: Device to place images on
            
        Returns:
            torch.Tensor: [B, 3, H, W] tensor of images
        """
        if len(self.paths) == 0:
            # Return dummy batch if no images available
            return torch.zeros(batch_size, 3, self.image_size, self.image_size, device=device)
        
        import random
        # Sample with replacement if batch_size > available images
        picks = random.choices(self.paths, k=batch_size)
        
        imgs = []
        for p in picks:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(self.tf(img))
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
                # Use dummy image if loading fails
                imgs.append(torch.zeros(3, self.image_size, self.image_size))
        
        return torch.stack(imgs, 0).to(device)  # [B, 3, 224, 224]
    
    def __len__(self):
        """Return number of available images"""
        return len(self.paths)

