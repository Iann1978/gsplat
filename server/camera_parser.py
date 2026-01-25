"""Parser for JSON camera data and dataset class."""

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class CameraParser:
    """Parser for JSON camera data."""
    
    def __init__(self, data_dir: str, camera_dir: str, normalize: bool = False):
        """Initialize camera parser.
        
        Args:
            data_dir: Directory containing images
            camera_dir: Directory containing .cam.json files with camera data
            normalize: Whether to normalize the world space
        """
        self.data_dir = data_dir
        
        # Check if camera directory exists
        if not os.path.isdir(camera_dir):
            raise ValueError(f"Camera directory {camera_dir} does not exist.")
        
        # Load all .cam.json files from the directory
        camera_files = glob.glob(os.path.join(camera_dir, "*.cam.json"))
        if not camera_files:
            raise ValueError(f"No .cam.json files found in {camera_dir}")
        
        # Load and merge all camera data
        camera_data = {}
        for camera_file in sorted(camera_files):
            with open(camera_file, "r") as f:
                file_data = json.load(f)
                # Each file contains one camera entry: {"IMAGE_NAME.JPG": {...}}
                camera_data.update(file_data)
        
        # Extract image names, camera matrices, and intrinsics
        self.image_names = []
        self.image_paths = []
        camtoworlds_list = []
        Ks_list = []
        imsizes_list = []
        
        image_dir = os.path.join(data_dir, "images")
        if not os.path.exists(image_dir):
            raise ValueError(f"Image folder {image_dir} does not exist.")
        
        # Sort image names for consistency
        sorted_image_names = sorted(camera_data.keys())
        
        for image_name in sorted_image_names:
            cam_info = camera_data[image_name]
            
            # Get camera-to-world matrix (4x4)
            if "camtoworld" in cam_info:
                c2w = np.array(cam_info["camtoworld"], dtype=np.float32)
            elif "c2w" in cam_info:
                c2w = np.array(cam_info["c2w"], dtype=np.float32)
            else:
                raise ValueError(f"Missing camtoworld/c2w in camera data for {image_name}")
            
            # Get intrinsic matrix (3x3)
            if "K" in cam_info:
                K = np.array(cam_info["K"], dtype=np.float32)
            elif "intrinsic" in cam_info:
                K = np.array(cam_info["intrinsic"], dtype=np.float32)
            else:
                raise ValueError(f"Missing K/intrinsic in camera data for {image_name}")
            
            # Get image size
            if "width" in cam_info and "height" in cam_info:
                width = int(cam_info["width"])
                height = int(cam_info["height"])
            else:
                # Try to load image to get size
                image_path = os.path.join(image_dir, image_name)
                if os.path.exists(image_path):
                    img = imageio.imread(image_path)
                    height, width = img.shape[:2]
                else:
                    raise ValueError(f"Cannot determine image size for {image_name}")
            
            # Store data
            self.image_names.append(image_name)
            image_path = os.path.join(image_dir, image_name)
            self.image_paths.append(image_path)
            camtoworlds_list.append(c2w)
            Ks_list.append(K)
            imsizes_list.append((width, height))
        
        # Convert to numpy arrays
        self.camtoworlds = np.stack(camtoworlds_list, axis=0)  # (N, 4, 4)
        
        # Store intrinsics and sizes (using camera_id as index)
        # For simplicity, assume each image can have different camera intrinsics
        self.Ks_dict = {i: Ks_list[i] for i in range(len(Ks_list))}
        self.imsize_dict = {i: imsizes_list[i] for i in range(len(imsizes_list))}
        self.camera_ids = list(range(len(self.image_names)))
        
        # Normalize world space if requested
        if normalize:
            # Calculate scene scale from camera positions
            camera_locations = self.camtoworlds[:, :3, 3]
            scene_center = np.mean(camera_locations, axis=0)
            dists = np.linalg.norm(camera_locations - scene_center, axis=1)
            max_dist = np.max(dists)
            
            # Normalize to unit sphere
            scale = 1.0 / (max_dist + 1e-8)
            self.camtoworlds[:, :3, 3] = (self.camtoworlds[:, :3, 3] - scene_center) * scale
            
            self.transform = np.eye(4)
            self.transform[:3, 3] = -scene_center
            self.transform[:3, :3] *= scale
        else:
            self.transform = np.eye(4)
        
        # Calculate scene scale
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists) if len(dists) > 0 else 1.0
        
        print(f"[CameraParser] Loaded {len(self.image_names)} images.")
        print(f"[CameraParser] Scene scale: {self.scene_scale:.4f}")


class CameraDataset(Dataset):
    """Dataset class for camera data."""
    
    def __init__(
        self,
        parser: CameraParser,
        split: str = "train",
        test_every: int = 8,
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        """Initialize dataset.
        
        Args:
            parser: CameraParser instance
            split: "train" or "val"
            test_every: Every N images is a test image
            patch_size: Optional random crop size for training
            load_depths: Whether to load depth data (not implemented)
        """
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        
        # Split indices
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % test_every != 0]
        else:
            self.indices = indices[indices % test_every == 0]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get a data sample.
        
        Returns:
            Dictionary with:
                - K: torch.Tensor (3, 3) - Intrinsic matrix
                - camtoworld: torch.Tensor (4, 4) - Camera-to-world matrix
                - image: torch.Tensor (H, W, 3) - RGB image (0-255, float)
                - image_id: int - Image index
        """
        index = self.indices[item]
        
        # Load image
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        
        # Get camera data
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        camtoworlds = self.parser.camtoworlds[index]
        
        # Random crop for training
        if self.patch_size is not None and self.split == "train":
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y
        
        # Convert to torch tensors
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }
        
        return data
