"""
Module: dataset_analyzer.py

This module provides tools for analyzing our dataset.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class ImageStats:
    """Statistics for a single image and its corresponding mask."""
    width: int
    height: int
    channels: int
    mask_unique_values: Set[int]
    file_size_kb: float


@dataclass
class DatasetStats:
    """Overall statistics for a dataset of images and masks."""
    total_images: int
    image_sizes: Counter
    image_channels: Counter
    mask_classes: Set[int]
    class_distribution: Dict[int, int]
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    mean_width: float
    mean_height: float
    file_sizes_kb: Dict[str, List[float]]
    aspect_ratios: Counter


class DatasetAnalyzer:
    """
    Analyzer for image datasets used in semantic segmentation tasks.
    
    This class provides methods to analyze datasets containing images and their 
    corresponding segmentation masks, calculating various statistics to understand
    the dataset better.
    """
    
    def __init__(
        self, 
        dataset_path: Union[str, Path],
        img_dir_name: str = "color",
        mask_dir_name: str = "label",
        img_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        mask_extensions: Tuple[str, ...] = (".png",),
    ):
        """
        Initialize the DatasetAnalyzer.
        
        Args:
            dataset_path: Path to the dataset directory
            img_dir_name: Name of the subdirectory containing images
            mask_dir_name: Name of the subdirectory containing masks
            img_extensions: Tuple of valid image file extensions
            mask_extensions: Tuple of valid mask file extensions
        """
        self.dataset_path = Path(dataset_path)
        self.img_dir_name = img_dir_name
        self.mask_dir_name = mask_dir_name
        self.img_extensions = img_extensions
        self.mask_extensions = mask_extensions
        
        # Validate dataset path
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
    
    def _find_image_mask_pairs(self, subset_dir: Path) -> List[Tuple[Path, Path]]:
        """
        Find all image and mask file pairs in the given directory.
        
        Args:
            subset_dir: Path to subset directory (e.g., Train, Val, Test)
            
        Returns:
            List of tuples containing (image_path, mask_path)
        """
        img_dir = subset_dir / self.img_dir_name
        mask_dir = subset_dir / self.mask_dir_name
        
        if not img_dir.exists() or not mask_dir.exists():
            return []
        
        # Get all image files
        img_files = []
        for ext in self.img_extensions:
            img_files.extend(list(img_dir.glob(f"*{ext}")))
        
        # Find corresponding mask files
        pairs = []
        for img_path in img_files:
            img_stem = img_path.stem
            
            # Try to find a matching mask with the same stem but potentially different extension
            mask_path = None
            for ext in self.mask_extensions:
                potential_mask = mask_dir / f"{img_stem}{ext}"
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if mask_path:
                pairs.append((img_path, mask_path))
            
        return pairs
    
    def analyze_image(self, img_path: Path, mask_path: Path) -> ImageStats:
        """
        Analyze a single image and its corresponding mask.
        
        Args:
            img_path: Path to the image file
            mask_path: Path to the mask file
            
        Returns:
            ImageStats object containing statistics about the image
        """
        # Read image and get basic properties
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        height, width = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        
        # Read mask and get unique values
        mask = np.array(Image.open(mask_path))
        mask_unique_values = set(np.unique(mask).tolist())
        
        # Get file size in KB
        file_size_kb = img_path.stat().st_size / 1024
        
        return ImageStats(
            width=width,
            height=height,
            channels=channels,
            mask_unique_values=mask_unique_values,
            file_size_kb=file_size_kb,
        )
    
    def analyze_dataset(
        self, 
        subset_dirs: Optional[List[str]] = None,
        sample_limit: Optional[int] = None,
        verbose: bool = True
    ) -> DatasetStats:
        """
        Analyze the dataset and gather statistics.
        
        Args:
            subset_dirs: List of subdirectory names to analyze (e.g., ["Train", "Val", "Test"]).
                         If None, analyze all subdirectories under dataset_path.
            sample_limit: Maximum number of images to analyze (for testing)
            verbose: Whether to show progress bar
            
        Returns:
            DatasetStats object containing statistics about the dataset
        """
        if subset_dirs is None:
            # Use all directories in the dataset path
            subset_dirs = [p.name for p in self.dataset_path.iterdir() if p.is_dir()]
        
        # Find all image-mask pairs in the specified subset directories
        all_pairs = []
        for subset in subset_dirs:
            subset_path = self.dataset_path / subset
            if subset_path.exists():
                pairs = self._find_image_mask_pairs(subset_path)
                all_pairs.extend([(subset, img, mask) for img, mask in pairs])
                
        # Limit samples if requested
        if sample_limit and len(all_pairs) > sample_limit:
            all_pairs = all_pairs[:sample_limit]
        
        # Initialize statistics
        total_images = len(all_pairs)
        image_sizes = Counter()  # (width, height)
        image_channels = Counter()  # number of channels
        mask_classes = set()
        class_distribution = defaultdict(int)
        widths, heights = [], []
        file_sizes = defaultdict(list)
        aspect_ratios = Counter()
        
        # Process each image-mask pair
        iterator = tqdm(all_pairs, desc="Analyzing images") if verbose else all_pairs
        for subset, img_path, mask_path in iterator:
            try:
                stats = self.analyze_image(img_path, mask_path)
                
                # Update statistics
                image_sizes[(stats.width, stats.height)] += 1
                image_channels[stats.channels] += 1
                mask_classes.update(stats.mask_unique_values)
                
                # Update class distribution
                for class_id in stats.mask_unique_values:
                    class_distribution[class_id] += 1
                
                widths.append(stats.width)
                heights.append(stats.height)
                
                # Calculate aspect ratio (to two decimal places)
                aspect_ratio = round(stats.width / stats.height, 2)
                aspect_ratios[aspect_ratio] += 1
                
                # Group file sizes by subset
                file_sizes[subset].append(stats.file_size_kb)
                
            except Exception as e:
                if verbose:
                    print(f"Error analyzing {img_path}: {e}")
                continue
        
        # Calculate aggregate statistics
        min_width = min(widths) if widths else 0
        max_width = max(widths) if widths else 0
        min_height = min(heights) if heights else 0
        max_height = max(heights) if heights else 0
        mean_width = sum(widths) / len(widths) if widths else 0
        mean_height = sum(heights) / len(heights) if heights else 0
        
        return DatasetStats(
            total_images=total_images,
            image_sizes=image_sizes,
            image_channels=image_channels,
            mask_classes=mask_classes,
            class_distribution=dict(class_distribution),
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            mean_width=mean_width,
            mean_height=mean_height,
            file_sizes_kb=dict(file_sizes),
            aspect_ratios=aspect_ratios,
        )
    
    def print_dataset_summary(self, stats: DatasetStats) -> None:
        """
        Print a summary of the dataset statistics.
        
        Args:
            stats: DatasetStats object to summarize
        """
        print(f"Dataset Summary:")
        print(f"===============")
        print(f"Total Images: {stats.total_images}")
        print(f"Image Size Range: Width [{stats.min_width}-{stats.max_width}], Height [{stats.min_height}-{stats.max_height}]")
        print(f"Average Image Size: {stats.mean_width:.1f} x {stats.mean_height:.1f}")
        
        print("\nMost Common Image Sizes:")
        for (w, h), count in stats.image_sizes.most_common(5):
            print(f"  {w} x {h}: {count} images ({count/stats.total_images*100:.1f}%)")
            
        print("\nMost Common Aspect Ratios:")
        for ratio, count in stats.aspect_ratios.most_common(5):
            print(f"  {ratio}: {count} images ({count/stats.total_images*100:.1f}%)")
        
        print("\nImage Channels:")
        for channels, count in stats.image_channels.items():
            print(f"  {channels} channels: {count} images ({count/stats.total_images*100:.1f}%)")
            
        print("\nMask Classes:")
        print(f"  Class IDs found: {sorted(stats.mask_classes)}")
        
        print("\nClass Distribution:")
        for class_id, count in sorted(stats.class_distribution.items()):
            print(f"  Class {class_id}: {count} images ({count/stats.total_images*100:.1f}%)")


def main() -> None:
    """Example usage of the DatasetAnalyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze an image dataset for semantic segmentation")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--img-dir", type=str, default="color", help="Name of the image directory")
    parser.add_argument("--mask-dir", type=str, default="label", help="Name of the mask directory")
    parser.add_argument("--subsets", type=str, nargs="+", help="Subset directories to analyze")
    parser.add_argument("--sample", type=int, help="Limit the number of samples to analyze")
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer(
        dataset_path=args.dataset_path,
        img_dir_name=args.img_dir,
        mask_dir_name=args.mask_dir,
    )
    
    stats = analyzer.analyze_dataset(
        subset_dirs=args.subsets,
        sample_limit=args.sample,
        verbose=True
    )
    
    analyzer.print_dataset_summary(stats)
    
    
if __name__ == "__main__":
    main()