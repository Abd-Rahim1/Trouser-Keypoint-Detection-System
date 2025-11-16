import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

class AugmentedStatsGenerator:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.timestamp = datetime.now().strftime("%Y-%m-%d")
        
    def count_files_recursive(self, folder_path, extensions):
        """Count all files with given extensions recursively in a folder"""
        folder = Path(folder_path)
        if folder.exists():
            files = [f for f in folder.rglob("*") if f.suffix.lower() in extensions]
            return len(files), files
        return 0, []
    
    def get_image_stats(self, image_paths, sample_size=50):
        """Get basic image statistics from actual images"""
        if not image_paths:
            return {}
        sample_size = min(sample_size, len(image_paths))
        sizes = []
        total_size_bytes = 0
        for img_path in image_paths[:sample_size]:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    sizes.append((h, w))
                    total_size_bytes += img_path.stat().st_size
            except:
                continue
        if not sizes:
            return {}
        heights = [h for h, w in sizes]
        widths = [w for h, w in sizes]
        return {
            "average_resolution": f"{int(np.mean(heights))}x{int(np.mean(widths))}",
            "min_resolution": f"{min(heights)}x{min(widths)}",
            "max_resolution": f"{max(heights)}x{max(widths)}",
            "sample_size_analyzed": sample_size,
            "color_space": "RGB",
            "average_file_size_kb": round((total_size_bytes / sample_size) / 1024, 1)
        }

    def calculate_folder_size_mb(self, folder_path):
        folder = Path(folder_path)
        total_size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) if folder.exists() else 0
        return round(total_size / (1024 * 1024), 2)

    def generate_statistics(self):
        dataset_path = self.base_path / "Trousers_Dataset_With_Aug"
        if not dataset_path.exists():
            print(f"Dataset folder not found: {dataset_path}")
            return None
        
        # Count images and JSONs for each split
        img_ext = [".jpg", ".jpeg", ".png", ".bmp"]
        json_ext = [".json"]

        train_img_count, train_img_paths = self.count_files_recursive(dataset_path / "train" / "image", img_ext)
        train_json_count, _ = self.count_files_recursive(dataset_path / "train" / "annos", json_ext)

        val_img_count, val_img_paths = self.count_files_recursive(dataset_path / "validation" / "image", img_ext)
        val_json_count, _ = self.count_files_recursive(dataset_path / "validation" / "annos", json_ext)

        test_img_count, test_img_paths = self.count_files_recursive(dataset_path / "test" / "image", img_ext)
        test_json_count, _ = self.count_files_recursive(dataset_path / "test" / "annos", json_ext)

        total_images = train_img_count + val_img_count + test_img_count
        total_disk_mb = self.calculate_folder_size_mb(dataset_path)

        # Image stats from a sample
        all_image_paths = train_img_paths + val_img_paths + test_img_paths
        image_stats = self.get_image_stats(all_image_paths)

        stats = {
            "dataset_info": {
                "name": "Trousers Dataset (With Augmentation)",
                "version": "1.0",
                "description": "Augmented version of filtered trouser dataset with horizontal flipping",
                "creation_date": self.timestamp,
                "total_images": total_images,
                "total_disk_size_mb": total_disk_mb,
                "images_with_14_keypoints": total_images
            },
            "data_splits": {
                "training": {
                    "images": train_img_count,
                    "annotation_jsons": train_json_count,
                    "disk_size_mb": self.calculate_folder_size_mb(dataset_path / "train"),
                    "percentage": f"{(train_img_count/total_images)*100:.1f}%"
                },
                "validation": {
                    "images": val_img_count,
                    "annotation_jsons": val_json_count,
                    "disk_size_mb": self.calculate_folder_size_mb(dataset_path / "validation"),
                    "percentage": f"{(val_img_count/total_images)*100:.1f}%"
                },
                "test": {
                    "images": test_img_count,
                    "annotation_jsons": test_json_count,
                    "disk_size_mb": self.calculate_folder_size_mb(dataset_path / "test"),
                    "percentage": f"{(test_img_count/total_images)*100:.1f}%",
                    "notes": "Test split not augmented"
                }
            },
            "image_characteristics": {**image_stats},
            "keypoint_metadata": {
                "total_keypoints": 14,
                "keypoint_names": [
                    "left_hip", "right_hip", "left_knee", "right_knee",
                    "left_ankle", "right_ankle", "left_pocket", "right_pocket",
                    "waist_center", "crotch", "left_thigh", "right_thigh",
                    "left_hem", "right_hem"
                ],
                "symmetry_pairs": [
                    ["left_hip", "right_hip"],
                    ["left_knee", "right_knee"],
                    ["left_ankle", "right_ankle"],
                    ["left_pocket", "right_pocket"],
                    ["left_thigh", "right_thigh"],
                    ["left_hem", "right_hem"]
                ]
            }
        }
        return stats

    def save_statistics(self, stats, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {output_path}")

    def print_summary(self, stats):
        print("\nDataset Summary:")
        print("="*50)
        for split in ["training", "validation", "test"]:
            s = stats["data_splits"][split]
            print(f"{split.capitalize()} - Images: {s['images']}, Annotation JSONs: {s['annotation_jsons']}, Disk: {s['disk_size_mb']} MB, % of dataset: {s['percentage']}")

def main():
    dataset_base_path = r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\dataset"
    generator = AugmentedStatsGenerator(dataset_base_path)
    stats = generator.generate_statistics()
    if stats:
        output_path = generator.base_path / "Trousers_Dataset_With_Aug" / "dataset_with_aug_statistics.json"
        generator.save_statistics(stats, output_path)
        generator.print_summary(stats)

if __name__ == "__main__":
    main()
