import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

class TrousersDatasetStats:
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.timestamp = datetime.now().strftime("%Y-%m-%d")
        self.valid_img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

    # Count images recursively
    def count_images(self, folder):
        folder = Path(folder)
        if not folder.exists():
            return 0, []
        image_paths = [p for p in folder.rglob('*') if p.suffix in self.valid_img_ext]
        return len(image_paths), image_paths

    # Compute image resolution and size stats
    def compute_image_stats(self, paths, sample_size=100):
        if not paths:
            return {}

        sample_size = min(sample_size, len(paths))
        sample_paths = paths[:sample_size]

        heights, widths, file_sizes = [], [], []

        for img_path in sample_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                heights.append(h)
                widths.append(w)
                file_sizes.append(img_path.stat().st_size)

        if not heights:
            return {}

        return {
            "sampled_images": sample_size,
            "average_resolution": f"{int(np.mean(heights))}x{int(np.mean(widths))}",
            "min_resolution": f"{min(heights)}x{min(widths)}",
            "max_resolution": f"{max(heights)}x{max(widths)}",
            "avg_file_size_kb": round(np.mean(file_sizes) / 1024, 2)
        }

    # Compute folder size in MB
    def compute_folder_size_mb(self, folder):
        folder = Path(folder)
        if not folder.exists():
            return 0
        total = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
        return round(total / (1024 * 1024), 2)

    # Generate full stats dictionary
    def generate(self):
        base = self.dataset_root / "Trousers_Dataset_Without_Aug"
        if not base.exists():
            print(f" Dataset folder not found: {base}")
            return None

        print(" Generating statistics for Trousers_Dataset_Without_Aug...")

        # Count images per split
        train_count, train_imgs = self.count_images(base / "train")
        val_count, val_imgs = self.count_images(base / "validation")
        test_count, test_imgs = self.count_images(base / "test")

        total_images = train_count + val_count + test_count

        # Compute image stats from a sample of all images
        all_imgs = train_imgs + val_imgs + test_imgs
        img_stats = self.compute_image_stats(all_imgs)

        # Compute folder sizes
        train_size = self.compute_folder_size_mb(base / "train")
        val_size = self.compute_folder_size_mb(base / "validation")
        test_size = self.compute_folder_size_mb(base / "test")

        stats = {
            "dataset_name": "Trousers_Dataset",
            "created_on": self.timestamp,
            "total_images": total_images,
            "splits": {
                "train": {
                    "images": train_count,
                    "annotation_files": 0,
                    "size_mb": train_size,
                    "percentage": f"{(train_count/total_images)*100:.1f}%"
                },
                "validation": {
                    "images": val_count,
                    "annotation_files": 0,
                    "size_mb": val_size,
                    "percentage": f"{(val_count/total_images)*100:.1f}%"
                },
                "test": {
                    "images": test_count,
                    "annotation_files": 0,
                    "size_mb": test_size,
                    "percentage": f"{(test_count/total_images)*100:.1f}%"
                }
            },
            "image_statistics": img_stats,
            "disk_usage_mb": train_size + val_size + test_size
        }

        return stats

    # Save stats as JSON
    def save(self, stats, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f" Statistics saved to: {out_path}")


def main():
    dataset_root = r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\dataset"
    generator = TrousersDatasetStats(dataset_root)
    stats = generator.generate()

    if stats:
        output_file = Path(dataset_root) / "Trousers_Dataset_Without_Aug" / "dataset_with_aug_statistics.json"
        generator.save(stats, output_file)

if __name__ == "__main__":
    main()
