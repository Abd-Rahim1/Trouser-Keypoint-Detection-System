import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

class TrousersDatasetStats:
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.timestamp = datetime.now().strftime("%Y-%m-%d")
        self.valid_img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

    def count_images(self, folder):
        folder = Path(folder)
        if not folder.exists():
            return 0, []
        image_paths = [p for p in folder.rglob('*') if p.suffix in self.valid_img_ext]
        return len(image_paths), image_paths

    def compute_image_stats(self, paths, sample_size=100):
        if not paths:
            return {}

        sample_size = min(sample_size, len(paths))
        sample_paths = paths[:sample_size]

        heights, widths, file_sizes = [], [], []
        formats = []

        corrupted_count = 0
        small_images_count = 0

        for img_path in sample_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupted_count += 1
                    continue
                h, w = img.shape[:2]
                heights.append(h)
                widths.append(w)
                file_sizes.append(img_path.stat().st_size)
                formats.append(img_path.suffix.lower())
                if h < 64 or w < 64:
                    small_images_count += 1
            except Exception:
                corrupted_count += 1
                continue

        if not heights:
            return {}

        return {
            "sampled_images": sample_size,
            "corrupted_images": corrupted_count,
            "small_images_below_64x64": small_images_count,
            "average_resolution": f"{int(np.mean(heights))}x{int(np.mean(widths))}",
            "min_resolution": f"{min(heights)}x{min(widths)}",
            "max_resolution": f"{max(heights)}x{max(widths)}",
            "average_height": round(np.mean(heights), 1),
            "average_width": round(np.mean(widths), 1),
            "average_aspect_ratio": round(np.mean(np.array(widths)/np.array(heights)), 2),
            "avg_file_size_kb": round(np.mean(file_sizes) / 1024, 2),
            "image_formats": dict(Counter(formats)),
            "top_5_largest_images_kb": sorted([(p.name, round(p.stat().st_size/1024,2)) for p in sample_paths], key=lambda x: x[1], reverse=True)[:5]
        }

    def compute_folder_size_mb(self, folder):
        folder = Path(folder)
        if not folder.exists():
            return 0
        total = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
        return round(total / (1024 * 1024), 2)

    def generate(self):
        base = self.dataset_root / "Trousers_Dataset_Without_Aug"
        if not base.exists():
            print(f" Dataset folder not found: {base}")
            return None

        print(" Generating extended statistics for Trousers_Dataset_Without_Aug...")

        # Split counts
        train_count, train_imgs = self.count_images(base / "train")
        val_count, val_imgs = self.count_images(base / "validation")
        test_count, test_imgs = self.count_images(base / "test")

        total_images = train_count + val_count + test_count
        all_imgs = train_imgs + val_imgs + test_imgs
        img_stats = self.compute_image_stats(all_imgs)

        # Folder sizes
        train_size = self.compute_folder_size_mb(base / "train")
        val_size = self.compute_folder_size_mb(base / "validation")
        test_size = self.compute_folder_size_mb(base / "test")

        stats = {
            "dataset_name": "Trousers_Dataset_Without_Aug",
            "created_on": self.timestamp,
            "total_images": total_images,
            "splits": {
                "train": {"images": train_count, "size_mb": train_size, "percentage": f"{(train_count/total_images)*100:.1f}%"},
                "validation": {"images": val_count, "size_mb": val_size, "percentage": f"{(val_count/total_images)*100:.1f}%"},
                "test": {"images": test_count, "size_mb": test_size, "percentage": f"{(test_count/total_images)*100:.1f}%"},
            },
            "image_statistics": img_stats,
            "disk_usage_mb": train_size + val_size + test_size
        }

        return stats

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
        output_file = Path(dataset_root) / "Trousers_Dataset_Without_Aug" / "dataset_Without_Aug_statistics.json"
        generator.save(stats, output_file)

if __name__ == "__main__":
    main()
