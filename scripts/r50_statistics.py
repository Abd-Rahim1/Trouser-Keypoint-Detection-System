import json
from pathlib import Path

# Paths
dataset_stats_path = r"C:dataset\Trousers_Dataset_With_Aug\dataset_with_aug_statistics.json"
hyperparams_path = r"C:models\output_trousers_keypoints\hyperparameters.json"
metrics_path = r"C:models\output_trousers_keypoints\metrics.json"
train_coco_path = r"C:dataset\Trousers_Dataset_With_Aug\coco_format\train_coco.json"
val_coco_path = r"C:dataset\Trousers_Dataset_With_Aug\coco_format\validation_coco.json"
model_final_path = r"models\output_trousers_keypoints\model_final.pth"

output_path = r"C:models\output_trousers_keypoints\training_report_r50.json"

# Load dataset statistics
with open(dataset_stats_path, "r") as f:
    dataset_json = json.load(f)

dataset_info = dataset_json["dataset_info"]
data_splits = dataset_json["data_splits"]
keypoints_info = dataset_json.get("keypoint_metadata", {})

# Prepare dataset stats for report
dataset_stats = {
    "train_images": data_splits["training"]["images"],
    "validation_images": data_splits["validation"]["images"],
    "test_images": data_splits["test"]["images"],
    "num_keypoints": keypoints_info.get("total_keypoints", 14),
    "num_annotations": data_splits["training"]["annotation_jsons"] + data_splits["validation"]["annotation_jsons"],
    "augmentation": True,
    "dataset_name": dataset_info["name"]
}

# Load hyperparameters
with open(hyperparams_path, "r") as f:
    h = json.load(f)

hyperparams = {
    "learning_rate": h["SOLVER"]["BASE_LR"],
    "batch_size": h["SOLVER"]["IMS_PER_BATCH"],
    "max_iterations": h["SOLVER"]["MAX_ITER"],
    "num_classes": h["MODEL"]["ROI_HEADS"]["NUM_CLASSES"],
    "backbone": h["MODEL"]["BACKBONE"]["NAME"],
    "weights_init": h["MODEL"]["WEIGHTS"],
    "dataset_used": h["DATASETS"]["TRAIN"][0]
}

# Load metrics
metrics = {}
with open(metrics_path, "r") as f:
    for line in f:
        if line.strip():
            metrics = json.loads(line)

training_results = {
    "iteration": metrics.get("iteration"),
    "loss_final": metrics.get("total_loss"),
    "loss_box_reg": metrics.get("loss_box_reg"),
    "loss_cls": metrics.get("loss_cls"),
    "loss_keypoint": metrics.get("loss_keypoint"),
    "total_time_hours": metrics.get("time")
}

# Build final report
final_report = {
    "project": "Trouser Keypoint Detection",
    "dataset_statistics": {
        **dataset_stats,
        "paths": {
            "train_coco": train_coco_path,
            "val_coco": val_coco_path
        }
    },
    "hyperparameters": hyperparams,
    "training_results": training_results,
    "model_file": model_final_path
}

# Save final report
output_path_obj = Path(output_path)
output_path_obj.parent.mkdir(parents=True, exist_ok=True)
with open(output_path_obj, "w") as f:
    json.dump(final_report, f, indent=4)

print(f"Final training report saved to {output_path}")
