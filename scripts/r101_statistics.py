import json
from pathlib import Path

# Paths
dataset_stats_path = r"C:dataset\Trousers_Dataset_Without_Aug\dataset_without_aug_statistics.json"
metrics_path = r"C:models\output_trousers_keypoints_101\metrics.json"
model_final_path = r"C:models\output_trousers_keypoints_101\model_final.pth"
train_coco_path = r"C:dataset\Trousers_Dataset_Without_Aug\coco_format\train_coco.json"
val_coco_path = r"C:dataset\Trousers_Dataset_Without_Aug\coco_format\validation_coco.json"
output_path = r"C:models\output_trousers_keypoints_101\training_report_r101.json"

# Import cfg 
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))

# Use your actual training/testing datasets
cfg.DATASETS.TRAIN = ("trousers_train",)
cfg.DATASETS.TEST = ("trousers_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = os.path.join(str(model_final_path))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []
cfg.TEST.EVAL_PERIOD = 0
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 100

# Load dataset statistics
with open(dataset_stats_path, "r") as f:
    ds_json = json.load(f)

train_images = ds_json["splits"]["train"]["images"]
val_images = ds_json["splits"]["validation"]["images"]
test_images = ds_json["splits"]["test"]["images"]
total_annotations = ds_json["total_images"]

dataset_stats = {
    "train_images": train_images,
    "validation_images": val_images,
    "test_images": test_images,
    "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
    "num_annotations": total_annotations,
    "augmentation": False,
    "dataset_name": ds_json.get("dataset_name", "Trousers Dataset (Without Aug)"),
    "paths": {
        "train_coco": train_coco_path,
        "val_coco": val_coco_path
    }
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

# Extract hyperparameters from cfg
hyperparams = {
    "learning_rate": cfg.SOLVER.BASE_LR,
    "batch_size": cfg.SOLVER.IMS_PER_BATCH,
    "max_iterations": cfg.SOLVER.MAX_ITER,
    "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
    "backbone": cfg.MODEL.BACKBONE.NAME if hasattr(cfg.MODEL, "BACKBONE") else "R_101_FPN",
    "weights_init": cfg.MODEL.WEIGHTS,
    "dataset_used": cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else None
}

# Final report
final_report = {
    "project": "Trouser Keypoint Detection",
    "dataset_statistics": dataset_stats,
    "hyperparameters": hyperparams,
    "training_results": training_results,
    "model_file": str(model_final_path)
}

# Save JSON
output_file = Path(output_path)
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(final_report, f, indent=4)

print(f"Training report saved to: {output_file}")
