import json
from pathlib import Path
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import numpy as np

# Paths
dataset_stats_path = r"C:dataset\Trousers_Dataset_Without_Aug\dataset_without_aug_statistics.json"
metrics_path = r"C:models\output_trousers_keypoints_101_V2\metrics.json"
model_final_path = r"C:models\output_trousers_keypoints_101_V2\model_final.pth"
train_coco_path = r"C:dataset\Trousers_Dataset_Without_Aug\coco_format\train_coco.json"
val_coco_path = r"C:dataset\Trousers_Dataset_Without_Aug\coco_format\validation_coco.json"
output_path = r"C:models\output_trousers_keypoints_101_V2\final_training_report_.json"

# Detectron2 Config
classes = ["Trousers"]
n_keypoints = 14
Batch_size = 4
LearningRate = 0.001
max_iter = 20000

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("trousers_train",)
cfg.DATASETS.TEST = ("trousers_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = str(model_final_path)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = n_keypoints
cfg.SOLVER.IMS_PER_BATCH = Batch_size
cfg.SOLVER.BASE_LR = LearningRate
cfg.SOLVER.MAX_ITER = max_iter
cfg.SOLVER.STEPS = []
cfg.TEST.EVAL_PERIOD = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((n_keypoints, 1), dtype=float).tolist()
cfg.OUTPUT_DIR = r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\models\output_trousers_keypoints_101_V2"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Load dataset statistics
with open(dataset_stats_path, "r") as f:
    ds_json = json.load(f)

dataset_stats = {
    "train_images": ds_json["splits"]["train"]["images"],
    "validation_images": ds_json["splits"]["validation"]["images"],
    "test_images": ds_json["splits"]["test"]["images"],
    "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
    "num_annotations": ds_json["total_images"],
    "augmentation": False,
    "dataset_name": ds_json.get("dataset_name", "Trousers Dataset (Without Aug)"),
    "paths": {
        "train_coco": train_coco_path,
        "val_coco": val_coco_path
    }
}

# Load training metrics (line-delimited JSON)
metrics = {}
with open(metrics_path, "r") as f:
    for line in f:
        if line.strip():
            metrics = json.loads(line)  # last line will contain final metrics

training_results = {
    "iteration": metrics.get("iteration"),
    "loss_final": metrics.get("total_loss"),
    "loss_box_reg": metrics.get("loss_box_reg"),
    "loss_cls": metrics.get("loss_cls"),
    "loss_keypoint": metrics.get("loss_keypoint"),
    "total_time_hours": metrics.get("time")
}

# Hyperparameters
hyperparams = {
    "learning_rate": cfg.SOLVER.BASE_LR,
    "batch_size": cfg.SOLVER.IMS_PER_BATCH,
    "max_iterations": cfg.SOLVER.MAX_ITER,
    "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
    "backbone": cfg.MODEL.BACKBONE.NAME if hasattr(cfg.MODEL, "BACKBONE") else "R_101_FPN",
    "weights_init": cfg.MODEL.WEIGHTS,
    "dataset_used": cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else None
}

# Final JSON Report
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
