from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import torch
from tkinter import messagebox

def load_trouser_model(status_label=None):
    """Load the custom Detectron2 trousers keypoint model"""
    try:
        cfg = get_cfg()
        
        # Base model architecture (ResNet-101 FPN with keypoints)
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
        ))
         
        # Use GPU if available
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load your trained trousers model weights
        cfg.MODEL.WEIGHTS = r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\models\output_trousers_keypoints_101_V2\model_final.pth"
        
        # Custom dataset settings
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1              # trousers only
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14   # 14 trousers keypoints
        cfg.MODEL.KEYPOINT_ON = True
        
        # Inference threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        # Remove dataset dependency for inference
        cfg.DATASETS.TEST = ()
        cfg.INPUT.MASK_FORMAT = "polygon"
        
        # Initialize predictor
        predictor = DefaultPredictor(cfg)
        
        if status_label:
            status_label.config(text="Custom trousers model loaded successfully")
        
        return predictor
        
    except Exception as e:
        messagebox.showerror("Model Loading Error", f"Failed to load model: {str(e)}")
        if status_label:
            status_label.config(text="Model loading failed")
        return None