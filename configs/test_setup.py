# Test imports
try:
    import torch
    import torchvision
    import detectron2
    import cv2
    import numpy as np
    from PIL import Image
    import tkinter as tk
    
    print(" All basic imports successful!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print(f"Detectron2 version: {detectron2.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test Tkinter
    root = tk.Tk()
    root.withdraw()  # Hide window
    print(" Tkinter working!")
    root.destroy()
    
except ImportError as e:
    print(f" Import error: {e}")