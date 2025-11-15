import sys
import subprocess
import os

def check_detectron():
    print("=== Detectron2 Installation Check ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if detectron2 is installed
    try:
        import detectron2
        print("✓ Detectron2 is installed")
        print(f"Detectron2 path: {detectron2.__file__}")
        return True
    except ImportError as e:
        print("✗ Detectron2 is NOT installed")
        print(f"Error: {e}")
        return False

def check_packages():
    print("\n=== Checking Required Packages ===")
    packages = ['torch', 'torchvision', 'opencv-python', 'Pillow']
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")

if __name__ == "__main__":
    check_detectron()
    check_packages()
    
    if not check_detectron():
        print("\n=== Installation Instructions ===")
        print("Run these commands:")
        print("1. conda activate detectron-env")
        print("2. pip install 'git+https://github.com/facebookresearch/detectron2.git'")