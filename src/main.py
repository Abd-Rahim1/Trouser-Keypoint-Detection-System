#!/usr/bin/env python3
"""
Simple main file that imports and runs the app
"""
import sys
import os

# Make sure we can import from current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import and run
try:
    from gui_app import main
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory:")
    files = ["gui_app.py", "model_loader.py", "image_manager.py", "keypoint_operations.py", "utils/annotation_utils.py"]
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
    input("Press Enter to exit...")