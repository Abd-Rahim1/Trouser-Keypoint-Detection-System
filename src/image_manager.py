from tkinter import filedialog
import os

def select_image():
    """Open file dialog to select an image"""
    file_path = filedialog.askopenfilename(
        title="Select Trouser Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    return file_path

def upload_folder():
    """Upload a folder of images"""
    folder_path = filedialog.askdirectory(title="Select Folder with Images")
    if folder_path:
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        image_list = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_list.append(os.path.join(folder_path, file))
        
        # Sort images for consistent navigation
        image_list.sort()
        return image_list
    return None