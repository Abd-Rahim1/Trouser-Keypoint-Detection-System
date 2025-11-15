import json
import os
import cv2
from tkinter import messagebox

def check_annotation_exists(image_path):
    """Check if annotation file exists for the given image"""
    base_name = os.path.basename(image_path)
    base_no_ext = os.path.splitext(base_name)[0]
    annotation_path = os.path.join(
        r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\dataset\train\annos",
        base_no_ext + ".json"
    )
    return annotation_path if os.path.exists(annotation_path) else None

def load_annotation_file(annotation_path, keypoints):
    """Load keypoints from annotation file"""
    try:
        with open(annotation_path, 'r') as f:
            ann = json.load(f)

        if "landmarks" in ann and len(ann["landmarks"]) == 14 * 3:
            keypoints.clear()
            coords = ann["landmarks"]
            for i in range(0, len(coords), 3):
                x, y, v = coords[i], coords[i + 1], coords[i + 2]
                keypoints.append((x, y, int(v)))
            
            return True
        return False
    except Exception as e:
        print(f"Error loading annotation: {e}")
        return False

def save_to_deepfashion_format(image_path, original_image, keypoints):
    """Save image and keypoints in DeepFashion2 format"""
    base_name = os.path.basename(image_path)
    img_save_path = os.path.join(r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\dataset\train\image", base_name)
    base_name_no_ext = os.path.splitext(base_name)[0]
    json_save_path = os.path.join(
        r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\dataset\train\annos", base_name_no_ext + ".json"
    )

    # Save image
    bgr_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_save_path, bgr_image)

    # Prepare landmarks
    landmark_vector = []
    for x, y, v in keypoints:
        landmark_vector.extend([float(x), float(y), float(v)])

    # Save JSON
    ann = {
        "image": base_name,
        "landmarks": landmark_vector,
        "size": [original_image.shape[1], original_image.shape[0]]
    }

    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    with open(json_save_path, 'w') as f:
        json.dump(ann, f, indent=2)

    visible_count = sum(1 for _, _, v in keypoints if v > 0)
    messagebox.showinfo("Saved", f"Saved image + annotation ({visible_count} keypoints)")
    return visible_count

def show_save_location(image_path=None):
    """Show information about DeepFashion2 save locations"""
    image_save_dir = r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\dataset\train\image"
    annos_save_dir = r"C:\Users\Document\OneDrive\Desktop\trouser_keypoint_project\dataset\train\annos"
    
    info = "DeepFashion2 Save Locations:\n\n"
    info += f"Images saved to:\n{image_save_dir}\n\n"
    info += f"Annotations saved to:\n{annos_save_dir}\n\n"
    
    if image_path:
        base_name = os.path.basename(image_path)
        base_no_ext = os.path.splitext(base_name)[0]
        info += f"Current image will be saved as:\n"
        info += f"• Image: {os.path.join(image_save_dir, base_name)}\n"
        info += f"• Annotation: {os.path.join(annos_save_dir, base_no_ext + '.json')}\n"
    else:
        info += "Load an image to see specific save paths."
    
    messagebox.showinfo("DeepFashion2 Save Locations", info)