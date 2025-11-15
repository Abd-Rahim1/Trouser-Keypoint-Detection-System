import cv2
import numpy as np
from tkinter import messagebox

def predict_on_resized(original_image, predictor, keypoint_names, visibility_threshold, process_size=(256, 256)):
    """Predict keypoints on resized image"""
    try:
        # Resize original image to model input size
        h, w = original_image.shape[:2]
        resized_image = cv2.resize(original_image, process_size)
        resize_scale_x = w / process_size[0]
        resize_scale_y = h / process_size[1]

        # Predict on resized image
        bgr_resized = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        outputs = predictor(bgr_resized)

        if "instances" in outputs and len(outputs["instances"]) > 0:
            instances = outputs["instances"]
            scores = instances.scores.cpu().numpy()
            best_idx = np.argmax(scores)
            
            if hasattr(instances, 'pred_keypoints'):
                kp_tensor = instances.pred_keypoints[best_idx].cpu().numpy()
                keypoints = []
                raw_keypoints = kp_tensor.copy()
                
                for i in range(len(keypoint_names)):
                    if i < len(kp_tensor):
                        x, y, v = kp_tensor[i]
                        # Scale back to original resolution
                        x *= resize_scale_x
                        y *= resize_scale_y
                        visible = 1 if v > visibility_threshold else 0
                        keypoints.append((float(x), float(y), visible))
                    else:
                        keypoints.append((0, 0, 0))
                
                visible_count = sum(1 for _, _, v in keypoints if v > 0)
                return keypoints, raw_keypoints, visible_count
            else:
                messagebox.showwarning("Warning", "No keypoints detected in model output.")
                return None
        else:
            messagebox.showinfo("Info", "No instances detected in the image.")
            return None
            
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Failed to predict keypoints: {str(e)}")
        return None

def draw_keypoints(canvas, keypoints, keypoint_names, scale_factor, selected_keypoint=None):
    """Draw keypoints on the canvas"""
    # Clear existing keypoints
    for item in canvas.find_withtag("keypoint"):
        canvas.delete(item)
    
    for i, (x, y, visible) in enumerate(keypoints):
        if visible and x > 0 and y > 0:  # Check for valid coordinates
            # Scale coordinates for display
            display_x = x * scale_factor
            display_y = y * scale_factor
            
            # Choose color based on selection
            color = "red" if i == selected_keypoint else "yellow"
            outline_color = "black"
            
            # Draw larger circle for better visibility
            radius = 8
            canvas.create_oval(
                display_x - radius, display_y - radius,
                display_x + radius, display_y + radius,
                fill=color, outline=outline_color, width=3,
                tags="keypoint"
            )
            
            # Draw keypoint name with background
            canvas.create_rectangle(
                display_x - 30, display_y - 25,
                display_x + 30, display_y - 10,
                fill="black", outline="white",
                tags="keypoint"
            )
            
            canvas.create_text(
                display_x, display_y - 17,
                text=keypoint_names[i][:8],  # Truncate long names
                fill="white", font=("Arial", 8, "bold"),
                tags="keypoint"
            )
            
            # Also draw a small cross for precise positioning
            cross_size = 3
            canvas.create_line(
                display_x - cross_size, display_y,
                display_x + cross_size, display_y,
                fill="black", width=2,
                tags="keypoint"
            )
            canvas.create_line(
                display_x, display_y - cross_size,
                display_x, display_y + cross_size,
                fill="black", width=2,
                tags="keypoint"
            )

def apply_threshold(raw_keypoints, threshold, keypoint_names):
    """Apply new visibility threshold to existing keypoints"""
    keypoints = []
    for i in range(len(keypoint_names)):
        if i < len(raw_keypoints):
            x, y, v = raw_keypoints[i]
            visible = 1 if v > threshold else 0
            keypoints.append((float(x), float(y), visible))
        else:
            keypoints.append((0, 0, 0))
    
    visible_count = sum(1 for _, _, v in keypoints if v > 0)
    return keypoints, visible_count

def reset_keypoints(keypoint_names):
    """Reset keypoints to empty state"""
    return [(0, 0, 0) for _ in keypoint_names]