import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
from PIL import Image, ImageTk

from model_loader import load_trouser_model
from image_manager import select_image, upload_folder
from keypoint_operations import predict_on_resized, apply_threshold, reset_keypoints, draw_keypoints
from utils.annotation_utils import save_to_deepfashion_format, show_save_location, check_annotation_exists, load_annotation_file

class KeypointAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Trouser Keypoint Annotation Tool")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.canvas_image = None
        self.keypoints = []
        self.keypoint_circles = []
        self.selected_keypoint = None
        self.predictor = None
        self.scale_factor = 1.0
        self.raw_keypoints = []
        self.image_list = []
        self.image_index = -1
        self.process_size = (256, 256)
        
        # Keypoint names for trousers
        self.keypoint_names = [
            "right_hip", "waist_center", "left_hip", "right_thigh", "right_knee", "right_ankle", "right_hem",
            "right_pocket", "crotch", "left_pocket", "left_hem", "left_ankle", "left_knee", "left_thigh"
        ]
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        control_frame1 = ttk.Frame(control_frame)
        control_frame1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # First row of buttons
        button_frame1 = ttk.Frame(control_frame1)
        button_frame1.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(button_frame1, text="Select Image", command=self.select_image_wrapper).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="Upload Folder", command=self.upload_folder_wrapper).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="← Prev", command=self.prev_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="Next →", command=self.next_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="↺ Rotate Left", command=lambda: self.rotate_image("left")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="↻ Rotate Right", command=lambda: self.rotate_image("right")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="Predict on Resized", command=self.predict_on_resized_wrapper).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="Reset", command=self.reset_keypoints_wrapper).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="Save DeepFashion2", command=self.save_to_deepfashion_format_wrapper).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame1, text="Show Save Location", command=self.show_save_location_wrapper).pack(side=tk.LEFT, padx=(0, 10))
        
        # Second row with visibility threshold
        button_frame2 = ttk.Frame(control_frame1)
        button_frame2.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        ttk.Label(button_frame2, text="Visibility Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.visibility_threshold = tk.DoubleVar(value=0.001)
        threshold_spinbox = ttk.Spinbox(button_frame2, from_=0.0, to=1.0, increment=0.001, 
                                       textvariable=self.visibility_threshold, width=10)
        threshold_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame2, text="Apply Threshold", command=self.apply_threshold_wrapper).pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready - Select an image to begin")
        self.status_label.pack(side=tk.RIGHT)
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas frame
        canvas_frame = ttk.Frame(content_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='gray', cursor="cross")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        
        # Keypoints panel
        keypoints_frame = ttk.LabelFrame(content_frame, text="Keypoints", width=250)
        keypoints_frame.pack(side=tk.RIGHT, fill=tk.Y)
        keypoints_frame.pack_propagate(False)
        
        # Keypoints listbox
        self.keypoints_listbox = tk.Listbox(keypoints_frame, selectmode=tk.SINGLE)
        self.keypoints_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.keypoints_listbox.bind("<<ListboxSelect>>", self.on_keypoint_select)
        
        # Keypoint info frame
        info_frame = ttk.Frame(keypoints_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Selected Keypoint:").pack(anchor=tk.W)
        self.selected_label = ttk.Label(info_frame, text="None", foreground="blue")
        self.selected_label.pack(anchor=tk.W)
        
        ttk.Label(info_frame, text="Coordinates:").pack(anchor=tk.W, pady=(10, 0))
        self.coords_label = ttk.Label(info_frame, text="(0, 0)")
        self.coords_label.pack(anchor=tk.W)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(keypoints_frame, text="Instructions")
        instructions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        instructions = """1. Select an image using navigation or upload
2. If needed, rotate the image before predicting
3. Click 'Predict on Resized' to get keypoints
4. Click on keypoints to select them
5. Drag selected keypoints to adjust positions
6. When done, click 'Save DeepFashion2' to save"""
        ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT, wraplength=200).pack(padx=5, pady=5)

    def load_model(self):
        """Load the custom Detectron2 trousers keypoint model"""
        self.predictor = load_trouser_model(self.status_label)

    # Wrapper methods
    def select_image_wrapper(self):
        file_path = select_image()
        if file_path:
            self.load_image(file_path)

    def upload_folder_wrapper(self):
        folder_path = upload_folder()
        if folder_path:
            self.image_list = folder_path
            self.image_index = 0
            self.load_image(self.image_list[self.image_index])
            self.status_label.config(text=f"Loaded folder with {len(self.image_list)} images")

    def next_image(self):
        if self.image_list and self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.load_image(self.image_list[self.image_index])
            self.status_label.config(text=f"Image {self.image_index + 1}/{len(self.image_list)}")

    def prev_image(self):
        if self.image_list and self.image_index > 0:
            self.image_index -= 1
            self.load_image(self.image_list[self.image_index])
            self.status_label.config(text=f"Image {self.image_index + 1}/{len(self.image_list)}")

    def rotate_image(self, direction):
        if self.original_image is None:
            return
        
        if direction == "left":
            self.original_image = cv2.rotate(self.original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            self.original_image = cv2.rotate(self.original_image, cv2.ROTATE_90_CLOCKWISE)
            
        # Reset keypoints after rotation
        self.keypoints = [(0, 0, 0) for _ in self.keypoint_names]
        self.display_image_on_canvas()
        self.status_label.config(text=f"Image rotated {direction}")

    def predict_on_resized_wrapper(self):
        if self.original_image is None or self.predictor is None:
            messagebox.showwarning("Warning", "Please load an image and ensure model is loaded")
            return
        
        result = predict_on_resized(
            self.original_image, self.predictor, self.keypoint_names, 
            self.visibility_threshold.get(), self.process_size
        )
        
        if result:
            self.keypoints, self.raw_keypoints, visible_count = result
            self.update_keypoints_list()
            self.display_image_on_canvas()
            self.status_label.config(text=f"Predicted on resized - {visible_count} keypoints visible")

    def apply_threshold_wrapper(self):
        if not hasattr(self, 'raw_keypoints') or not self.raw_keypoints:
            messagebox.showwarning("Warning", "No keypoints to apply threshold to")
            return
            
        self.keypoints, visible_count = apply_threshold(
            self.raw_keypoints, self.visibility_threshold.get(), self.keypoint_names
        )
        self.update_keypoints_list()
        self.display_image_on_canvas()
        self.status_label.config(text=f"Threshold applied - {visible_count} visible")

    def reset_keypoints_wrapper(self):
        self.keypoints = reset_keypoints(self.keypoint_names)
        self.selected_keypoint = None
        self.selected_label.config(text="None")
        self.coords_label.config(text="(0, 0)")
        self.keypoints_listbox.delete(0, tk.END)
        if self.original_image is not None:
            self.display_image_on_canvas()

    def save_to_deepfashion_format_wrapper(self):
        if not self.keypoints or not self.image_path:
            messagebox.showwarning("Warning", "No keypoints or image to save")
            return
        
        visible_count = save_to_deepfashion_format(
            self.image_path, self.original_image, self.keypoints
        )
        self.status_label.config(text=f"Saved to DeepFashion2 format - {visible_count} keypoints")

    def show_save_location_wrapper(self):
        show_save_location(self.image_path)

    def load_image(self, file_path):
        """Load and display the selected image"""
        try:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                raise ValueError("Failed to load image")
                
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            # Calculate scale factor for canvas display
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = self.original_image.shape[:2]
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                self.scale_factor = min(scale_x, scale_y, 1.0)
            else:
                self.scale_factor = 1.0

            # Check for existing annotation
            annotation_path = check_annotation_exists(file_path)
            
            if annotation_path and load_annotation_file(annotation_path, self.keypoints):
                base_name = os.path.basename(file_path)
                self.status_label.config(text=f"Loaded with annotations: {base_name}")
                self.update_keypoints_list()
            else:
                self.keypoints = [(0, 0, 0) for _ in self.keypoint_names]
                base_name = os.path.basename(file_path)
                self.status_label.config(text=f"Image loaded: {base_name} (no annotations)")

            self.display_image_on_canvas()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def display_image_on_canvas(self):
        """Display the image on canvas with current keypoints"""
        if self.original_image is None:
            return
        
        # Scale image for display
        img_height, img_width = self.original_image.shape[:2]
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        display_img = cv2.resize(self.original_image, (new_width, new_height))
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(display_img)
        self.canvas_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)
        
        # Update canvas scroll region
        self.canvas.configure(scrollregion=(0, 0, new_width, new_height))
        
        # Draw keypoints
        draw_keypoints(self.canvas, self.keypoints, self.keypoint_names, self.scale_factor, self.selected_keypoint)

    def update_keypoints_list(self):
        """Update the keypoints listbox"""
        self.keypoints_listbox.delete(0, tk.END)
        for i, (x, y, visible) in enumerate(self.keypoints):
            status = "✓" if visible else "✗"
            self.keypoints_listbox.insert(tk.END, f"{status} {self.keypoint_names[i]}: ({int(x)}, {int(y)})")

    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if not self.keypoints:
            return
        
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Find nearest keypoint
        min_distance = float('inf')
        nearest_keypoint = None
        
        for i, (x, y, visible) in enumerate(self.keypoints):
            if visible:
                display_x = x * self.scale_factor
                display_y = y * self.scale_factor
                distance = ((canvas_x - display_x) ** 2 + (canvas_y - display_y) ** 2) ** 0.5
                if distance < 20 and distance < min_distance:
                    min_distance = distance
                    nearest_keypoint = i
        
        if nearest_keypoint is not None:
            self.selected_keypoint = nearest_keypoint
            self.keypoints_listbox.selection_clear(0, tk.END)
            self.keypoints_listbox.selection_set(nearest_keypoint)
            self.selected_label.config(text=self.keypoint_names[nearest_keypoint])
            x, y, _ = self.keypoints[nearest_keypoint]
            self.coords_label.config(text=f"({int(x)}, {int(y)})")
            self.display_image_on_canvas()

    def on_canvas_drag(self, event):
        """Handle canvas drag events"""
        if self.selected_keypoint is not None:
            # Get canvas coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Convert to original image coordinates
            original_x = canvas_x / self.scale_factor
            original_y = canvas_y / self.scale_factor
            
            # Update keypoint position
            self.keypoints[self.selected_keypoint] = (original_x, original_y, 1)
            
            # Update display
            self.coords_label.config(text=f"({int(original_x)}, {int(original_y)})")
            self.update_keypoints_list()
            self.display_image_on_canvas()

    def on_canvas_release(self, event):
        """Handle canvas release events"""
        pass

    def on_canvas_motion(self, event):
        """Handle canvas motion events"""
        if self.selected_keypoint is not None:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            original_x = canvas_x / self.scale_factor
            original_y = canvas_y / self.scale_factor
            self.coords_label.config(text=f"({int(original_x)}, {int(original_y)})")

    def on_keypoint_select(self, event):
        """Handle keypoint selection from listbox"""
        selection = self.keypoints_listbox.curselection()
        if selection:
            self.selected_keypoint = selection[0]
            self.selected_label.config(text=self.keypoint_names[self.selected_keypoint])
            x, y, _ = self.keypoints[self.selected_keypoint]
            self.coords_label.config(text=f"({int(x)}, {int(y)})")
            self.display_image_on_canvas()

def main():
    root = tk.Tk()
    app = KeypointAnnotator(root)
    root.mainloop()