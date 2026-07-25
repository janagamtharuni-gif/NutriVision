"""
main.py
-------
Healthy Food Detection Using AI - Desktop Tkinter Application

Captures live video feed from webcam using OpenCV, performs real-time frame
classification using a trained TensorFlow/Keras model (.h5), and displays
nutritional breakdown and health status (Healthy / Unhealthy).
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

# Import local modules
from model_handler import FoodClassifier
from nutrition_data import get_food_info


class HealthyFoodApp:
    """Main Tkinter Desktop Application Class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NutriVision - Healthy Food Detection Using AI")
        self.root.geometry("1024x680")
        self.root.minsize(900, 600)
        self.root.configure(bg="#f4f6f9")

        # Application state variables
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        self.after_id = None

        # Initialize TensorFlow Classifier
        self.classifier = FoodClassifier(model_path="keras_model.h5", labels_path="labels.txt")
        self.model_loaded = False

        # Build GUI Layout
        self._setup_styles()
        self._build_header()
        self._build_main_content()
        self._build_footer()

        # Handle window close event safely
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Attempt deferred model loading so GUI appears instantly
        self.root.after(200, self.init_model)

    def _setup_styles(self):
        """Configures ttk widget styles and color palette."""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Color Palette
        self.COLOR_BG = "#f4f6f9"
        self.COLOR_PRIMARY = "#2563eb"     # Blue
        self.COLOR_SUCCESS = "#16a34a"     # Green (Healthy)
        self.COLOR_DANGER = "#dc2626"      # Red (Unhealthy)
        self.COLOR_TEXT_DARK = "#0f172a"
        self.COLOR_TEXT_MUTED = "#64748b"
        self.COLOR_CARD = "#ffffff"

        self.style.configure("TFrame", background=self.COLOR_BG)
        self.style.configure("Card.TFrame", background=self.COLOR_CARD, relief="flat")

    def _build_header(self):
        """Creates top navigation/header bar."""
        header = tk.Frame(self.root, bg="#1e293b", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header,
            text="🥗 NutriVision",
            font=("Helvetica", 18, "bold"),
            fg="#ffffff",
            bg="#1e293b",
            padx=20,
            pady=12
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(
            header,
            text="Real-time TensorFlow Vision & Nutrition Assistant",
            font=("Helvetica", 11, "italic"),
            fg="#94a3b8",
            bg="#1e293b",
            padx=20
        )
        subtitle_label.pack(side=tk.RIGHT)

    def _build_main_content(self):
        """Creates main grid split: Camera viewport (Left) & Nutrition details (Right)."""
        main_container = tk.Frame(self.root, bg=self.COLOR_BG, padx=15, pady=15)
        main_container.pack(fill=tk.BOTH, expand=True)

        main_container.columnconfigure(0, weight=3) # Camera view
        main_container.columnconfigure(1, weight=2) # Nutrition Panel
        main_container.rowconfigure(0, weight=1)

        # ---------------- LEFT PANEL: CAMERA VIEWPORT ----------------
        left_panel = tk.Frame(main_container, bg=self.COLOR_CARD, bd=1, relief="solid")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Camera Title
        cam_title_frame = tk.Frame(left_panel, bg="#f8fafc", pady=8, padx=10)
        cam_title_frame.pack(fill=tk.X)
        
        self.cam_status_indicator = tk.Label(
            cam_title_frame,
            text="● Offline",
            font=("Helvetica", 10, "bold"),
            fg="#94a3b8",
            bg="#f8fafc"
        )
        self.cam_status_indicator.pack(side=tk.RIGHT, padx=10)

        cam_label_title = tk.Label(
            cam_title_frame,
            text="Live Camera Stream",
            font=("Helvetica", 12, "bold"),
            fg=self.COLOR_TEXT_DARK,
            bg="#f8fafc"
        )
        cam_label_title.pack(side=tk.LEFT)

        # Video Frame Label Container
        self.video_label = tk.Label(
            left_panel,
            text="Camera feed is stopped.\nClick 'Start Camera' below to launch webcam.",
            font=("Helvetica", 12),
            fg=self.COLOR_TEXT_MUTED,
            bg="#0f172a",
            width=60,
            height=20
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control Buttons Frame
        btn_frame = tk.Frame(left_panel, bg=self.COLOR_CARD, pady=10)
        btn_frame.pack(fill=tk.X, padx=10)

        self.btn_start = tk.Button(
            btn_frame,
            text="▶ Start Camera",
            font=("Helvetica", 11, "bold"),
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            activeforeground="#ffffff",
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
            command=self.start_camera
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_detect = tk.Button(
            btn_frame,
            text="🔍 Detect Food",
            font=("Helvetica", 11, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.detect_food
        )
        self.btn_detect.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(
            btn_frame,
            text="⏹ Stop Camera",
            font=("Helvetica", 11, "bold"),
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.stop_camera
        )
        self.btn_stop.pack(side=tk.RIGHT, padx=5)

        # ---------------- RIGHT PANEL: NUTRITION & RESULTS ----------------
        right_panel = tk.Frame(main_container, bg=self.COLOR_CARD, bd=1, relief="solid", padx=15, pady=15)
        right_panel.grid(row=0, column=1, sticky="nsew")

        panel_header = tk.Label(
            right_panel,
            text="Nutritional Analysis",
            font=("Helvetica", 14, "bold"),
            fg=self.COLOR_TEXT_DARK,
            bg=self.COLOR_CARD
        )
        panel_header.pack(anchor="w", pady=(0, 10))

        # Status / Tag Badge
        self.badge_frame = tk.Frame(right_panel, bg="#e2e8f0", padx=12, pady=6)
        self.badge_frame.pack(fill=tk.X, pady=(0, 15))

        self.lbl_category_badge = tk.Label(
            self.badge_frame,
            text="Awaiting Detection",
            font=("Helvetica", 12, "bold"),
            fg="#475569",
            bg="#e2e8f0"
        )
        self.lbl_category_badge.pack()

        # Detected Food Title
        self.lbl_food_name = tk.Label(
            right_panel,
            text="No Food Detected Yet",
            font=("Helvetica", 16, "bold"),
            fg=self.COLOR_TEXT_DARK,
            bg=self.COLOR_CARD,
            anchor="w"
        )
        self.lbl_food_name.pack(fill=tk.X, pady=(0, 5))

        # Confidence Bar / Score
        self.lbl_confidence = tk.Label(
            right_panel,
            text="Confidence: -- %",
            font=("Helvetica", 10, "bold"),
            fg=self.COLOR_TEXT_MUTED,
            bg=self.COLOR_CARD,
            anchor="w"
        )
        self.lbl_confidence.pack(fill=tk.X, pady=(0, 15))

        # Separator
        ttk.Separator(right_panel, orient="horizontal").pack(fill=tk.X, pady=10)

        # Calories Block
        tk.Label(
            right_panel,
            text="CALORIC ESTIMATE",
            font=("Helvetica", 9, "bold"),
            fg=self.COLOR_TEXT_MUTED,
            bg=self.COLOR_CARD,
            anchor="w"
        ).pack(fill=tk.X)

        self.lbl_calories = tk.Label(
            right_panel,
            text="-- kcal",
            font=("Helvetica", 13, "bold"),
            fg="#1e293b",
            bg=self.COLOR_CARD,
            anchor="w",
            pady=4
        )
        self.lbl_calories.pack(fill=tk.X, pady=(0, 15))

        # Benefits & Notes Block
        tk.Label(
            right_panel,
            text="NUTRITIONAL BENEFITS & NOTES",
            font=("Helvetica", 9, "bold"),
            fg=self.COLOR_TEXT_MUTED,
            bg=self.COLOR_CARD,
            anchor="w"
        ).pack(fill=tk.X)

        self.lbl_benefits = tk.Label(
            right_panel,
            text="Place a food item (Apple, Banana, Orange, Pizza, Burger, or Chips) in front of the camera and click 'Detect Food'.",
            font=("Helvetica", 10),
            fg="#334155",
            bg="#f8fafc",
            anchor="nw",
            justify=tk.LEFT,
            wraplength=320,
            padx=10,
            pady=10,
            relief="flat",
            bd=1
        )
        self.lbl_benefits.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def _build_footer(self):
        """Creates bottom status bar."""
        footer = tk.Frame(self.root, bg="#e2e8f0", height=25)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_bar = tk.Label(
            footer,
            text="System initializing...",
            font=("Helvetica", 9),
            fg=self.COLOR_TEXT_MUTED,
            bg="#e2e8f0",
            anchor="w",
            padx=10
        )
        self.status_bar.pack(fill=tk.X)

    def init_model(self):
        """Loads TensorFlow model and class labels asynchronously."""
        self.status_bar.config(text="Loading TensorFlow Keras model (keras_model.h5)...")
        success, message = self.classifier.load_model_and_labels()

        if success:
            self.model_loaded = True
            self.status_bar.config(text=f"Ready. {message}")
        else:
            self.model_loaded = False
            self.status_bar.config(text="Model Warning: Model file not loaded.")
            messagebox.showwarning(
                "Model Not Found",
                f"{message}\n\n"
                "To fix this:\n"
                "1. Train a model on Google Teachable Machine (6 classes: Apple, Banana, Orange, Pizza, Burger, Chips).\n"
                "2. Export as Keras (.h5) format.\n"
                "3. Place 'keras_model.h5' and 'labels.txt' in this application folder.\n"
                "4. Or run 'python generate_dummy_model.py' to generate a test model!"
            )

    def start_camera(self):
        """Initializes OpenCV video capture device (Webcam)."""
        if self.is_camera_running:
            return

        # Connect to default camera index 0
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Unable to open video camera feed (index 0).\n"
                "Please ensure your webcam is connected and not in use by another application."
            )
            self.cap = None
            return

        self.is_camera_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        if self.model_loaded:
            self.btn_detect.config(state=tk.NORMAL)

        self.cam_status_indicator.config(text="● Live Stream", fg="#16a34a")
        self.status_bar.config(text="Camera active. Position food item in viewport.")
        
        # Start smooth video frame streaming loop
        self.update_frame()

    def update_frame(self):
        """
        Continuously reads frames from OpenCV webcam, converts BGR -> RGB,
        wraps with PIL Image/ImageTk, and renders on Tkinter Label widget.
        """
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Store current raw frame for detection
            self.current_frame = frame.copy()

            # BGR -> RGB color conversion to prevent inverted blue skin/colors
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit display viewport cleanly (e.g. 560x380)
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.resize((560, 380), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update Label widget image reference
            self.video_label.img_tk = img_tk  # keep reference to avoid garbage collection
            self.video_label.config(image=img_tk, text="")

        # Schedule next frame update after 15ms (~60 FPS)
        self.after_id = self.root.after(15, self.update_frame)

    def detect_food(self):
        """Passes current captured frame to classifier and renders nutrition report."""
        if not self.model_loaded:
            messagebox.showwarning("Model Missing", "Model is not loaded. Please ensure 'keras_model.h5' and 'labels.txt' exist.")
            return

        if self.current_frame is None:
            messagebox.showwarning("Frame Missing", "No camera frame captured yet. Start camera first.")
            return

        try:
            self.status_bar.config(text="Analyzing image frame with TensorFlow model...")
            
            # Predict top class and confidence
            raw_label, confidence = self.classifier.predict(self.current_frame)
            
            # Lookup nutritional facts from database
            food_info = get_food_info(raw_label)

            # Update GUI Elements
            self.lbl_food_name.config(text=f"{food_info.get('icon', '🥗')} {food_info['name']}")
            self.lbl_confidence.config(text=f"Model Confidence: {confidence:.2f}%")
            self.lbl_calories.config(text=food_info['calories'])
            self.lbl_benefits.config(text=food_info['benefits'])

            # Render Health Status Category Tag Badge
            category = food_info['category']
            if category == "Healthy":
                self.badge_frame.config(bg="#dcfce7")
                self.lbl_category_badge.config(text="HEALTHY CHOICE  ✓", fg="#15803d", bg="#dcfce7")
            elif category == "Unhealthy":
                self.badge_frame.config(bg="#fee2e2")
                self.lbl_category_badge.config(text="UNHEALTHY / PROCESS  ⚠", fg="#b91c1c", bg="#fee2e2")
            else:
                self.badge_frame.config(bg="#e2e8f0")
                self.lbl_category_badge.config(text="UNCLASSIFIED ITEM", fg="#475569", bg="#e2e8f0")

            self.status_bar.config(text=f"Detected '{food_info['name']}' ({confidence:.1f}% confidence).")

            # Automatically end camera after capturing image frame
            if self.is_camera_running:
                self.stop_camera()

        except Exception as e:
            messagebox.showerror("Inference Error", f"Failed to perform classification: {str(e)}")
            self.status_bar.config(text="Classification failed.")

    def stop_camera(self):
        """Safely stops webcam loop and releases hardware resources."""
        self.is_camera_running = False

        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Reset Controls and Labels
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_detect.config(state=tk.DISABLED)
        
        self.cam_status_indicator.config(text="● Offline", fg="#94a3b8")
        self.video_label.config(
            image="",
            text="Camera feed stopped.\nClick 'Start Camera' to resume webcam streaming."
        )
        self.status_bar.config(text="Camera feed stopped.")

    def on_closing(self):
        """Clean shutdown handler on window close."""
        self.stop_camera()
        self.root.destroy()


def main():
    """Application entry point."""
    root = tk.Tk()
    app = HealthyFoodApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
