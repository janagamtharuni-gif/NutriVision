"""
model_handler.py
----------------
TensorFlow/Keras Model Loading and Inference Handler for Food Detection.

Loads trained Keras model (.h5 format) and class labels (labels.txt),
preprocesses OpenCV video frames (BGR -> RGB, resize, normalize),
and performs prediction.
"""

import os
from typing import Tuple, List, Optional
import numpy as np

# Suppress TensorFlow logging verbosity before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None

try:
    import cv2
except ImportError:
    cv2 = None


class FoodClassifier:
    """
    Handles TensorFlow model loading, image preprocessing, and classification.
    """

    def __init__(self, model_path: str = "keras_model.h5", labels_path: str = "labels.txt", input_size: Tuple[int, int] = (224, 224)):
        """
        Initializes classifier paths and attributes.

        Args:
            model_path (str): Path to trained Keras model file (.h5)
            labels_path (str): Path to class labels text file
            input_size (Tuple[int, int]): Expected image input dimensions (width, height)
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.input_size = input_size
        self.model = None
        self.labels: List[str] = []
        self.is_loaded = False

    def load_model_and_labels(self) -> Tuple[bool, str]:
        """
        Loads the Keras .h5 model file and reads labels.txt.

        Returns:
            Tuple[bool, str]: (Success status, status message or error details)
        """
        if tf is None or load_model is None:
            return False, "TensorFlow package is not installed. Please install requirements."

        # Check model file existence
        if not os.path.exists(self.model_path):
            return False, f"Model file '{self.model_path}' not found in project directory."

        # Check labels file existence
        if not os.path.exists(self.labels_path):
            return False, f"Labels file '{self.labels_path}' not found in project directory."

        try:
            # Load Keras Model (compile=False avoids potential custom optimizer errors)
            self.model = load_model(self.model_path, compile=False)

            # Read Labels
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]

            if not self.labels:
                return False, "Labels file 'labels.txt' is empty."

            self.is_loaded = True
            return True, f"Successfully loaded model and {len(self.labels)} class labels."

        except Exception as e:
            self.is_loaded = False
            return False, f"Failed to load model: {str(e)}"

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocesses OpenCV BGR frame for Teachable Machine / Keras model.
        
        Steps:
        1. Convert BGR to RGB color space.
        2. Resize image to model expected dimensions (224x224).
        3. Convert pixel values to float32.
        4. Normalize values to [-1, 1] range: (img / 127.5) - 1.0
        5. Add batch dimension -> shape: (1, 224, 224, 3)

        Args:
            frame_bgr (np.ndarray): Original OpenCV frame in BGR format.

        Returns:
            np.ndarray: Preprocessed 4D numpy array ready for inference.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for frame preprocessing.")

        # 1. BGR -> RGB conversion
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize frame
        resized = cv2.resize(frame_rgb, self.input_size, interpolation=cv2.INTER_AREA)

        # 3 & 4. Normalize pixel values to range [-1, 1] (Teachable Machine standard)
        normalized = (resized.astype(np.float32) / 127.5) - 1.0

        # 5. Expand batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        batch_data = np.expand_dims(normalized, axis=0)

        return batch_data

    def predict(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Classifies an OpenCV video frame.

        Args:
            frame_bgr (np.ndarray): OpenCV image frame in BGR format.

        Returns:
            Tuple[Optional[str], float]: (Predicted raw class label string, confidence score 0-100%)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model_and_labels() first.")

        # Preprocess frame
        input_data = self.preprocess_frame(frame_bgr)

        # Perform inference
        predictions = self.model.predict(input_data, verbose=0)
        
        # Get top class index and probability
        top_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][top_idx]) * 100.0

        # Extract predicted class label
        if top_idx < len(self.labels):
            raw_label = self.labels[top_idx]
        else:
            raw_label = f"Class_{top_idx}"

        return raw_label, round(confidence, 2)


if __name__ == "__main__":
    # Diagnostic test script
    classifier = FoodClassifier()
    success, msg = classifier.load_model_and_labels()
    print(f"Status: {success} | Message: {msg}")
