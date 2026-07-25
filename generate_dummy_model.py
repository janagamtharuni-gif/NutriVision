"""
generate_dummy_model.py
------------------------
Utility script to generate a dummy Keras model (.h5) and labels.txt file
for testing the Healthy Food Detection desktop app instantly.
"""

import os

def create_dummy_model():
    print("Generating test Keras model (keras_model.h5) and labels.txt...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("Error: TensorFlow is required to run this script. Run 'pip install tensorflow' first.")
        return

    # Build dummy Keras model matching 224x224x3 input shape and 6 output classes
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Save model as keras_model.h5
    model_path = "keras_model.h5"
    model.save(model_path)
    print(f"Created '{model_path}' successfully.")

    # Generate labels.txt
    labels = ["Apple", "Banana", "Orange", "Pizza", "Burger", "Chips"]
    labels_path = "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for idx, label in enumerate(labels):
            f.write(f"{idx} {label}\n")
            
    print(f"Created '{labels_path}' with classes: {', '.join(labels)}")
    print("Done! You can now run 'python main.py'.")

if __name__ == "__main__":
    create_dummy_model()
