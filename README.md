# 🥗 NutriVision - Healthy Food Detection Using AI

A desktop AI application built with **Python**, **Tkinter**, **OpenCV**, and **TensorFlow / Keras** that classifies food items in real-time using your computer's webcam and displays detailed nutritional information (calories, benefits, and Healthy/Unhealthy classification).

---

## 🌟 Features

- **Real-Time Webcam Streaming:** High-performance live video preview rendered inside Tkinter GUI with smooth 60 FPS BGR->RGB color processing.
- **AI Food Classification:** Utilizes a custom-trained Keras Deep Learning model (`keras_model.h5`) for 6 food classes:
  - 🍎 **Apple** (Healthy - 95 kcal)
  - 🍌 **Banana** (Healthy - 105 kcal)
  - 🍊 **Orange** (Healthy - 62 kcal)
  - 🍕 **Pizza** (Unhealthy - 285 kcal)
  - 🍔 **Burger** (Unhealthy - 354 kcal)
  - 🍟 **Chips** (Unhealthy - 152 kcal)
- **Instant Nutritional Analysis:** Shows confidence percentage, calorie estimates, key vitamins/antioxidants, and health warnings.
- **Offline Capable:** Runs 100% locally on your computer with zero cloud dependency once setup is complete.
- **Error Handling:** Friendly alert messages for missing camera, missing model files, or hardware disconnects.

---

## 📁 Project Structure

```
.
├── main.py                # Main Tkinter desktop GUI app & camera thread loop
├── model_handler.py       # TensorFlow/Keras FoodClassifier class & image preprocessing
├── nutrition_data.py      # Food nutrition dictionary and helper queries
├── generate_dummy_model.py# Helper script to test GUI without Teachable Machine
├── requirements.txt       # Python package dependencies
├── README.md              # Complete setup documentation
├── keras_model.h5         # Trained Keras model (place in root after export)
└── labels.txt             # Model class labels (place in root after export)
```

---

## 🚀 Step 1: Installation & Setup

### Prerequisites
- **Python 3.9, 3.10, or 3.11** installed on your system.
- Computer webcam or external USB camera.

### Steps:
1. Clone or download this project folder to your local computer.
2. Open your terminal / command prompt in the project folder.
3. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   # On Windows:
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Step 2: Training Your Model on Google Teachable Machine (Free)

Follow these simple steps to generate your `keras_model.h5` and `labels.txt` files:

1. **Open Google Teachable Machine:**
   Navigating to [Teachable Machine by Google](https://teachablemachine.withgoogle.com/) and select **Image Project** -> **Standard Image Model**.

2. **Create the 6 Classes:**
   Rename the class labels to match the exact names:
   - `Class 1`: `Apple`
   - `Class 2`: `Banana`
   - `Class 3`: `Orange`
   - `Class 4`: `Pizza`
   - `Class 5`: `Burger`
   - `Class 6`: `Chips`

3. **Collect Training Data:**
   - Click the **Webcam** button for each class and hold up the real food item (or show images on your phone screen) while recording 50-100 image samples per class.
   - Alternatively, upload 30-50 JPEG/PNG images per category from Google Images.

4. **Train the Model:**
   - Click **Train Model** (leave default Epochs = 50, Batch Size = 16).
   - Wait 1-2 minutes for browser training to complete.

5. **Export Model as Keras (.h5):**
   - Click **Export Model** at top right.
   - Select the **TensorFlow** tab.
   - Choose **Keras** export format.
   - Click **Download my model**.

6. **Place Downloaded Files in Project Root:**
   - Extract the downloaded `.zip` file.
   - Copy both `keras_model.h5` and `labels.txt` directly into the root directory of this project alongside `main.py`.

> 💡 **Quick Test Mode:** Don't have a model ready yet? Run `python generate_dummy_model.py` in your terminal to instantly auto-generate a valid test `keras_model.h5` and `labels.txt` so you can test run `main.py` right away!

---

## ▶️ Step 3: Running the Application

Execute the application using python:

```bash
python main.py
```

### Usage Instructions:
1. Click **▶ Start Camera** to connect to your webcam.
2. Hold a food item (Apple, Banana, Orange, Pizza, Burger, or Chips) in front of the lens.
3. Click **🔍 Detect Food** to freeze frame analysis and view nutritional details.
4. Click **⏹ Stop Camera** when finished to safely release camera hardware resources.

---

## 🛠 Tech Stack Details

- **Python 3.9+** - Core language.
- **Tkinter** - Built-in native cross-platform GUI framework.
- **OpenCV (`opencv-python`)** - Frame acquisition from webcam & BGR color space operations.
- **TensorFlow / Keras** - Deep Learning inference engine for `keras_model.h5`.
- **Pillow (PIL)** - Converts OpenCV NumPy arrays to Tkinter-compatible images.
- **NumPy** - Vectorized matrix normalization and `(1, 224, 224, 3)` array reshaping.
