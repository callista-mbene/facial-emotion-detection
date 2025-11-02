import tensorflow as tf
import os
from tensorflow.keras.models import load_model

# Step 1: Load the existing Keras model (.h5 file)
print("🔹 Loading model...")
model = load_model('face_emotionModel.h5')

# Step 2: Convert model to TensorFlow Lite format
print("🔹 Converting model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization flag reduces size and memory usage
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Step 3: Convert and save
tflite_model = converter.convert()

with open('face_emotionModel_lite.tflite', 'wb') as f:
    f.write(tflite_model)

# Step 4: Print comparison sizes
old_size = os.path.getsize('face_emotionModel.h5') / 1024 / 1024
new_size = os.path.getsize('face_emotionModel_lite.tflite') / 1024 / 1024

print(f"✅ Conversion complete!")
print(f"📦 Original .h5 model size: {old_size:.2f} MB")
print(f"📦 Optimized .tflite model size: {new_size:.2f} MB")
print("➡️ New file saved as: face_emotionModel_lite.tflite")


# Optional: Quick test (replace with one of your images)
print("🔹 Testing TFLite model on sample input...")

interpreter = tf.lite.Interpreter(model_path='face_emotionModel_lite.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create dummy data (shape must match model input)
import numpy as np
dummy_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print("✅ Model ran successfully! Output shape:", output.shape)
