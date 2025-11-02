import tensorflow as tf

model = tf.keras.models.load_model("face_emotionModel.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("face_emotionModel_lite.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved: face_emotionModel_lite.tflite")
