import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from flask import Flask, render_template, request
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tensorflow as tf

# Configure TensorFlow GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ============================================
# FLASK APP CONFIGURATION
# ============================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion labels (must match training order)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

EMOTION_RESPONSES = {
    'angry': "You look angry. What's bothering you?",
    'disgust': "You seem disgusted. Is everything okay?",
    'fear': "You look scared. Don't worry, everything will be fine!",
    'happy': "You're smiling! Keep spreading that joy!",
    'sad': "You seem sad. Hope things get better soon!",
    'surprise': "You look surprised! What happened?",
    'neutral': "You have a neutral expression. Have a great day!"
}

# ============================================
# DATABASE
# ============================================
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            student_id TEXT,
            image_filename TEXT NOT NULL,
            emotion TEXT NOT NULL,
            emotion_response TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ============================================
# LOAD / CONVERT MODEL TO TFLITE
# ============================================
MODEL_PATH = 'face_emotionModel.h5'
TFLITE_MODEL_PATH = 'face_emotionModel.tflite'
GDRIVE_FILE_ID = '14c8yE75DnnVp8IKQf16FHIl_jAtstR8K'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading...")
        try:
            import gdown
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            print("Error downloading model:", e)
            return False
    return True

def ensure_tflite_model():
    """Convert .h5 to .tflite if not exists."""
    if not os.path.exists(TFLITE_MODEL_PATH):
        print("Converting Keras model to TensorFlow Lite...")
        model = load_model(MODEL_PATH)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(TFLITE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)
        print("✓ TFLite model created successfully!")

if download_model():
    ensure_tflite_model()
    # Load lightweight TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    print("✓ TFLite model loaded successfully!")
else:
    interpreter = None
    print("⚠ Model unavailable")

# ============================================
# HELPER FUNCTIONS
# ============================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def detect_emotion(image_path):
    """Detect emotion using the optimized TFLite model."""
    if interpreter is None:
        return 'neutral', 0.0

    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            face_img = cv2.resize(gray, (48, 48))
        else:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))

        img_array = face_img / 255.0
        img_array = img_array.astype(np.float32).reshape(1, 48, 48, 1)

        # Run inference with TFLite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        emotion = EMOTIONS[emotion_idx]

        return emotion, float(confidence)

    except Exception as e:
        print("Error detecting emotion:", e)
        return 'neutral', 0.0


def save_to_database(name, email, student_id, image_filename, emotion, emotion_response):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''
        INSERT INTO students (name, email, student_id, image_filename, emotion, emotion_response, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, email, student_id, image_filename, emotion, emotion_response, timestamp))
    conn.commit()
    conn.close()

# ============================================
# ROUTES
# ============================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    try:
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        student_id = request.form.get('student_id', '').strip()

        if not name or not email:
            return "Error: Name and email are required!", 400

        if 'image' not in request.files:
            return "Error: No image uploaded!", 400

        file = request.files['image']
        if file.filename == '':
            return "Error: No image selected!", 400

        if not allowed_file(file.filename):
            return "Error: Invalid file type! Please upload PNG, JPG, or JPEG.", 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        emotion, confidence = detect_emotion(filepath)
        emotion_response = EMOTION_RESPONSES[emotion]
        save_to_database(name, email, student_id, filename, emotion, emotion_response)

        # CSS left untouched
        result_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Result - Emotion Detection</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 20px;
                }}
                .container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    max-width: 600px;
                    width: 100%;
                    text-align: center;
                }}
                h1 {{
                    color: #667eea;
                    margin-bottom: 30px;
                    font-size: 2.5em;
                }}
                .result-image {{
                    max-width: 100%;
                    border-radius: 15px;
                    margin: 20px 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }}
                .emotion {{
                    font-size: 2em;
                    color: #764ba2;
                    font-weight: bold;
                    margin: 20px 0;
                    text-transform: capitalize;
                }}
                .response {{
                    font-size: 1.3em;
                    color: #555;
                    margin: 20px 0;
                    line-height: 1.6;
                }}
                .confidence {{
                    color: #888;
                    font-size: 1em;
                    margin: 10px 0;
                }}
                .student-info {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                    text-align: left;
                }}
                .student-info p {{
                    margin: 8px 0;
                    color: #555;
                }}
                .btn {{
                    display: inline-block;
                    padding: 15px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 50px;
                    font-size: 1.1em;
                    margin-top: 30px;
                    transition: transform 0.3s;
                }}
                .btn:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✨ Result</h1>
                <img src="/{filepath}" alt="Your photo" class="result-image">
                <div class="emotion">Detected Emotion: {emotion}</div>
                <div class="confidence">Confidence: {confidence*100:.1f}%</div>
                <div class="response">{emotion_response}</div>
                <div class="student-info">
                    <p><strong>Name:</strong> {name}</p>
                    <p><strong>Email:</strong> {email}</p>
                    {f'<p><strong>Student ID:</strong> {student_id}</p>' if student_id else ''}
                    <p><strong>Submitted:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <a href="/" class="btn">Submit Another</a>
            </div>
        </body>
        </html>
        '''
        return result_html

    except Exception as e:
        return f"Error processing request: {e}", 500

# ============================================
# RUN
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
