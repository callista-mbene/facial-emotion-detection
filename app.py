from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# ============================================
# FLASK APP CONFIGURATION
# ============================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion labels (must match training order)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion responses (what to display to user)
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
# DATABASE SETUP
# ============================================
def init_db():
    """Initialize the database and create table if not exists"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
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
    print("✓ Database initialized successfully!")

# Initialize database when app starts
init_db()

# ============================================
# LOAD TRAINED MODEL
# ============================================
MODEL_PATH = 'face_emotionModel.h5'
GDRIVE_FILE_ID = '14c8yE75DnnVp8IKQf16FHIl_jAtstR8K'

def download_model():
    """Download model from Google Drive if not exists"""
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        try:
            import gdown
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            print("✓ Model downloaded successfully!")
        except Exception as e:
            print(f"⚠ Error downloading model: {e}")
            return False
    else:
        print("✓ Model found locally.")
    return True

print("Loading emotion detection model...")
try:
    # Download model if needed
    if download_model():
        model = load_model(MODEL_PATH)
        print("✓ Model loaded successfully!")
    else:
        model = None
        print("⚠ Model download failed. App will run without predictions.")
except Exception as e:
    print(f"⚠ Warning: Could not load model - {e}")
    print("The app will run, but predictions won't work until model is available.")
    model = None
    
# ============================================
# HELPER FUNCTIONS
# ============================================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_emotion(image_path):
    """
    Detect emotion from an image
    Returns: (emotion_label, confidence)
    """
    if model is None:
        return 'neutral', 0.0
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((48, 48))  # Resize to 48x48
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = img_array.reshape(1, 48, 48, 1)  # Reshape for model
        
        # Predict emotion
        predictions = model.predict(img_array, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        emotion = EMOTIONS[emotion_idx]
        
        return emotion, float(confidence)
    
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return 'neutral', 0.0

def save_to_database(name, email, student_id, image_filename, emotion, emotion_response):
    """Save student data to database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
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
    """Display the main form"""
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handle form submission"""
    try:
        # Get form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        student_id = request.form.get('student_id', '').strip()
        
        # Validate inputs
        if not name or not email:
            return "Error: Name and email are required!", 400
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return "Error: No image uploaded!", 400
        
        file = request.files['image']
        
        if file.filename == '':
            return "Error: No image selected!", 400
        
        if not allowed_file(file.filename):
            return "Error: Invalid file type! Please upload PNG, JPG, or JPEG.", 400
        
        # Save the uploaded image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect emotion
        emotion, confidence = detect_emotion(filepath)
        emotion_response = EMOTION_RESPONSES[emotion]
        
        # Save to database
        save_to_database(name, email, student_id, filename, emotion, emotion_response)
        
        # Create result page HTML
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
        return f"Error processing request: {str(e)}", 500

# ============================================
# RUN APP
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)