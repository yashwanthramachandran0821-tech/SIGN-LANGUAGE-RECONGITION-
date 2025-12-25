from flask import Flask, render_template, request, jsonify, Response, session
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from datetime import datetime
import os
import threading

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'sign_language_translator_secret_key'
CORS(app)

# Global variables
model = None
mp_hands = mp.solutions.hands
hands = None
camera = None
prediction_history = []
current_translation = ""
is_recording = False

# ASL class names
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'space', 'del', 'nothing']

def init_model():
    """Initialize the ASL recognition model"""
    global model, hands
    
    try:
        # Load model
        if os.path.exists('models/asl_model.h5'):
            model = keras.models.load_model('models/asl_model.h5')
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Model not found. Using MediaPipe only.")
            model = None
        
        # Initialize MediaPipe
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        hands = None

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_np = image
    
    # Resize
    image_np = cv2.resize(image_np, (64, 64))
    
    # Normalize
    image_np = image_np.astype('float32') / 255.0
    
    # Expand dimensions
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

def detect_hand_landmarks(image):
    """Detect hand landmarks using MediaPipe"""
    if hands is None:
        return None, image
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    results = hands.process(rgb_image)
    
    # Draw landmarks
    annotated_image = image.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
    
    return results, annotated_image

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sign from uploaded image"""
    try:
        # Check if image is provided
        if 'image' not in request.files and 'image_data' not in request.form:
            return jsonify({'error': 'No image provided', 'success': False}), 400
        
        # Get image data
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image selected', 'success': False}), 400
            
            # Read image
            image = Image.open(file.stream)
        else:
            # Base64 encoded image from webcam
            image_data = request.form['image_data']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert to OpenCV format
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detect hand landmarks
        landmarks, annotated_image = detect_hand_landmarks(image_np)
        
        # Make prediction
        if model is not None:
            # Preprocess for model
            processed_image = preprocess_image(image_np)
            
            # Predict
            predictions = model.predict(processed_image, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_class = CLASS_NAMES[predicted_idx]
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = [
                {'class': CLASS_NAMES[i], 'confidence': float(predictions[0][i])}
                for i in top_indices
            ]
        else:
            # Fallback to simple gesture detection
            predicted_class, confidence = detect_simple_gesture(landmarks)
            top_predictions = [{'class': predicted_class, 'confidence': confidence}]
        
        # Store in session history
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append({
            'timestamp': datetime.now().isoformat(),
            'prediction': predicted_class,
            'confidence': confidence,
            'image': base64.b64encode(cv2.imencode('.jpg', annotated_image)[1]).decode()
        })
        
        # Keep only last 20 predictions
        if len(session['history']) > 20:
            session['history'] = session['history'][-20:]
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'hand_detected': landmarks is not None and landmarks.multi_hand_landmarks is not None,
            'annotated_image': base64.b64encode(cv2.imencode('.jpg', annotated_image)[1]).decode() if landmarks else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

def detect_simple_gesture(landmarks):
    """Simple gesture detection using MediaPipe landmarks"""
    if landmarks is None or not landmarks.multi_hand_landmarks:
        return 'nothing', 0.0
    
    # Simplified gesture detection
    hand_landmarks = landmarks.multi_hand_landmarks[0]
    
    # Get finger states
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]
    
    extended_fingers = 0
    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            extended_fingers += 1
    
    # Map to gestures
    if extended_fingers == 0:
        return 'A', 0.8
    elif extended_fingers == 5:
        return 'B', 0.7
    elif extended_fingers == 2 and hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        return 'V', 0.75
    else:
        return 'unknown', 0.5

@app.route('/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    history = session.get('history', [])
    return jsonify({'history': history, 'success': True})

@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate text to sign language instructions"""
    try:
        data = request.get_json()
        text = data.get('text', '').upper()
        
        if not text:
            return jsonify({'error': 'No text provided', 'success': False}), 400
        
        # Generate sign descriptions
        signs = []
        for char in text:
            if char == ' ':
                signs.append({
                    'character': 'SPACE',
                    'description': 'Make a flat hand and move it forward',
                    'tips': 'Keep palm facing down'
                })
            elif char.isalpha():
                signs.append({
                    'character': char,
                    'description': get_sign_description(char),
                    'tips': get_sign_tips(char)
                })
            else:
                signs.append({
                    'character': char,
                    'description': 'No specific sign available',
                    'tips': 'Spell using fingerspelling'
                })
        
        return jsonify({
            'success': True,
            'original_text': text,
            'translation': signs,
            'word_count': len(text.split()),
            'character_count': len(text)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

def get_sign_description(letter):
    """Get description for ASL letter"""
    descriptions = {
        'A': 'Make a fist with thumb alongside fingers',
        'B': 'Hold all fingers straight up with palm facing out',
        'C': 'Form a C shape with your hand',
        'D': 'Point index finger up, other fingers in fist',
        'E': 'Fold fingers into palm with thumb across',
        'F': 'Touch thumb to index finger, other fingers up',
        'G': 'Point index finger sideways',
        'H': 'Point index and middle fingers sideways',
        'I': 'Point pinky finger up',
        'J': 'Draw a J shape with pinky finger',
        'K': 'Make a V with index and middle, thumb in middle',
        'L': 'Make an L shape with thumb and index',
        'M': 'Tuck thumb under three fingers',
        'N': 'Tuck thumb under index and middle fingers',
        'O': 'Make an O shape with all fingers',
        'P': 'Point index down and make a P shape',
        'Q': 'Point index down and make a Q shape',
        'R': 'Cross index and middle fingers',
        'S': 'Make a fist with thumb across fingers',
        'T': 'Make a fist with thumb between index and middle',
        'U': 'Point index and middle fingers up together',
        'V': 'Make peace sign (index and middle fingers up)',
        'W': 'Point up index, middle, and ring fingers',
        'X': 'Bend index finger, others in fist',
        'Y': 'Point thumb and pinky out',
        'Z': 'Draw a Z in the air with index finger'
    }
    return descriptions.get(letter, 'Use fingerspelling')

def get_sign_tips(letter):
    """Get tips for making the sign"""
    tips = {
        'A': 'Keep thumb visible, not tucked in',
        'B': 'Keep fingers together and straight',
        'C': 'Keep space between thumb and fingers',
        'V': 'Keep other fingers folded down',
        'Y': 'Rock hand back and forth',
        'L': 'Keep thumb perpendicular to index'
    }
    return tips.get(letter, 'Keep hand steady and clear')

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    session.pop('history', None)
    return jsonify({'success': True, 'message': 'History cleared'})

@app.route('/vocabulary', methods=['GET'])
def get_vocabulary():
    """Get available signs vocabulary"""
    return jsonify({
        'success': True,
        'vocabulary': CLASS_NAMES,
        'total_signs': len(CLASS_NAMES),
        'letters': CLASS_NAMES[:26],
        'special': CLASS_NAMES[26:]
    })

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update application settings"""
    try:
        data = request.get_json()
        
        # Update session settings
        if 'auto_clear' in data:
            session['auto_clear'] = data['auto_clear']
        if 'confidence_threshold' in data:
            session['confidence_threshold'] = max(0.1, min(1.0, data['confidence_threshold']))
        if 'language' in data:
            session['language'] = data['language']
        
        return jsonify({
            'success': True,
            'message': 'Settings updated',
            'settings': {
                'auto_clear': session.get('auto_clear', False),
                'confidence_threshold': session.get('confidence_threshold', 0.7),
                'language': session.get('language', 'en')
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# Video streaming
def generate_frames():
    """Generate video frames for streaming"""
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Detect landmarks
        landmarks, annotated_frame = detect_hand_landmarks(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 Starting Sign Language Translator...")
    print("🔧 Initializing model...")
    init_model()
    
    print("🌐 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
