from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import uuid
from datetime import datetime
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = None
mp_hands = mp.solutions.hands
hands = None

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'space', 'del', 'nothing']

def init_model():
    """Initialize model and MediaPipe"""
    global model, hands
    
    try:
        if os.path.exists('models/asl_model.h5'):
            model = keras.models.load_model('models/asl_model.h5')
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Model not found")
            model = None
        
        # Initialize MediaPipe
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        model = None
        hands = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_data):
    """Preprocess image for prediction"""
    try:
        # Decode base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Resize
        image_np = cv2.resize(image_np, (64, 64))
        
        # Normalize
        image_np = image_np.astype('float32') / 255.0
        
        # Add batch dimension
        image_np = np.expand_dims(image_np, axis=0)
        
        return image_np
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Sign Language Recognition API',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model is not None,
        'endpoints': [
            '/api/health',
            '/api/predict',
            '/api/predict/batch',
            '/api/translate',
            '/api/vocabulary',
            '/api/upload'
        ]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict sign from image"""
    try:
        # Check content type
        if request.content_type.startswith('application/json'):
            data = request.get_json()
            
            if not data or 'image' not in data:
                return jsonify({
                    'error': 'No image data provided',
                    'success': False
                }), 400
            
            image_data = data['image']
            
        elif request.content_type.startswith('multipart/form-data'):
            # File upload
            if 'file' not in request.files:
                return jsonify({
                    'error': 'No file uploaded',
                    'success': False
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected',
                    'success': False
                }), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'error': 'File type not allowed',
                    'success': False
                }), 400
            
            # Save file temporarily
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and encode file
            with open(filepath, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{image_data}"
            
            # Clean up
            os.remove(filepath)
            
        else:
            return jsonify({
                'error': 'Unsupported content type',
                'success': False
            }), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({
                'error': 'Failed to process image',
                'success': False
            }), 400
        
        # Make prediction
        if model:
            predictions = model.predict(processed_image, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_class = CLASS_NAMES[predicted_idx]
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            top_predictions = [
                {
                    'class': CLASS_NAMES[i],
                    'confidence': float(predictions[0][i]),
                    'label': CLASS_NAMES[i]
                }
                for i in top_indices
            ]
            
            response = {
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'all_predictions': predictions[0].tolist(),
                'top_predictions': top_predictions,
                'timestamp': datetime.utcnow().isoformat(),
                'model_used': 'cnn'
            }
            
        else:
            # Fallback to MediaPipe detection
            response = {
                'success': True,
                'prediction': 'A',  # Placeholder
                'confidence': 0.8,
                'note': 'Using placeholder prediction (model not loaded)',
                'timestamp': datetime.utcnow().isoformat(),
                'model_used': 'placeholder'
            }
        
        # Add sequence support if requested
        if request.args.get('sequence', 'false').lower() == 'true':
            sequence_id = request.args.get('sequence_id', str(uuid.uuid4()))
            response['sequence_id'] = sequence_id
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False,
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict multiple images"""
    try:
        data = request.get_json()
        images = data.get('images', [])
        mode = data.get('mode', 'letters')  # letters or words
        
        if not images:
            return jsonify({
                'error': 'No images provided',
                'success': False
            }), 400
        
        predictions = []
        
        for img_data in images:
            processed_image = preprocess_image(img_data)
            
            if processed_image is None:
                predictions.append({
                    'error': 'Failed to process image',
                    'success': False
                })
                continue
            
            if model:
                pred = model.predict(processed_image, verbose=0)
                idx = np.argmax(pred[0])
                confidence = float(pred[0][idx])
                pred_class = CLASS_NAMES[idx]
                
                predictions.append({
                    'success': True,
                    'prediction': pred_class,
                    'confidence': confidence,
                    'top_3': [
                        {'class': CLASS_NAMES[i], 'confidence': float(pred[0][i])}
                        for i in np.argsort(pred[0])[-3:][::-1]
                    ]
                })
            else:
                predictions.append({
                    'success': True,
                    'prediction': 'A',
                    'confidence': 0.8,
                    'note': 'Placeholder prediction'
                })
        
        # Form word from predictions
        if mode == 'letters':
            word = ''.join([p['prediction'] for p in predictions 
                          if p.get('success') and p['prediction'] in CLASS_NAMES[:26]])
        else:
            word = ' '.join([p['prediction'] for p in predictions 
                           if p.get('success') and p['prediction'] not in ['space', 'del', 'nothing']])
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'word': word,
            'total_predictions': len(predictions),
            'mode': mode,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """Translate text to sign language instructions"""
    try:
        data = request.get_json()
        text = data.get('text', '').upper()
        language = data.get('language', 'ASL')  # ASL or ISL
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'success': False
            }), 400
        
        # ASL descriptions
        asl_descriptions = {
            'A': {'description': 'Make a fist with thumb alongside fingers', 'video': 'A.mp4'},
            'B': {'description': 'Hold all fingers straight up with palm facing out', 'video': 'B.mp4'},
            'C': {'description': 'Form a C shape with your hand', 'video': 'C.mp4'},
            'D': {'description': 'Point index finger up, other fingers in fist', 'video': 'D.mp4'},
            'E': {'description': 'Fold fingers into palm with thumb across', 'video': 'E.mp4'},
            'F': {'description': 'Touch thumb to index finger, other fingers up', 'video': 'F.mp4'},
            'G': {'description': 'Point index finger sideways', 'video': 'G.mp4'},
            'H': {'description': 'Point index and middle fingers sideways', 'video': 'H.mp4'},
            'I': {'description': 'Point pinky finger up', 'video': 'I.mp4'},
            'J': {'description': 'Draw a J shape with pinky finger', 'video': 'J.mp4'},
            'K': {'description': 'Make a V with index and middle, thumb in middle', 'video': 'K.mp4'},
            'L': {'description': 'Make an L shape with thumb and index', 'video': 'L.mp4'},
            'M': {'description': 'Tuck thumb under three fingers', 'video': 'M.mp4'},
            'N': {'description': 'Tuck thumb under index and middle fingers', 'video': 'N.mp4'},
            'O': {'description': 'Make an O shape with all fingers', 'video': 'O.mp4'},
            'P': {'description': 'Point index down and make a P shape', 'video': 'P.mp4'},
            'Q': {'description': 'Point index down and make a Q shape', 'video': 'Q.mp4'},
            'R': {'description': 'Cross index and middle fingers', 'video': 'R.mp4'},
            'S': {'description': 'Make a fist with thumb across fingers', 'video': 'S.mp4'},
            'T': {'description': 'Make a fist with thumb between index and middle', 'video': 'T.mp4'},
            'U': {'description': 'Point index and middle fingers up together', 'video': 'U.mp4'},
            'V': {'description': 'Make peace sign (index and middle fingers up)', 'video': 'V.mp4'},
            'W': {'description': 'Point up index, middle, and ring fingers', 'video': 'W.mp4'},
            'X': {'description': 'Bend index finger, others in fist', 'video': 'X.mp4'},
            'Y': {'description': 'Point thumb and pinky out', 'video': 'Y.mp4'},
            'Z': {'description': 'Draw a Z in the air with index finger', 'video': 'Z.mp4'},
            ' ': {'description': 'Space - pause briefly', 'video': 'space.mp4'}
        }
        
        translation = []
        word_list = text.split()
        
        for word in word_list:
            word_signs = []
            for char in word:
                if char in asl_descriptions:
                    word_signs.append({
                        'character': char,
                        'description': asl_descriptions[char]['description'],
                        'video_url': f'/static/signs/{asl_descriptions[char]["video"]}',
                        'tips': 'Keep hand steady and clear'
                    })
                elif char.isdigit():
                    word_signs.append({
                        'character': char,
                        'description': f'Sign for number {char}',
                        'video_url': f'/static/signs/numbers/{char}.mp4',
                        'tips': 'Use number signs'
                    })
            
            translation.append({
                'word': word,
                'spelling': word_signs,
                'has_word_sign': len(word) == 1  # Single letters might have word signs
            })
        
        return jsonify({
            'success': True,
            'original_text': text,
            'translation': translation,
            'language': language,
            'character_count': len(text),
            'word_count': len(word_list),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/vocabulary', methods=['GET'])
def get_vocabulary():
    """Get available signs vocabulary"""
    categories = {
        'alphabet': CLASS_NAMES[:26],
        'special': CLASS_NAMES[26:],
        'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'common_words': ['HELLO', 'THANK YOU', 'YES', 'NO', 'PLEASE', 'SORRY', 'HELP']
    }
    
    return jsonify({
        'success': True,
        'vocabulary': CLASS_NAMES,
        'categories': categories,
        'total_signs': len(CLASS_NAMES),
        'languages_supported': ['ASL', 'BSL', 'ISL'],
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file for processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part', 'success': False}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file', 'success': False}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Return file info
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'size': os.path.getsize(filepath),
                'uploaded_at': datetime.utcnow().isoformat(),
                'download_url': f'/api/download/{filename}'
            })
        
        return jsonify({'error': 'File type not allowed', 'success': False}), 400
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download uploaded file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found', 'success': False}), 404
        
        from flask import send_file
        return send_file(filepath, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get API usage statistics"""
    return jsonify({
        'success': True,
        'statistics': {
            'total_requests': 0,  # Would track in production
            'predictions_made': 0,
            'translations_made': 0,
            'uptime': '0 days',
            'active_sessions': 0
        },
        'timestamp': datetime.utcnow().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'success': False}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed', 'success': False}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'success': False}), 500

if __name__ == '__main__':
    print("🚀 Starting Sign Language Recognition API...")
    print("🔧 Initializing model...")
    init_model()
    
    print("🌐 Starting API server on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)
