import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math

class MediaPipeSignDetector:
    def __init__(self):
        """Initialize MediaPipe-based sign detector"""
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture recognition parameters
        self.gesture_history = deque(maxlen=10)
        self.current_gesture = "No Hand"
        self.confidence = 0.0
        
        # Define gestures based on finger states
        self.gestures = {
            'A': self.detect_A,
            'B': self.detect_B,
            'C': self.detect_C,
            'D': self.detect_D,
            'F': self.detect_F,
            'I': self.detect_I,
            'L': self.detect_L,
            'OK': self.detect_OK,
            'PALM': self.detect_palm,
            'PEACE': self.detect_peace,
            'ROCK': self.detect_rock,
            'THUMBS_UP': self.detect_thumbs_up,
            'THUMBS_DOWN': self.detect_thumbs_down,
            'VICTORY': self.detect_victory,
            'Y': self.detect_Y,
            'ILY': self.detect_ily  # I Love You
        }
        
        # Colors for display
        self.colors = {
            'hand': (0, 255, 0),
            'connection': (255, 0, 0),
            'text': (255, 255, 255),
            'highlight': (0, 255, 255),
            'warning': (0, 165, 255)
        }
    
    def get_finger_states(self, landmarks, hand_type="right"):
        """Get states of all fingers (0: closed, 1: open)"""
        # Finger tip indices
        tip_ids = [4, 8, 12, 16, 20]
        
        # For thumb, use different reference
        finger_states = []
        
        # Thumb (special handling)
        if hand_type == "right":
            thumb_open = landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x
        else:
            thumb_open = landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x
        
        finger_states.append(1 if thumb_open else 0)
        
        # Other four fingers
        for i in range(1, 5):
            finger_open = landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y
            finger_states.append(1 if finger_open else 0)
        
        return finger_states
    
    def get_hand_type(self, landmarks):
        """Determine if hand is left or right"""
        # Check wrist vs pinky position
        if landmarks[0].x < landmarks[17].x:
            return "right"
        else:
            return "left"
    
    # Gesture detection functions
    def detect_A(self, finger_states):
        # A: All fingers closed, thumb alongside
        return all(state == 0 for state in finger_states)
    
    def detect_B(self, finger_states):
        # B: All fingers open, thumb may be open or closed
        return all(state == 1 for state in finger_states[1:])
    
    def detect_C(self, finger_states):
        # C: All fingers slightly curved (partial open)
        open_fingers = sum(finger_states)
        return 2 <= open_fingers <= 4
    
    def detect_D(self, finger_states):
        # D: Only index finger open
        return finger_states == [0, 1, 0, 0, 0]
    
    def detect_F(self, finger_states):
        # F: Thumb and index touching, others open
        return finger_states[0] == 1 and finger_states[1] == 1 and all(s == 0 for s in finger_states[2:])
    
    def detect_I(self, finger_states):
        # I: Only pinky open
        return finger_states == [0, 0, 0, 0, 1]
    
    def detect_L(self, finger_states):
        # L: Thumb and index open forming L shape
        return finger_states[0] == 1 and finger_states[1] == 1 and all(s == 0 for s in finger_states[2:])
    
    def detect_OK(self, landmarks):
        # OK: Thumb and index tip close together
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        return distance < 0.05
    
    def detect_palm(self, finger_states):
        # Palm: All fingers open
        return all(state == 1 for state in finger_states)
    
    def detect_peace(self, finger_states):
        # Peace/V: Index and middle open
        return finger_states == [0, 1, 1, 0, 0]
    
    def detect_rock(self, finger_states):
        # Rock: Index and pinky open
        return finger_states == [0, 1, 0, 0, 1]
    
    def detect_thumbs_up(self, finger_states):
        # Thumbs up: Only thumb open
        return finger_states == [1, 0, 0, 0, 0]
    
    def detect_thumbs_down(self, finger_states):
        # Thumbs down: Only thumb open, different orientation
        # This is simplified - would need orientation check
        return finger_states == [1, 0, 0, 0, 0]
    
    def detect_victory(self, finger_states):
        # Victory: Index and middle open and separated
        return finger_states == [0, 1, 1, 0, 0]
    
    def detect_Y(self, finger_states):
        # Y: Thumb and pinky open
        return finger_states == [1, 0, 0, 0, 1]
    
    def detect_ily(self, finger_states):
        # I Love You: Thumb, index, pinky open
        return finger_states == [1, 1, 0, 0, 1]
    
    def detect_gesture(self, landmarks):
        """Detect gesture from hand landmarks"""
        if landmarks is None:
            return "No Hand", 0.0
        
        # Get hand type and finger states
        hand_type = self.get_hand_type(landmarks)
        finger_states = self.get_finger_states(landmarks, hand_type)
        
        # Try to detect each gesture
        detected_gestures = []
        confidences = []
        
        # Check static gestures
        for gesture_name, detector in self.gestures.items():
            if gesture_name in ['OK']:
                # Special detectors that need landmarks
                if detector(landmarks):
                    detected_gestures.append(gesture_name)
                    confidences.append(0.8)
            else:
                # Finger state detectors
                if detector(finger_states):
                    detected_gestures.append(gesture_name)
                    
                    # Calculate confidence based on how well it matches
                    confidence = 0.7
                    
                    # Increase confidence for clear gestures
                    if gesture_name == 'A' and sum(finger_states) == 0:
                        confidence = 0.9
                    elif gesture_name == 'B' and sum(finger_states) == 5:
                        confidence = 0.9
                    
                    confidences.append(confidence)
        
        if not detected_gestures:
            # Count open fingers for generic classification
            open_fingers = sum(finger_states)
            if open_fingers == 0:
                return "Fist", 0.6
            elif open_fingers == 5:
                return "Open Hand", 0.6
            else:
                return f"{open_fingers} Fingers", 0.5
        
        # Return the gesture with highest confidence
        max_idx = np.argmax(confidences)
        return detected_gestures[max_idx], confidences[max_idx]
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb)
        
        # Draw landmarks and detect gesture
        annotated_frame = frame.copy()
        gesture = "No Hand"
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=self.colors['hand'], thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=self.colors['connection'], thickness=2, circle_radius=2)
                )
                
                # Detect gesture for this hand
                gesture, confidence = self.detect_gesture(hand_landmarks.landmark)
                
                # Add to history
                self.gesture_history.append(gesture)
                
                # Get most common gesture from history
                if len(self.gesture_history) >= 5:
                    from collections import Counter
                    most_common = Counter(self.gesture_history).most_common(1)[0][0]
                    self.current_gesture = most_common
                    self.confidence = confidence
        
        return annotated_frame, self.current_gesture, self.confidence
    
    def draw_info(self, frame, gesture, confidence, fps=0):
        """Draw information on frame"""
        h, w = frame.shape[:2]
        
        # Draw header
        cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
        cv2.putText(frame, "MediaPipe Sign Detector", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Draw gesture info
        cv2.rectangle(frame, (0, h - 100), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['highlight'], 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Draw FPS
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Draw instructions
        cv2.putText(frame, "Press Q to quit", (w - 120, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return frame
    
    def run_realtime(self):
        """Run real-time detection"""
        print("🎥 Starting MediaPipe Sign Detector...")
        print("🤲 Show your hand to the camera")
        print("🎯 Detects: A, B, C, D, F, I, L, OK, Peace, Rock, Thumbs Up, Y, ILY")
        print("⏹️ Press Q to quit")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, gesture, confidence = self.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Draw info
            display_frame = self.draw_info(processed_frame, gesture, confidence, fps)
            
            # Show frame
            cv2.imshow("MediaPipe Sign Detector", display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Application closed")
    
    def detect_from_image(self, image_path):
        """Detect gesture from image file"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image {image_path}")
            return None
        
        # Process image
        processed_image, gesture, confidence = self.process_frame(image)
        
        # Display results
        print(f"📷 Image: {image_path}")
        print(f"🎯 Detected Gesture: {gesture}")
        print(f"📊 Confidence: {confidence:.1%}")
        
        # Show image
        cv2.imshow("Detected Gesture", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return gesture, confidence

def main():
    """Main function"""
    print("=" * 50)
    print("✋ MediaPipe Sign Language Detector")
    print("=" * 50)
    print("\n📋 Features:")
    print("• Real-time hand tracking")
    print("• Gesture recognition without training")
    print("• No model training required")
    print("• Works offline")
    print("\n🎯 Detects common gestures and ASL letters")
    
    # Create detector
    detector = MediaPipeSignDetector()
    
    # Check command line arguments
    import sys
    if len(sys.argv) > 1:
        # Process image file
        image_path = sys.argv[1]
        detector.detect_from_image(image_path)
    else:
        # Run real-time detection
        detector.run_realtime()

if __name__ == "__main__":
    main()
