import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle

class ASLModelTrainer:
    def __init__(self, data_path=None):
        """Initialize ASL Model Trainer"""
        self.data_path = data_path or 'data'
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                           'space', 'del', 'nothing']
        self.model = None
        self.history = None
        self.img_size = (64, 64)
        
    def create_synthetic_data(self, num_samples=100):
        """Create synthetic data if no real dataset is available"""
        print("🔄 Creating synthetic training data...")
        
        # Define hand landmarks for each letter (simplified)
        letter_templates = self._get_letter_templates()
        
        X_train = []
        y_train = []
        
        for class_idx, letter in enumerate(self.class_names):
            if letter in letter_templates:
                for i in range(num_samples):
                    # Create synthetic image
                    img = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                    # Draw hand shape based on template
                    template = letter_templates[letter]
                    self._draw_hand_shape(img, template)
                    
                    # Add noise and variations
                    img = self._add_variations(img)
                    
                    X_train.append(img)
                    y_train.append(class_idx)
        
        return np.array(X_train), np.array(y_train)
    
    def _get_letter_templates(self):
        """Define simplified hand shapes for each letter"""
        return {
            'A': {'fingers': [0, 0, 0, 0, 0], 'palm_pos': (32, 32)},  # Fist
            'B': {'fingers': [1, 1, 1, 1, 1], 'palm_pos': (32, 32)},  # All fingers up
            'C': {'fingers': [1, 1, 1, 1, 0], 'palm_pos': (32, 32)},  # C shape
            'D': {'fingers': [0, 1, 0, 0, 0], 'palm_pos': (32, 32)},  # Index finger
            'E': {'fingers': [0, 0, 0, 0, 1], 'palm_pos': (32, 32)},  # Pinky
            'I': {'fingers': [0, 0, 0, 0, 1], 'palm_pos': (32, 20)},  # Pinky up
            'L': {'fingers': [1, 1, 0, 0, 0], 'palm_pos': (32, 32)},  # L shape
            'V': {'fingers': [0, 1, 1, 0, 0], 'palm_pos': (32, 32)},  # Peace sign
            'Y': {'fingers': [1, 0, 0, 0, 1], 'palm_pos': (32, 32)},  # Y shape
            'space': {'fingers': [0, 0, 0, 0, 0], 'palm_pos': (32, 32)},
            'nothing': {'fingers': [0, 0, 0, 0, 0], 'palm_pos': None}
        }
    
    def _draw_hand_shape(self, img, template):
        """Draw hand shape on image"""
        if template['palm_pos']:
            palm_x, palm_y = template['palm_pos']
            
            # Draw palm
            cv2.circle(img, (palm_x, palm_y), 10, (255, 255, 255), -1)
            
            # Draw fingers
            finger_positions = [
                (palm_x - 15, palm_y - 30),  # Thumb
                (palm_x, palm_y - 40),       # Index
                (palm_x + 10, palm_y - 35),  # Middle
                (palm_x + 20, palm_y - 30),  # Ring
                (palm_x + 30, palm_y - 25)   # Pinky
            ]
            
            for i, (finger_up, pos) in enumerate(zip(template['fingers'], finger_positions)):
                if finger_up:
                    cv2.line(img, (palm_x, palm_y), pos, (255, 255, 255), 3)
                    cv2.circle(img, pos, 5, (255, 255, 255), -1)
    
    def _add_variations(self, img):
        """Add variations to synthetic image"""
        # Random brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = hsv[:,:,2] * np.random.uniform(0.8, 1.2)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((32, 32), angle, 1)
        img = cv2.warpAffine(img, M, (64, 64))
        
        # Random noise
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def create_enhanced_model(self):
        """Create enhanced CNN model for ASL recognition"""
        model = models.Sequential([
            # First convolution block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolution block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolution block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        model.summary()
        return model
    
    def train(self, epochs=30, batch_size=32):
        """Train the ASL recognition model"""
        print("📊 Starting model training...")
        
        # Load or create data
        if os.path.exists(os.path.join(self.data_path, 'train')):
            print("📁 Loading dataset from files...")
            X_train, y_train, X_val, y_val = self._load_real_data()
        else:
            print("🔧 Generating synthetic data...")
            X_train, y_train = self.create_synthetic_data(num_samples=200)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        print(f"📈 Training samples: {len(X_train)}")
        print(f"📈 Validation samples: {len(X_val)}")
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest'
        )
        
        # Create model
        self.model = self.create_enhanced_model()
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('models/asl_model.h5')
        print("✅ Model saved as models/asl_model.h5")
        
        # Evaluate model
        self.evaluate_model(X_val, y_val)
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def _load_real_data(self):
        """Load real dataset from directory structure"""
        import glob
        
        X_train = []
        y_train = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, 'train', class_name, '*.jpg')
            images = glob.glob(class_path)
            
            for img_path in images[:100]:  # Limit to 100 per class
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, self.img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X_train.append(img)
                    y_train.append(class_idx)
                except:
                    continue
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Normalize
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        
        return X_train, y_train, X_val, y_val
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 Evaluating model...")
        
        # Make predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        
        # Save evaluation metrics
        metrics = {
            'accuracy': accuracy,
            'class_names': self.class_names,
            'confusion_matrix': cm.tolist()
        }
        
        with open('models/evaluation_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    print("=" * 50)
    print("🤖 ASL Sign Language Model Trainer")
    print("=" * 50)
    
    # Create trainer
    trainer = ASLModelTrainer()
    
    # Train model
    print("\n🚀 Starting training process...")
    trainer.train(epochs=20)
    
    print("\n🎉 Training completed successfully!")
    print("📁 Model saved: models/asl_model.h5")
    print("📊 Plots saved: training_history.png, confusion_matrix.png")

if __name__ == "__main__":
    main()
