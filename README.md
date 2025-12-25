# Sign Language Recognition and Translation System

A comprehensive AI-powered system for real-time sign language recognition and translation, bridging communication between Deaf and hearing communities.

## 🌟 Features

### 🤖 Core Capabilities
- **Real-time ASL Alphabet Recognition**: Detect 29 ASL signs (A-Z, space, del, nothing)
- **Multiple Detection Methods**:
  - CNN-based deep learning model
  - MediaPipe hand landmark detection
  - Hybrid approach combining both
- **Dual Interface**:
  - Web application with camera streaming
  - REST API for integration
- **Bi-directional Translation**:
  - Sign-to-Text: Camera → Text/Speech
  - Text-to-Sign: Text → Sign instructions

### 🎯 Technical Features
- **Modular Architecture**: Separate modules for training, inference, and API
- **Pre-trained Models**: Includes CNN model for ASL recognition
- **Real-time Processing**: 30+ FPS on modern hardware
- **Cross-platform**: Works on Windows, macOS, Linux
- **No Internet Required**: Works completely offline

### 🎨 User Experience
- **Responsive Web Interface**: Works on desktop and mobile
- **Visual Feedback**: Hand landmarks, confidence scores
- **Prediction History**: Track and review past translations
- **Settings Customization**: Adjust confidence thresholds, languages
- **Export Functionality**: Save translations as text files

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam for real-time detection
- 4GB+ RAM
- 2GB+ free disk space

### Installation

#### Option 1: One-command Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/sign-language-translator.git
cd sign-language-translator

# Run setup script
python setup.py
