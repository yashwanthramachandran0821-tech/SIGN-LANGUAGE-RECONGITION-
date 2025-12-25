class SignLanguageTranslator {
    constructor() {
        // DOM Elements
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Buttons
        this.startBtn = document.getElementById('startBtn');
        this.captureBtn = document.getElementById('captureBtn');
        this.continuousBtn = document.getElementById('continuousBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.translateBtn = document.getElementById('translateBtn');
        this.speakBtn = document.getElementById('speakBtn');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        this.saveBtn = document.getElementById('saveBtn');
        
        // Display Elements
        this.predictionText = document.getElementById('predictionText');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.confidenceBar = document.getElementById('confidenceBar');
        this.topPredictions = document.getElementById('topPredictions');
        this.textInput = document.getElementById('textInput');
        this.signOutput = document.getElementById('signOutput');
        this.historyList = document.getElementById('historyList');
        this.videoStatus = document.getElementById('videoStatus');
        
        // Status Elements
        this.modelStatus = document.getElementById('modelStatus');
        this.cameraStatus = document.getElementById('cameraStatus');
        this.recordingStatus = document.getElementById('recordingStatus');
        
        // Settings
        this.confidenceThreshold = document.getElementById('confidenceThreshold');
        this.thresholdValue = document.getElementById('thresholdValue');
        this.autoClear = document.getElementById('autoClear');
        this.languageSelect = document.getElementById('languageSelect');
        
        // State Variables
        this.isStreaming = false;
        this.continuousMode = false;
        this.isRecording = false;
        this.intervalId = null;
        this.currentText = '';
        this.predictionHistory = [];
        this.stream = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        
        // Initialize
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing Sign Language Translator...');
        
        this.setupEventListeners();
        await this.setupWebcam();
        this.updateStatus();
        this.loadSettings();
        this.loadHistory();
        
        console.log('✅ Initialization complete');
    }
    
    setupEventListeners() {
        // Camera Controls
        this.startBtn.addEventListener('click', () => this.toggleWebcam());
        this.captureBtn.addEventListener('click', () => this.captureAndPredict());
        this.continuousBtn.addEventListener('click', () => this.toggleContinuousMode());
        this.clearBtn.addEventListener('click', () => this.clearText());
        this.saveBtn.addEventListener('click', () => this.saveTranslation());
        
        // Translation Controls
        this.translateBtn.addEventListener('click', () => this.translateTextToSign());
        this.speakBtn.addEventListener('click', () => this.speakText());
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        
        // Settings
        this.confidenceThreshold.addEventListener('input', (e) => {
            this.thresholdValue.textContent = `${e.target.value}%`;
            this.saveSettings();
        });
        
        this.autoClear.addEventListener('change', () => this.saveSettings());
        this.languageSelect.addEventListener('change', () => this.saveSettings());
        
        // Text input auto-translate
        this.textInput.addEventListener('input', (e) => {
            if (e.target.value.length > 0) {
                // Auto-translate after delay
                clearTimeout(this.translateTimeout);
                this.translateTimeout = setTimeout(() => {
                    if (e.target.value.length > 0) {
                        this.translateTextToSign();
                    }
                }, 1000);
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'r':
                        e.preventDefault();
                        this.toggleWebcam();
                        break;
                    case 'c':
                        e.preventDefault();
                        this.clearText();
                        break;
                    case 's':
                        e.preventDefault();
                        this.saveTranslation();
                        break;
                    case 't':
                        e.preventDefault();
                        this.translateTextToSign();
                        break;
                }
            }
            
            // Space bar to capture
            if (e.code === 'Space' && this.isStreaming) {
                e.preventDefault();
                this.captureAndPredict();
            }
        });
    }
    
    async setupWebcam() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user',
                    frameRate: { ideal: 30 }
                },
                audio: false
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            this.video.onloadedmetadata = () => {
                this.video.play();
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.cameraStatus.innerHTML = '<i class="fas fa-check-circle"></i> Camera Ready';
            };
            
            this.video.onerror = (error) => {
                console.error('Video error:', error);
                this.cameraStatus.innerHTML = '<i class="fas fa-times-circle"></i> Camera Error';
            };
            
        } catch (error) {
            console.error('Error accessing webcam:', error);
            this.showError('Could not access webcam. Please check permissions.');
            this.cameraStatus.innerHTML = '<i class="fas fa-times-circle"></i> No Camera';
        }
    }
    
    toggleWebcam() {
        if (this.isStreaming) {
            this.stopWebcam();
        } else {
            this.startWebcam();
        }
    }
    
    startWebcam() {
        this.isStreaming = true;
        this.startBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
        this.startBtn.classList.remove('btn-primary');
        this.startBtn.classList.add('btn-danger');
        this.captureBtn.disabled = false;
        this.videoStatus.textContent = 'LIVE';
        this.videoStatus.classList.add('recording');
        this.updateStatus('Camera active - Show your hand sign');
    }
    
    stopWebcam() {
        this.isStreaming = false;
        this.startBtn.innerHTML = '<i class="fas fa-play"></i> Start Camera';
        this.startBtn.classList.remove('btn-danger');
        this.startBtn.classList.add('btn-primary');
        this.captureBtn.disabled = true;
        this.stopContinuousMode();
        this.videoStatus.textContent = 'OFF';
        this.videoStatus.classList.remove('recording');
        this.updateStatus('Camera stopped');
    }
    
    async captureAndPredict() {
        if (!this.isStreaming) {
            this.showError('Please start the camera first');
            return;
        }
        
        try {
            // Draw current frame to canvas
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Get image data
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            // Show loading state
            this.showLoading();
            
            // Send to server for prediction
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `image_data=${encodeURIComponent(imageData)}`
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.handlePrediction(result);
            } else {
                this.showError('Prediction failed: ' + result.error);
            }
            
        } catch (error) {
            console.error('Error during prediction:', error);
            this.showError('Failed to communicate with server');
        } finally {
            this.hideLoading();
        }
    }
    
    handlePrediction(result) {
        const confidence = result.confidence;
        const threshold = parseInt(this.confidenceThreshold.value) / 100;
        
        // Update confidence display
        this.confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
        this.confidenceBar.style.width = `${confidence * 100}%`;
        
        // Update confidence bar color
        if (confidence > 0.9) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #38b000, #2d6a4f)';
        } else if (confidence > 0.7) {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #ff9e00, #ff6d00)';
        } else {
            this.confidenceBar.style.background = 'linear-gradient(90deg, #ff0054, #d00000)';
        }
        
        // Only add to text if confidence is above threshold
        if (confidence >= threshold) {
            const prediction = result.prediction;
            
            if (prediction === 'space') {
                this.currentText += ' ';
            } else if (prediction === 'del') {
                this.currentText = this.currentText.slice(0, -1);
            } else if (prediction !== 'nothing') {
                this.currentText += prediction;
            }
            
            // Update display
            this.predictionText.textContent = this.currentText;
            
            // Add to history
            this.addToHistory(prediction, confidence);
            
            // Show prediction in UI
            this.showPrediction(prediction, confidence);
        }
        
        // Update top predictions
        if (result.top_predictions) {
            this.updateTopPredictions(result.top_predictions);
        }
        
        // Auto-clear if enabled
        if (this.autoClear.checked && this.currentText.length > 100) {
            setTimeout(() => this.clearText(), 3000);
        }
    }
    
    showPrediction(prediction, confidence) {
        // Create prediction chip
        const chip = document.createElement('div');
        chip.className = 'prediction-chip';
        chip.innerHTML = `
            <strong>${prediction}</strong>
            <span class="chip-confidence">${(confidence * 100).toFixed(0)}%</span>
        `;
        
        // Add to top predictions area
        this.topPredictions.insertBefore(chip, this.topPredictions.firstChild);
        
        // Limit to 5 chips
        if (this.topPredictions.children.length > 5) {
            this.topPredictions.removeChild(this.topPredictions.lastChild);
        }
        
        // Add animation
        chip.style.animation = 'fadeIn 0.3s ease';
        setTimeout(() => chip.style.animation = '', 300);
    }
    
    updateTopPredictions(predictions) {
        this.topPredictions.innerHTML = '';
        
        predictions.forEach(pred => {
            const chip = document.createElement('div');
            chip.className = 'prediction-chip';
            chip.innerHTML = `
                <strong>${pred.class}</strong>
                <span class="chip-confidence">${(pred.confidence * 100).toFixed(0)}%</span>
            `;
            this.topPredictions.appendChild(chip);
        });
    }
    
    toggleContinuousMode() {
        if (this.continuousMode) {
            this.stopContinuousMode();
        } else {
            this.startContinuousMode();
        }
    }
    
    startContinuousMode() {
        if (!this.isStreaming) {
            this.showError('Please start the camera first');
            return;
        }
        
        this.continuousMode = true;
        this.continuousBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Continuous';
        this.continuousBtn.classList.remove('btn-warning');
        this.continuousBtn.classList.add('btn-danger');
        this.isRecording = true;
        this.recordingStatus.innerHTML = '<i class="fas fa-circle"></i> Recording';
        
        // Capture every 500ms
        this.intervalId = setInterval(() => {
            this.captureAndPredict();
        }, 500);
        
        this.updateStatus('Continuous mode active');
    }
    
    stopContinuousMode() {
        this.continuousMode = false;
        this.continuousBtn.innerHTML = '<i class="fas fa-sync"></i> Continuous Mode';
        this.continuousBtn.classList.remove('btn-danger');
        this.continuousBtn.classList.add('btn-warning');
        this.isRecording = false;
        this.recordingStatus.innerHTML = '<i class="fas fa-circle"></i> Paused';
        
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
    
    clearText() {
        this.currentText = '';
        this.predictionText.textContent = 'Your translated text will appear here...';
        this.confidenceValue.textContent = '0%';
        this.confidenceBar.style.width = '0%';
        this.topPredictions.innerHTML = '';
        this.updateStatus('Text cleared');
        
        // Add to history
        this.addToHistory('CLEAR', 1.0);
    }
    
    async translateTextToSign() {
        const text = this.textInput.value.trim();
        
        if (!text) {
            this.showError('Please enter some text to translate');
            return;
        }
        
        try {
            this.showLoading('Translating...');
            
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: text,
                    language: this.languageSelect.value 
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displaySignInstructions(result.translation);
            } else {
                this.showError('Translation failed: ' + result.error);
            }
            
        } catch (error) {
            console.error('Error translating text:', error);
            this.showError('Failed to translate text');
        } finally {
            this.hideLoading();
        }
    }
    
    displaySignInstructions(translation) {
        let html = '<h3><i class="fas fa-hands"></i> Sign Instructions:</h3>';
        
        if (Array.isArray(translation)) {
            translation.forEach((wordSigns, wordIndex) => {
                html += `<div class="word-section">`;
                html += `<h4>Word: <strong>${wordSigns.word}</strong></h4>`;
                
                if (wordSigns.spelling) {
                    html += `<div class="spelling-grid">`;
                    wordSigns.spelling.forEach(sign => {
                        html += `
                            <div class="sign-instruction">
                                <div class="sign-char">${sign.character}</div>
                                <div class="sign-desc">${sign.description}</div>
                                ${sign.tips ? `<div class="sign-tips"><small>💡 ${sign.tips}</small></div>` : ''}
                            </div>
                        `;
                    });
                    html += `</div>`;
                }
                
                html += `</div>`;
            });
        }
        
        this.signOutput.innerHTML = html;
        this.updateStatus('Text translated to sign instructions');
    }
    
    async speakText() {
        const text = this.textInput.value.trim() || this.currentText;
        
        if (!text) {
            this.showError('No text to speak');
            return;
        }
        
        try {
            // Try to use server-side TTS first
            const response = await fetch('/text_to_speech', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateStatus('Text sent for speech synthesis');
            }
            
        } catch (error) {
            console.log('Server TTS failed, using browser fallback');
        }
        
        // Fallback to Web Speech API
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Configure voice
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            
            // Get available voices
            const voices = speechSynthesis.getVoices();
            if (voices.length > 0) {
                // Prefer natural-sounding voices
                const preferredVoice = voices.find(voice => 
                    voice.lang.startsWith('en') && voice.name.includes('Natural')
                ) || voices[0];
                utterance.voice = preferredVoice;
            }
            
            speechSynthesis.speak(utterance);
            this.updateStatus('Speaking text...');
        } else {
            this.showError('Text-to-speech not supported in your browser');
        }
    }
    
    addToHistory(prediction, confidence) {
        const historyItem = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            prediction: prediction,
            confidence: confidence,
            text: this.currentText
        };
        
        this.predictionHistory.unshift(historyItem);
        
        // Keep only last 50 items
        if (this.predictionHistory.length > 50) {
            this.predictionHistory = this.predictionHistory.slice(0, 50);
        }
        
        this.updateHistoryDisplay();
        this.saveHistory();
    }
    
    updateHistoryDisplay() {
        this.historyList.innerHTML = '';
        
        this.predictionHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <div class="history-content">
                    <div class="history-prediction">${item.prediction}</div>
                    <div class="history-time">${item.timestamp}</div>
                </div>
                <div class="history-confidence">${(item.confidence * 100).toFixed(0)}%</div>
            `;
            
            // Click to restore text
            historyItem.addEventListener('click', () => {
                this.currentText = item.text;
                this.predictionText.textContent = this.currentText;
                this.updateStatus('Text restored from history');
            });
            
            this.historyList.appendChild(historyItem);
        });
    }
    
    clearHistory() {
        if (confirm('Clear all prediction history?')) {
            this.predictionHistory = [];
            this.updateHistoryDisplay();
            localStorage.removeItem('signLanguageHistory');
            this.updateStatus('History cleared');
        }
    }
    
    saveTranslation() {
        if (!this.currentText.trim()) {
            this.showError('No text to save');
            return;
        }
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `translation-${timestamp}.txt`;
        const content = `Sign Language Translation\n` +
                       `Date: ${new Date().toLocaleString()}\n` +
                       `Text: ${this.currentText}\n\n`;
        
        // Create download link
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.updateStatus('Translation saved');
    }
    
    loadSettings() {
        const settings = JSON.parse(localStorage.getItem('signLanguageSettings') || '{}');
        
        if (settings.confidenceThreshold) {
            this.confidenceThreshold.value = settings.confidenceThreshold;
            this.thresholdValue.textContent = `${settings.confidenceThreshold}%`;
        }
        
        if (settings.autoClear !== undefined) {
            this.autoClear.checked = settings.autoClear;
        }
        
        if (settings.language) {
            this.languageSelect.value = settings.language;
        }
    }
    
    saveSettings() {
        const settings = {
            confidenceThreshold: this.confidenceThreshold.value,
            autoClear: this.autoClear.checked,
            language: this.languageSelect.value
        };
        
        localStorage.setItem('signLanguageSettings', JSON.stringify(settings));
    }
    
    loadHistory() {
        const savedHistory = localStorage.getItem('signLanguageHistory');
        if (savedHistory) {
            this.predictionHistory = JSON.parse(savedHistory);
            this.updateHistoryDisplay();
        }
    }
    
    saveHistory() {
        localStorage.setItem('signLanguageHistory', 
            JSON.stringify(this.predictionHistory.slice(0, 100)));
    }
    
    updateStatus(message) {
        if (message) {
            console.log('Status:', message);
            
            // Show temporary status message
            const statusDiv = document.getElementById('statusMessage');
            if (statusDiv) {
                statusDiv.textContent = message;
                statusDiv.style.display = 'block';
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 3000);
            }
        }
    }
    
    showError(message) {
        console.error('Error:', message);
        
        // Show error in UI
        const errorDiv = document.getElementById('errorMessage') || this.createMessageDiv('errorMessage');
        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
        errorDiv.style.display = 'block';
        
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
    
    showLoading(message = 'Processing...') {
        const loadingDiv = document.getElementById('loadingMessage') || this.createMessageDiv('loadingMessage');
        loadingDiv.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${message}`;
        loadingDiv.style.display = 'block';
    }
    
    hideLoading() {
        const loadingDiv = document.getElementById('loadingMessage');
        if (loadingDiv) {
            loadingDiv.style.display = 'none';
        }
    }
    
    createMessageDiv(id) {
        const div = document.createElement('div');
        div.id = id;
        div.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            z-index: 1000;
            display: none;
            animation: slideIn 0.3s ease;
        `;
        
        if (id === 'errorMessage') {
            div.style.background = 'linear-gradient(45deg, #ff0054, #d00000)';
            div.style.color = 'white';
        } else if (id === 'loadingMessage') {
            div.style.background = 'linear-gradient(45deg, #00b4d8, #0077b6)';
            div.style.color = 'white';
        } else {
            div.style.background = 'linear-gradient(45deg, #38b000, #2d6a4f)';
            div.style.color = 'white';
        }
        
        document.body.appendChild(div);
        return div;
    }
    
    // Health check
    async checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.modelStatus.innerHTML = data.model_loaded ? 
                    '<i class="fas fa-check-circle"></i> Model Loaded' :
                    '<i class="fas fa-exclamation-triangle"></i> Model Not Loaded';
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.modelStatus.innerHTML = '<i class="fas fa-times-circle"></i> Server Error';
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    const translator = new SignLanguageTranslator();
    window.signLanguageTranslator = translator; // Make available globally
    
    // Check server health every 30 seconds
    setInterval(() => translator.checkHealth(), 30000);
    translator.checkHealth();
    
    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        
        .prediction-chip {
            animation: fadeIn 0.3s ease;
        }
    `;
    document.head.appendChild(style);
});
