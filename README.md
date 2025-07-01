# 🗣️ Darija Speech Recognition MVP
<div align="center" style="margin-bottom: 20px;">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/LibROSA-0.9%2B-yellow" alt="LibROSA">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

A **Minimum Viable Product (MVP)** for speech recognition in **Moroccan Darija** (Moroccan Arabic dialect). This is a proof-of-concept project that recognizes three basic voice commands using a small, self-collected dataset.

## 🎯 Project Overview

<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007acc; margin: 15px 0;">
<strong>MVP Scope:</strong> This project currently recognizes 3 basic Darija commands:
<ul>
<li>🚪 "حل الباب" (ḥall bab) - Open the door</li>
<li>💡 "شعل الضوء" (šaʿʿal ḍ-ḍaw) - Turn on the light</li>
<li>📺 "تفي الضو" (tfayeḍ t-talfaza) - Turn on the TV</li>
</ul>
</div>

**Current Status:** This is an early-stage experiment with a limited dataset collected manually. The goal is to demonstrate the feasibility of Darija speech recognition and serve as a foundation for future expansion.

### What's included:
- 🤖 Basic neural network model for 3-command recognition
- 🎤 Simple CLI tool to test voice commands
- 📊 Small training dataset (self-collected)
- 📓 Jupyter notebook showing the training process
- 🔧 Basic prediction and testing scripts

### Limitations:
- **Limited vocabulary:** Only 3 commands currently supported
- **Small dataset:** Training data is limited and self-collected
- **Prototype quality:** This is an MVP/proof-of-concept, not production-ready
- **Accuracy:** Recognition accuracy may vary due to dataset size

---

## 🧠 Model Architecture & Technology Stack

### Deep Learning Models Used

#### 1. **Primary Model: Multi-Layer LSTM Network**
```
Input: MFCC Features (40 features × 100 time steps)
    ↓
LSTM Layer 1: 128 units (return_sequences=True)
    ↓
Dropout: 0.3
    ↓
LSTM Layer 2: 64 units (return_sequences=True)  
    ↓
Dropout: 0.3
    ↓
LSTM Layer 3: 32 units
    ↓
Dropout: 0.3
    ↓
Dense Layer 1: 64 units (ReLU activation)
    ↓
Dropout: 0.5
    ↓
Dense Layer 2: 32 units (ReLU activation)
    ↓
Output Layer: N classes (Softmax activation)
```

**Why LSTM?**
- **Sequential Processing**: Speech is inherently sequential data
- **Long-term Dependencies**: Can remember patterns across time
- **Variable Length Handling**: Adapts to different speech durations
- **Temporal Relationships**: Captures phonetic transitions in Darija

#### 2. **Feature Extraction: MFCC (Mel-Frequency Cepstral Coefficients)**
- **40 MFCC coefficients** per frame
- **22,050 Hz sampling rate**
- **3-second audio duration** (padded/truncated)
- **100 time steps** maximum sequence length

### Technical Specifications

| Component | Details |
|-----------|---------|
| **Audio Processing** | LibROSA library for audio loading and MFCC extraction |
| **Deep Learning Framework** | TensorFlow/Keras 2.10+ |
| **Model Type** | Sequential LSTM with Dense layers |
| **Optimizer** | Adam optimizer |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Regularization** | Dropout layers (0.3-0.5) |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## 🗂️ Project Structure

```
darija_speech_recognizer/
├── 📓 darija_speech_recognizer.ipynb    # Main training notebook
├── 🤖 darija_voice_assistant.py         # CLI assistant for testing
├── 📊 augmented_data.csv                # Dataset metadata
├── 🎵 audio/
│   ├── augmented_dataset/               # Training audio files
│   ├── 7allbab.wav                     # Test sample
│   ├── cha3aldo.wav                    # Test sample
│   └── tfido.wav                       # Test sample
├── 🧠 deployment_model/
│   ├── darija_speech_model.h5          # Trained model
│   ├── label_encoder.pkl               # Label encoder
│   ├── model_config.json               # Model configuration
│   └── requirements.txt                # Dependencies
├── 📈 plots/
│   ├── training_history.png            # Training metrics
│   └── confusion_matrix.png            # Model performance
├── 📋 requirements.txt                  # Project dependencies
└── 📖 README.md                        # This file
```

---

## 🎙️ Dataset & Data Processing

### Dataset Composition
- **Training Dataset**: Audio files stored in `audio/augmented_dataset/`
- **Metadata**: Stored in `augmented_data.csv` with intent labels and Darija phrases
- **Test Samples**: 3 example audio files:
  - `7allbab.wav` - Test audio sample
  - `cha3aldo.wav` - Test audio sample  
  - `tfido.wav` - Test audio sample
- **Audio Format**: WAV files, 22.05 kHz, mono channel
- **Duration**: 3 seconds (standardized)

### Data Augmentation Techniques
The project uses advanced audio augmentation to improve model robustness:

```python
# Augmentation techniques applied:
- Background noise injection
- Speed variation (0.8x - 1.2x)
- Volume adjustment (0.5x - 1.5x)  
- Pitch shifting (±2 semitones)
- Time stretching
- Echo/reverb effects
```

### Feature Engineering Pipeline
1. **Audio Loading**: LibROSA loads WAV files at 22.05 kHz
2. **Duration Normalization**: Pad/truncate to 3 seconds
3. **MFCC Extraction**: 40 coefficients per time frame
4. **Sequence Padding**: Standardize to 100 time steps
5. **Feature Normalization**: Scale features for neural network input

---

## 🏗️ Model Training Process

### Training Configuration
```python
# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = Adam default (0.001)
EARLY_STOPPING_PATIENCE = 10
LR_REDUCTION_PATIENCE = 5
```

### Training Pipeline
1. **Data Loading**: Load CSV metadata and audio files
2. **Feature Extraction**: Convert audio to MFCC features
3. **Data Splitting**: 80% training, 20% testing (stratified)
4. **Model Compilation**: Configure LSTM architecture
5. **Training**: Fit model with callbacks for optimization
6. **Evaluation**: Generate metrics and visualizations
7. **Model Saving**: Export for deployment

### Performance Monitoring
- **Training/Validation Accuracy** tracking
- **Loss curves** visualization  
- **Confusion Matrix** for class-wise performance
- **Classification Report** with precision/recall/F1
- **Early Stopping** to prevent overfitting

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.10+
LibROSA 0.9+
NumPy, Pandas, Scikit-learn
Matplotlib, Seaborn (for visualization)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CodeByIman/darija_speech_recognizer.git
cd darija_speech_recognizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model (optional)**
```bash
jupyter notebook darija_speech_recognizer.ipynb
```

4. **Test the assistant**
```bash
python darija_voice_assistant.py
```

---

## 🧪 Usage Examples

### Command Line Testing
```bash
# Test with the provided sample audio files
python darija_voice_assistant.py

# The assistant will:
# 1. Load the trained model
# 2. Test the available audio samples (7allbab.wav, cha3aldo.wav, tfido.wav)
# 3. Display predictions with confidence scores for each sample
```

## 📊 Model Performance

### Expected Performance Metrics
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Inference Time**: <1 second per audio file
- **Model Size**: ~2-5 MB

**Test Audio Files Available:**
- `7allbab.wav` - Test sample
- `cha3aldo.wav` - Test sample
- `tfido.wav` - Test sample


## 🔧 Advanced Features

### Model Customization
- **Hyperparameter Tuning**: Adjust LSTM units, dropout rates
- **Architecture Modifications**: Add/remove layers, change activation functions
- **Feature Engineering**: Experiment with different audio features (spectrograms, chromagrams)
- **Data Augmentation**: Custom augmentation strategies for Darija

### Performance Optimization
- **Model Quantization**: Reduce model size for mobile deployment
- **TensorRT Integration**: GPU acceleration for inference
- **Batch Prediction**: Process multiple audio files efficiently
- **Caching**: Cache extracted features for faster repeated predictions

---

## 📈 Future Enhancements

### Technical Improvements
- [ ] **Transformer Architecture**: Experiment with attention mechanisms
- [ ] **Real-time Processing**: Stream audio processing capabilities
- [ ] **Multi-language Support**: Extend to other Arabic dialects
- [ ] **Voice Activity Detection**: Automatic speech segment detection
- [ ] **Speaker Adaptation**: Personalized models for different speakers

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Areas
- **Dataset Expansion**: Add more Darija voice samples
- **Model Improvements**: Experiment with new architectures
- **Web Interface**: Enhance the user interface
- **Documentation**: Improve code documentation
- **Testing**: Add unit tests and integration tests


---

## 🙏 Acknowledgments

- **TensorFlow/Keras** team for the deep learning framework
- **LibROSA** developers for audio processing tools
- **Streamlit** for the web app framework
- **Moroccan Arabic** linguistics community
- **Open source contributors** who make projects like this possible

---

## 📞 Contact & Support

- **Author**: Iman
- **GitHub**: [@CodeByIman](https://github.com/CodeByIman)
- **Issues**: [GitHub Issues](https://github.com/CodeByIman/darija_speech_recognizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CodeByIman/darija_speech_recognizer/discussions)

---

### Key Technologies
- **TensorFlow**: https://tensorflow.org
- **LibROSA**: https://librosa.org
- **Streamlit**: https://streamlit.io
- **Scikit-learn**: https://scikit-learn.org

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**🔄 Fork it to contribute to Darija NLP research!**

</div>
