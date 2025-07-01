import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import pickle
import pyaudio
import wave
import threading
import time
from collections import deque
import matplotlib.pyplot as plt

class DarijaVoiceAssistant:
    def __init__(self, model_path='darija_speech_model.h5', 
                 encoder_path='label_encoder.pkl'):
        """
        Real-time Darija voice recognition assistant
        
        Args:
            model_path: Path to trained model
            encoder_path: Path to label encoder
        """
        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create full paths
        self.model_path = os.path.join(self.script_dir, model_path)
        self.encoder_path = os.path.join(self.script_dir, encoder_path)
        
        print(f"🔍 Looking for model at: {self.model_path}")
        print(f"🔍 Looking for encoder at: {self.encoder_path}")
        
        self.model = None
        self.label_encoder = None
        self.feature_dim = 40
        self.max_length = 100
        self.sample_rate = 22050
        self.recording = False
        self.audio_buffer = deque(maxlen=3 * self.sample_rate)  # 3 seconds buffer
        
        # Audio recording parameters
        self.chunk = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = self.sample_rate
        
        # Load model
        self.load_model()
        
        # Intent actions mapping
        self.intent_actions = {
            'open_door': self.open_door,
            'turn_on_light': self.turn_on_light,
            'turn_off_light': self.turn_off_light
        }
        
        # Darija phrases for feedback
        self.darija_responses = {
            'open_door': "Bab mf7oul (Door opened)",
            'turn_on_light': "Dou meshaal (Light turned on)",
            'turn_off_light': "Dou mfi (Light turned off)",
            'unknown': "Ma fhemtsh (I didn't understand)"
        }
    
    def check_files_exist(self):
        """Check if required files exist and list directory contents"""
        print(f"\n📁 Current working directory: {os.getcwd()}")
        print(f"📁 Script directory: {self.script_dir}")
        
        print("\n📋 Files in current directory:")
        for file in os.listdir('.'):
            if file.endswith(('.h5', '.pkl', '.py')):
                print(f"  ✓ {file}")
        
        print(f"\n🔍 Model file exists: {os.path.exists(self.model_path)}")
        print(f"🔍 Encoder file exists: {os.path.exists(self.encoder_path)}")
        
        if not os.path.exists(self.model_path):
            # Try to find the model file in current directory
            for file in os.listdir('.'):
                if file.endswith('.h5') and 'darija' in file.lower():
                    print(f"💡 Found model file: {file}")
                    self.model_path = os.path.abspath(file)
                    break
        
        if not os.path.exists(self.encoder_path):
            # Try to find the encoder file in current directory
            for file in os.listdir('.'):
                if file.endswith('.pkl') and 'label' in file.lower():
                    print(f"💡 Found encoder file: {file}")
                    self.encoder_path = os.path.abspath(file)
                    break
    
    def load_model(self):
        """Load the trained model and label encoder"""
        # First check if files exist
        self.check_files_exist()
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            if not os.path.exists(self.encoder_path):
                raise FileNotFoundError(f"Encoder file not found at {self.encoder_path}")
            
            print(f"📚 Loading model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            
            print(f"📚 Loading encoder from: {self.encoder_path}")
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("✓ Model loaded successfully!")
            print(f"✓ Recognized intents: {list(self.label_encoder.classes_)}")
            
        except FileNotFoundError as e:
            print(f"✗ File not found: {e}")
            print("\n💡 Please make sure you have:")
            print("  1. darija_speech_model.h5 - The trained model file")
            print("  2. label_encoder.pkl - The label encoder file")
            print("\n📝 If you haven't trained the model yet, please run the training script first.")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Please check if the model files are corrupted or compatible.")
    
    def extract_features(self, audio_data):
        """Extract MFCC features from audio data"""
        try:
            # Ensure audio is the right length (3 seconds)
            if len(audio_data) < self.sample_rate * 3:
                audio_data = np.pad(audio_data, (0, self.sample_rate * 3 - len(audio_data)))
            else:
                audio_data = audio_data[:self.sample_rate * 3]
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=self.feature_dim)
            
            # Pad or truncate to fixed length
            if mfccs.shape[1] < self.max_length:
                mfccs = np.pad(mfccs, ((0, 0), (0, self.max_length - mfccs.shape[1])), mode='constant')
            else:
                mfccs = mfccs[:, :self.max_length]
                
            return mfccs.T  # Transpose to (time_steps, features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros((self.max_length, self.feature_dim))
    
    def predict_intent(self, audio_data, confidence_threshold=0.7):
        """
        Predict intent from audio data
        
        Args:
            audio_data: Raw audio data
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            intent: Predicted intent
            confidence: Prediction confidence
        """
        if self.model is None:
            return "unknown", 0.0
        
        # Extract features
        features = self.extract_features(audio_data)
        features = features.reshape(1, self.max_length, self.feature_dim)
        
        # Make prediction
        prediction = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return "unknown", confidence
        
        # Decode prediction
        predicted_intent = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_intent, confidence
    
    def start_listening(self):
        """Start continuous audio recording"""
        if self.model is None:
            print("❌ Cannot start listening: Model not loaded!")
            return
            
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            print("🎤 Listening for Darija commands...")
            print("Say: '7ell l-bab', 'sha3al dou', or 'tfi dou'")
            print("Press Ctrl+C to stop")
            
            while True:
                # Read audio data
                data = self.stream.read(self.chunk)
                audio_array = np.frombuffer(data, dtype=np.float32)
                
                # Add to buffer
                self.audio_buffer.extend(audio_array)
                
                # Process every 0.5 seconds
                if len(self.audio_buffer) >= self.sample_rate * 3:  # 3 seconds of audio
                    self.process_audio_buffer()
                    time.sleep(0.1)  # Small delay to prevent overprocessing
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping voice assistant...")
        except Exception as e:
            print(f"❌ Error during listening: {e}")
        finally:
            self.stop_listening()
    
    def process_audio_buffer(self):
        """Process the audio buffer for intent recognition"""
        if len(self.audio_buffer) < self.sample_rate * 3:
            return
        
        # Convert buffer to numpy array
        audio_data = np.array(self.audio_buffer)
        
        # Check if there's enough audio energy (simple voice activity detection)
        energy = np.sum(audio_data ** 2)
        if energy < 0.01:  # Threshold for silence
            return
        
        # Predict intent
        intent, confidence = self.predict_intent(audio_data)
        
        if intent != "unknown" and confidence > 0.7:
            print(f"\n🎯 Detected: {intent} (confidence: {confidence:.2f})")
            self.execute_action(intent, confidence)
        elif confidence > 0.5:  # Lower confidence but still detected something
            print(f"🤔 Possible: {intent} (confidence: {confidence:.2f}) - Not confident enough")
    
    def execute_action(self, intent, confidence):
        """Execute the action based on recognized intent"""
        print(f"✨ {self.darija_responses.get(intent, 'Unknown command')}")
        
        if intent in self.intent_actions:
            self.intent_actions[intent]()
        else:
            print("❓ Command not implemented yet")
    
    def open_door(self):
        """Action for opening door"""
        print("🚪 Opening door...")
        # Add your door opening logic here
        # Example: send signal to smart door lock
    
    def turn_on_light(self):
        """Action for turning on light"""
        print("💡 Turning on light...")
        # Add your light control logic here
        # Example: send signal to smart light switch
    
    def turn_off_light(self):
        """Action for turning off light"""
        print("🔌 Turning off light...")
        # Add your light control logic here
        # Example: send signal to smart light switch
    
    def stop_listening(self):
        """Stop audio recording"""
        try:
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, 'audio'):
                self.audio.terminate()
        except Exception as e:
            print(f"Error stopping audio: {e}")
    
    def test_with_file(self, audio_file_path):
        """Test the model with a specific audio file"""
        if self.model is None:
            print("❌ Cannot test: Model not loaded!")
            return "error", 0.0
            
        try:
            # Make sure file path is absolute
            if not os.path.isabs(audio_file_path):
                audio_file_path = os.path.abspath(audio_file_path)
            
            if not os.path.exists(audio_file_path):
                print(f"❌ Audio file not found: {audio_file_path}")
                return "error", 0.0
            
            # Load audio file
            audio_data, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            
            # Predict intent
            intent, confidence = self.predict_intent(audio_data)
            
            print(f"\n📁 File: {os.path.basename(audio_file_path)}")
            print(f"🎯 Predicted intent: {intent}")
            print(f"📊 Confidence: {confidence:.3f}")
            print(f"💬 Response: {self.darija_responses.get(intent, 'Unknown')}")
            
            return intent, confidence
            
        except Exception as e:
            print(f"Error testing file {audio_file_path}: {e}")
            return "error", 0.0
    
    def batch_test(self, audio_files):
        """Test multiple audio files"""
        if self.model is None:
            print("❌ Cannot batch test: Model not loaded!")
            return []
            
        results = []
        print("\n🧪 Batch Testing Results:")
        print("-" * 50)
        
        for audio_file in audio_files:
            intent, confidence = self.test_with_file(audio_file)
            results.append({
                'file': audio_file,
                'intent': intent,
                'confidence': confidence
            })
        
        return results
    
    def record_and_predict(self, duration=3):
        """Record audio for specified duration and predict"""
        if self.model is None:
            print("❌ Cannot record: Model not loaded!")
            return "error", 0.0
            
        print(f"🎤 Recording for {duration} seconds...")
        
        try:
            # Initialize recording
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            frames = []
            for i in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
            
            # Predict intent
            intent, confidence = self.predict_intent(audio_data)
            
            print(f"🎯 Predicted: {intent} (confidence: {confidence:.3f})")
            print(f"💬 Response: {self.darija_responses.get(intent, 'Unknown')}")
            
            if intent != "unknown":
                self.execute_action(intent, confidence)
            
            return intent, confidence
            
        except Exception as e:
            print(f"❌ Error during recording: {e}")
            return "error", 0.0

def main():
    """Main function to run the voice assistant"""
    print("🚀 Starting Darija Voice Assistant")
    print("=" * 40)
    
    # Initialize assistant
    assistant = DarijaVoiceAssistant()
    
    if assistant.model is None:
        print("\n❌ Model not loaded. Please check the following:")
        print("1. Make sure 'darija_speech_model.h5' exists in the current directory")
        print("2. Make sure 'label_encoder.pkl' exists in the current directory")
        print("3. If files don't exist, run the training script first")
        print("4. Check if files are corrupted")
        
        # Ask user if they want to continue anyway
        continue_choice = input("\nDo you want to continue anyway? (y/n): ").lower().strip()
        if continue_choice != 'y':
            return
    
    print("\nChoose an option:")
    print("1. Start continuous listening")
    print("2. Record and predict once")
    print("3. Test with audio file")
    print("4. Check file status")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            assistant.start_listening()
            break
        elif choice == '2':
            assistant.record_and_predict(duration=3)
        elif choice == '3':
            file_path = input("Enter audio file path: ").strip()
            assistant.test_with_file(file_path)
        elif choice == '4':
            assistant.check_files_exist()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()