# 🗣️ Darija Speech Recognizer & Voice Assistant

A lightweight speech command recognition system for **Moroccan Darija**.

🎯 The goal: recognize short voice commands like "فتح الباب" (open the door), "شعل الضوء" (turn on the light), etc., and test them via a simple command-line assistant.

---

## 🚀 Overview

This project includes:

- 🧠 A trained speech recognition model
- 💬 A CLI assistant to test predictions
- 🎧 Test audio samples
- 📓 A Jupyter notebook for training
- ⚙️ Scripts to run and interact with the model

---

## 🗂️ Project Structure

| File/Folder                   | Description                                                   |
|------------------------------|---------------------------------------------------------------|
| `darija_speech_recognizer.ipynb` | Jupyter notebook used to train the model                    |
| `darija_voice_assistant.py`     | Command-line interface to test voice commands               |
| `model.h5`                      | Trained Keras model file                                     |
| `label_encoder.pkl`             | Label encoder to decode predicted labels                    |
| `audio/`                        | Test audios: `7allbab.wav`, `cha3aldo.wav`, `tfido.wav`     |
| `requirements.txt`             | Python dependencies                                          |
| `.gitignore`                   | Files/folders excluded from Git version control             |



## 🎙️ Dataset

- ✅ **32 recorded audio commands** in Moroccan Darija
- 🔁 **Data Augmentation** using a Python audio library (e.g., `audiomentations`) to simulate:
  - Noisy background
  - Different speech speeds (slow/fast)
  - Variations in volume (quiet/loud)



## 🛠️ Getting Started

### 📥 Clone the Repository


git clone https://github.com/CodeByIman/darija_speech_recognizer.git
cd darija_speech_recognizer
python darija_speech_recognizer.py
python darija_voice_assistant.py
