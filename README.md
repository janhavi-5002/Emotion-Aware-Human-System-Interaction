# 🤖 Emotion-Aware Human-System Interaction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### 🏆 HCL IIT Hackathon 2026 — Problem Statement 9
**Real-time Multimodal Emotion Detection with Intelligent Soft Bot**

</div>

---

## 📌 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Emotion-Bot Behavior Mapping](#emotion-bot-behavior-mapping)
- [Project Structure](#project-structure)
- [Evaluation Criteria](#evaluation-criteria)
- [Team](#team)

---

## 🎯 Overview

This project implements a **real-time emotion-aware human-system interaction** platform that:

- Detects human emotional states from **facial expressions** using Deep Learning
- Analyzes **voice/audio cues** for multimodal emotion recognition
- Controls an **animated soft bot** that adapts its behavior based on detected emotions
- Logs and visualizes **emotion timelines** during interaction
- Provides a **live web dashboard** for real-time monitoring

> Built as part of **HCL IIT Hackathon 2026** under Problem Statement 9: Emotion-Aware Human-System Interaction

---

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────┐
│                    INPUT LAYER                       │
│         📷 Camera Input    🎤 Microphone Input       │
└──────────────┬──────────────────────┬───────────────┘
↓                      ↓
┌──────────────────────┐  ┌──────────────────────────┐
│  FACE DETECTION      │  │  VOICE DETECTION          │
│  DeepFace + OpenCV   │  │  Librosa + SoundDevice    │
│  Custom CNN (FER2013)│  │  Audio Feature Extraction │
└──────────┬───────────┘  └────────────┬─────────────┘
↓                           ↓
┌─────────────────────────────────────────────────────┐
│              EMOTION CLASSIFIER                      │
│     happy / sad / angry / neutral /                  │
│     surprise / fear / disgust                        │
│         + Multimodal Fusion Layer                    │
└─────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│                BEHAVIOR MAPPER                       │
│     Maps each emotion → unique bot behavior          │
└─────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│              SOFT BOT SIMULATOR                      │
│     Animated Web Bot (Flask + HTML5 Canvas)          │
│     Real-time behavior adaptation                    │
└─────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────┐
│         EMOTION TIMELINE LOGGER + VISUALIZER         │
│     CSV Logging + Matplotlib Dashboard               │
└─────────────────────────────────────────────────────┘


---

## ✨ Features

### Core Features
- 🎥 **Real-time Face Emotion Detection** — 30 FPS webcam processing with DeepFace
- 🧠 **Custom CNN Model** — Trained on FER-2013 dataset (28,000+ images)
- 🎤 **Voice Emotion Detection** — Real-time audio analysis using Librosa
- 🔀 **Multimodal Fusion** — Combines face + voice for accurate emotion prediction
- 🤖 **Animated Soft Bot** — Web-based humanoid bot with 7 unique behaviors
- 📊 **Live Dashboard** — Real-time emotion confidence bars and statistics
- 📈 **Emotion Timeline** — Visual graph of emotions over session
- 💾 **CSV Logging** — Every emotion logged with timestamp and confidence

### Technical Features
- ⚡ **Threaded Architecture** — Smooth performance with background processing
- 🔄 **Emotion Smoothing** — Anti-flicker using sliding window average
- 📦 **Face Bounding Box** — Visual face tracking with emotion label
- 🎯 **7 Emotion Classes** — happy, sad, angry, neutral, surprise, fear, disgust
- 📱 **Web Interface** — Accessible from any browser on local network

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Face Detection | DeepFace + OpenCV | Real-time facial emotion detection |
| Custom Model | TensorFlow/Keras CNN | FER-2013 trained emotion classifier |
| Voice Detection | Librosa + SoundDevice | Audio emotion analysis |
| Web Bot UI | Flask + HTML5 Canvas | Animated soft bot interface |
| Live Dashboard | Matplotlib | Real-time emotion visualization |
| Data Logging | Python CSV | Emotion timeline logging |
| Performance | Python Threading | Smooth real-time processing |
| Version Control | Git + GitHub | Team collaboration |

---

## 📊 Dataset

### FER-2013 (Facial Expression Recognition)
- **Source:** [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Size:** 35,887 grayscale images (48×48 pixels)
- **Classes:** 7 emotions
- **Split:** 28,709 training / 7,178 testing

### RAVDESS (Audio Emotions)
- **Source:** [Zenodo RAVDESS](https://zenodo.org/record/1188976)
- **Type:** Speech and song emotional audio recordings
- **Used for:** Voice emotion feature calibration

### Distribution
| Emotion | Train Images | Test Images |
|---------|-------------|-------------|
| Happy | 7,215 | 1,774 |
| Neutral | 4,965 | 1,233 |
| Sad | 4,830 | 1,247 |
| Angry | 3,995 | 958 |
| Surprise | 3,171 | 831 |
| Fear | 4,097 | 1,024 |
| Disgust | 436 | 111 |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- MacOS / Linux / Windows
- Webcam + Microphone
- 4GB RAM minimum

### Step 1 — Clone repository
```bash
git clone https://github.com/Mohit211205/Emotion_hack.git
cd Emotion_hack
```

### Step 2 — Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate     # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install system dependencies (Mac only)
```bash
brew install portaudio
pip install pyaudio
```

---

## ▶️ How to Run

### Terminal 1 — Main Emotion Detector
```bash
source .venv/bin/activate
python3 app.py
```
> Webcam opens with real-time emotion detection

### Terminal 2 — Web Soft Bot
```bash
source .venv/bin/activate
python3 bot_server.py
```
> Open browser at `http://localhost:8080`

### Terminal 3 — Live Dashboard (Optional)
```bash
source .venv/bin/activate
python3 dashboard.py
```
> Live matplotlib dashboard with emotion graphs

### Train Custom Model (Optional)
```bash
python3 train_model.py
```
> Trains CNN on FER-2013 dataset (~30 minutes)

---

## 🤖 Emotion-Bot Behavior Mapping

| Emotion | Confidence | Bot Response | Bot Action | Bot Color |
|---------|-----------|--------------|------------|-----------|
| 😊 Happy | High | "You look happy!" | Dancing 🕺 | Green |
| 😢 Sad | Medium | "Are you okay?" | Comforting 🤗 | Blue |
| 😤 Angry | High | "Take a deep breath" | Backing away 🚶 | Red |
| 😐 Neutral | — | "Ready to assist" | Standing by 🤖 | Grey |
| 😲 Surprise | High | "Wow, surprised!" | Looking around 👀 | Cyan |
| 😨 Fear | Medium | "Don't worry!" | Moving closer 🤝 | Purple |
| 🤢 Disgust | Low | "Something wrong?" | Pausing ⏸️ | Orange |

---

## 📁 Project Structure

Emotion_hack/
│
├── app.py                  # Main application — webcam + emotion detection
├── bot_server.py           # Flask web server for animated soft bot
├── voice_emotion.py        # Audio/voice emotion detection module
├── dashboard.py            # Live matplotlib emotion dashboard
├── train_model.py          # CNN model training on FER-2013
│
├── emotion_model.h5        # Trained model (generated after training)
├── emotion_log.csv         # Session emotion log (auto-generated)
├── emotion_timeline.png    # Timeline graph (auto-generated)
├── training_results.png    # Training accuracy plot (auto-generated)
│
├── dataset/                # FER-2013 dataset (not in repo)
│   ├── train/
│   └── test/
│
├── requirements.txt        # Python dependencies
└── README.md               # This file


---

## 📋 Evaluation Criteria Coverage

| Criteria | Implementation | Status |
|----------|---------------|--------|
| Emotion Recognition Accuracy | Custom CNN on FER-2013 + DeepFace | ✅ |
| Behavior Adaptation Quality | 7 unique bot behaviors per emotion | ✅ |
| Real-time Performance | Threaded, 30FPS, 5-frame interval | ✅ |
| Multimodal Integration Bonus | Face + Voice fusion | ✅ Bonus |
| Demo & Presentation | Live web bot + dashboard | ✅ |
| Emotion Timeline Logging | CSV + Matplotlib graph | ✅ |
| Minimum 4 emotions | 7 emotions detected | ✅ |

---

## 👥 Team

| Name | Role | GitHub |
|------|------|--------|
| Mohit | Team Lead & Integration | [@Mohit211205](https://github.com/Mohit211205) |
| Janhavi | ML & Emotion Detection | [@janhavi-5002](https://github.com/janhavi-5002) |

> **HCL IIT Hackathon 2026** — Problem Statement 9: Emotion-Aware Human-System Interaction

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Built with ❤️ for HCL IIT Hackathon 2026
</div>
