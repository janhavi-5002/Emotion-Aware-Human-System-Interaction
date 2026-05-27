import sounddevice as sd
import numpy as np
import librosa
import threading
import pickle
import os
import time

SAMPLE_RATE = 22050
DURATION    = 3
MODEL_PATH  = "voice_model.pkl"

current_voice_emotion = "neutral"
voice_lock = threading.Lock()

# ─── Load trained model ───────────────────────────
model_data     = None
voice_model    = None
label_encoder  = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        voice_model   = model_data['model']
        label_encoder = model_data['label_encoder']
        print("Voice ML model loaded!")
    except Exception as e:
        print(f"Voice model load failed: {e} — using rule-based fallback")
else:
    print("voice_model.pkl not found — using rule-based fallback")

# ─── Feature extraction ───────────────────────────
def extract_features(audio):
    try:
        if len(audio) < int(SAMPLE_RATE * DURATION):
            audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION) - len(audio)))
        features = []
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta, axis=1))
        chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
        features.extend(np.mean(chroma, axis=1))
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE)
        features.extend(np.mean(mel[:20], axis=1))
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio)
        features.append(float(np.mean(zcr)))
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))
        return np.array(features, dtype=np.float32).reshape(1, -1)
    except:
        return None

# ─── Rule-based fallback ──────────────────────────
def rule_based(audio):
    try:
        energy = np.mean(librosa.feature.rms(y=audio))
        pitches, mags = librosa.piptrack(y=audio, sr=SAMPLE_RATE)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        tempo, _ = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
        if energy > 0.08 and pitch > 200:
            return "angry"
        elif energy > 0.06 and float(tempo) > 120:
            return "happy"
        elif energy < 0.02 and pitch < 150:
            return "sad"
        else:
            return "neutral"
    except:
        return "neutral"

# ─── Analyze audio ────────────────────────────────
def analyze_voice(audio_data):
    global current_voice_emotion
    try:
        audio = audio_data.flatten().astype(np.float32)

        if voice_model is not None:
            feats = extract_features(audio)
            if feats is not None:
                pred = voice_model.predict(feats)[0]
                emotion = label_encoder.inverse_transform([pred])[0]
            else:
                emotion = rule_based(audio)
        else:
            emotion = rule_based(audio)

        with voice_lock:
            current_voice_emotion = emotion
        print(f"Voice: {emotion}")

    except Exception as e:
        print(f"Voice error: {e}")

# ─── Record loop ──────────────────────────────────
def record_and_analyze():
    while True:
        try:
            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=1, dtype='float32')
            sd.wait()
            t = threading.Thread(target=analyze_voice, args=(audio,))
            t.daemon = True
            t.start()
        except Exception as e:
            print(f"Recording error: {e}")
            time.sleep(1)

def start_voice_detection():
    t = threading.Thread(target=record_and_analyze)
    t.daemon = True
    t.start()
    print("Voice detection started!")

def get_voice_emotion():
    with voice_lock:
        return current_voice_emotion
