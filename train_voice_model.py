import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = "voice_dataset"
MODEL_PATH   = "voice_model.pkl"
SAMPLE_RATE  = 22050
DURATION     = 3.0

CREMAD_MAP  = {'ANG':'angry','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral','SAD':'sad'}
RAVDESS_MAP = {'01':'neutral','02':'neutral','03':'happy','04':'sad','05':'angry','06':'fear','07':'disgust','08':'surprise'}
VALID_EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']

def get_label(filepath):
    fname = os.path.basename(filepath)
    name  = os.path.splitext(fname)[0]
    p1 = name.split('_')
    if len(p1) >= 3:
        label = CREMAD_MAP.get(p1[2].upper())
        if label:
            return label
    p2 = name.split('-')
    if len(p2) == 7:
        return RAVDESS_MAP.get(p2[2])
    return None

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < int(sr * DURATION):
            audio = np.pad(audio, (0, int(sr * DURATION) - len(audio)))
        features = []
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta, axis=1))
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        features.extend(np.mean(mel[:20], axis=1))
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio)
        features.append(float(np.mean(zcr)))
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))
        return np.array(features, dtype=np.float32)
    except:
        return None

print("Loading dataset...")
X, y = [], []

all_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for f in files:
        if f.endswith('.wav'):
            all_files.append(os.path.join(root, f))

# Also check AudioWAV subfolder
audioWAV = os.path.join(DATASET_PATH, "AudioWAV")
if os.path.exists(audioWAV):
    for f in os.listdir(audioWAV):
        if f.endswith('.wav'):
            fp = os.path.join(audioWAV, f)
            if fp not in all_files:
                all_files.append(fp)

print(f"Found {len(all_files)} WAV files")

for i, fpath in enumerate(all_files):
    if i % 500 == 0:
        print(f"Processing {i}/{len(all_files)}... ({len(X)} valid so far)")

    label = get_label(fpath)
    if label not in VALID_EMOTIONS:
        continue

    feats = extract_features(fpath)
    if feats is None:
        continue

    X.append(feats)
    y.append(label)

print(f"\nTotal valid samples: {len(X)}")

if len(X) == 0:
    print("ERROR: No valid samples found!")
    exit()

from collections import Counter
print("\nEmotion distribution:")
for emo, cnt in sorted(Counter(y).items()):
    print(f"  {emo:10s}: {cnt}")

X = np.array(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\nTraining Random Forest...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, verbose=1,class_weight='balanced'))
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))

with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': le}, f)
print(f"\nModel saved: {MODEL_PATH}")
