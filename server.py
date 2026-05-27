import asyncio, base64, time, os, io
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np
from deepface import DeepFace
from transformers import pipeline

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading audio model...")
audio_classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    device=-1
)
print("All models ready!")

emotion_log = []

AUDIO_MAP = {
    "happy":"happy","sad":"sad","angry":"angry",
    "fearful":"fear","surprised":"surprise",
    "disgust":"disgust","calm":"neutral","neutral":"neutral"
}

BEHAVIOR = {
    "happy":   {"animation":"wave",    "color":"#00ff88","msg":"Khush ho! 😊"},
    "sad":     {"animation":"droop",   "color":"#4488ff","msg":"Udaas ho 😢"},
    "angry":   {"animation":"retreat", "color":"#ff4444","msg":"Gussa mat karo! 😠"},
    "fear":    {"animation":"shake",   "color":"#ff8800","msg":"Daro mat! 😨"},
    "surprise":{"animation":"jump",    "color":"#ffff00","msg":"Surprised! 😲"},
    "disgust": {"animation":"turn",    "color":"#aa44ff","msg":"Eww! 🤢"},
    "neutral": {"animation":"idle",    "color":"#ffffff","msg":"Neutral 😐"},
}

def fuse_emotions(face_emotion, face_conf, audio_emotion, audio_conf):
    if face_conf > 60:
        return face_emotion, "face"
    elif audio_conf > 40:
        return audio_emotion, "audio"
    else:
        return face_emotion, "face"

@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected!")
    while True:
        try:
            data = await ws.receive_json()

            face_emotion, face_conf = "neutral", 0.0
            if "frame" in data and data["frame"]:
                try:
                    frame_b64 = data["frame"].split(",")[1] if "," in data["frame"] else data["frame"]
                    img_bytes = base64.b64decode(frame_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False, silent=True)
                    face_emotion = result[0]["dominant_emotion"]
                    face_conf = float(result[0]["emotion"][face_emotion])
                except Exception as e:
                    print("Face error:", e)

            audio_emotion, audio_conf = "neutral", 0.0
            if "audio" in data and data["audio"]:
                try:
                    audio_b64 = data["audio"].split(",")[1] if "," in data["audio"] else data["audio"]
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    if len(audio_array) > 1000:
                        results = audio_classifier(audio_array, sampling_rate=16000)
                        top = results[0]
                        audio_emotion = AUDIO_MAP.get(top["label"], "neutral")
                        audio_conf = top["score"] * 100
                except Exception as e:
                    print("Audio error:", e)

            final_emotion, source = fuse_emotions(face_emotion, face_conf, audio_emotion, audio_conf)

            entry = {
                "time": round(time.time(), 2),
                "emotion": final_emotion,
                "face_emotion": face_emotion,
                "face_conf": round(face_conf, 1),
                "audio_emotion": audio_emotion,
                "audio_conf": round(audio_conf, 1),
                "source": source
            }
            emotion_log.append(entry)
            if len(emotion_log) > 100:
                emotion_log.pop(0)

            behavior = BEHAVIOR.get(final_emotion, BEHAVIOR["neutral"])
            await ws.send_json({
                "emotion": final_emotion,
                "face_emotion": face_emotion,
                "face_conf": round(face_conf, 1),
                "audio_emotion": audio_emotion,
                "audio_conf": round(audio_conf, 1),
                "source": source,
                "behavior": behavior,
                "log": emotion_log[-30:]
            })

        except Exception as e:
            print("WS Error:", e)
            break
