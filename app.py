
# ─── Imports ───────────────────────────────────────────────────
import cv2
from deepface import DeepFace
import numpy as np
import time
import csv
import threading
import math
from collections import deque
from voice_emotion import start_voice_detection, get_voice_emotion

# ─── Config ────────────────────────────────────────────────────
DETECT_EVERY_N_FRAMES = 5
SMOOTH_WINDOW = 5
CSV_FILE = "emotion_log.csv"
CAM_W, CAM_H = 640, 480
BOT_W, BOT_H = 400, 480

# ─── State ─────────────────────────────────────────────────────
emotion_history = deque(maxlen=SMOOTH_WINDOW)
current_emotion = "neutral"
current_region  = {}
emotion_scores  = {}
start_time      = time.time()
frame_count     = 0
fps             = 0
fps_time        = time.time()
lock            = threading.Lock()

# ─── Emotion palette ───────────────────────────────────────────
PALETTE = {
    "happy":    ((0, 220, 100),  "You look happy!",         "Dancing!"),
    "sad":      ((90, 100, 255), "Are you okay?",           "Comforting..."),
    "angry":    ((50,  50, 255), "Take a deep breath.",     "Backing away."),
    "neutral":  ((180,180, 180), "Ready to assist.",        "Standing by."),
    "surprise": ((220,230,  0),  "Wow, surprised!",         "Looking around!"),
    "fear":     ((200,  60,255), "Don't worry!",            "Moving closer."),
    "disgust":  ((0,  165, 255), "Something wrong?",        "Pausing..."),
}

# ─── CSV ───────────────────────────────────────────────────────
with open(CSV_FILE, "w", newline="") as f:
    csv.writer(f).writerow(["timestamp","emotion","confidence"])

def log_csv(emotion, conf):
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([round(time.time()-start_time,2), emotion, round(conf,2)])

def smoothed():
    if not emotion_history: return "neutral"
    return max(set(emotion_history), key=list(emotion_history).count)

# ─── Custom model setup ────────────────────────────────────────
import tensorflow as tf

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
custom_model = None

try:
    import tf_keras
    custom_model = tf_keras.models.load_model("emotion_model.h5")
    print("✅ Custom trained model loaded (tf_keras)!")
except Exception as e1:
    try:
        custom_model = tf.keras.models.load_model(
            "emotion_model.h5",
            custom_objects=None,
            compile=False
        )
        print("✅ Custom trained model loaded (tf.keras)!")
    except Exception as e2:
        try:
            import h5py
            from tensorflow.keras.models import model_from_json
            with h5py.File("emotion_model.h5", "r") as f:
                model_config = f.attrs.get("model_config")
                if model_config:
                    model_config = model_config.replace(
                        '"batch_shape"', '"batch_input_shape"'
                    )
                    custom_model = model_from_json(model_config)
                    custom_model.load_weights("emotion_model.h5")
                    print("✅ Custom trained model loaded (patched)!")
        except Exception as e3:
            print(f"⚠️  All load methods failed. Using DeepFace default.")
            print(f"   Error: {e3}")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_with_custom_model(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

    if len(faces) == 0:
        return None, None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi.astype("float32") / 255.0
    face_roi = np.expand_dims(face_roi, axis=(0, -1))

    preds = custom_model.predict(face_roi, verbose=0)[0]
    scores = {EMOTION_LABELS[i]: float(preds[i] * 100) for i in range(len(EMOTION_LABELS))}
    dominant = EMOTION_LABELS[int(np.argmax(preds))]
    region = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    return dominant, scores, region

# ─── Detection thread ──────────────────────────────────────────
def detect_emotion(frame):
    global current_emotion, current_region, emotion_scores
    try:
        if custom_model is not None:
            det, scores, region = predict_with_custom_model(frame)
            if det is None:
                return
            with lock:
                emotion_history.append(det)
                current_emotion = smoothed()
                current_region  = region
                emotion_scores  = scores
                log_csv(current_emotion, scores.get(current_emotion, 0))
        else:
            res = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            with lock:
                det = res[0]["dominant_emotion"]
                emotion_history.append(det)
                current_emotion = smoothed()
                current_region  = res[0].get("region", {})
                emotion_scores  = res[0]["emotion"]
                log_csv(current_emotion, emotion_scores.get(current_emotion, 0))
    except:
        pass

# ═══════════════════════════════════════════════════════════════
#  BOT DRAWING  (pure OpenCV / NumPy, no canvas)
# ═══════════════════════════════════════════════════════════════
anim_t     = 0.0
particles  = []
tears      = []
shake_x    = 0
shake_tick = 0

def bgr(r, g, b):
    return (b, g, r)

def col(emotion):
    return PALETTE.get(emotion, PALETTE["neutral"])[0]

def lerp(a, b, t):
    return int(a + (b - a) * t)

def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=14, thickness=-1):
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    cv2.circle(img, (x1+radius, y1+radius), radius, color, thickness)
    cv2.circle(img, (x2-radius, y1+radius), radius, color, thickness)
    cv2.circle(img, (x1+radius, y2-radius), radius, color, thickness)
    cv2.circle(img, (x2-radius, y2-radius), radius, color, thickness)

def draw_bot(canvas, emotion, t):
    global particles, tears, shake_x, shake_tick

    c   = col(emotion)
    cr, cg, cb = c
    bc  = bgr(cr, cg, cb)

    # Background glow
    cx0, cy0 = BOT_W//2, BOT_H//2 - 20
    overlay = canvas.copy()
    cv2.circle(overlay, (cx0, cy0), 200, bgr(cr//4, cg//4, cb//4), -1)
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

    # Shake for angry
    if emotion == "angry":
        shake_tick += 1
        shake_x = int(math.sin(shake_tick * 0.9) * 5)
    else:
        shake_x = 0; shake_tick = 0

    # Bounce
    bounce_amt = 12 if emotion == "happy" else 4
    bounce_spd = 1.8 if emotion == "happy" else 0.7
    bounce_y   = int(math.sin(t * bounce_spd) * bounce_amt)

    cx = BOT_W // 2 + shake_x
    cy = BOT_H // 2 - 30 + bounce_y

    # ── Legs ──
    leg_swing = int(math.sin(t * 3) * 14) if emotion == "happy" else 0
    cv2.line(canvas, (cx-22, cy+100), (cx-22+leg_swing, cy+155), bc, 10)
    cv2.line(canvas, (cx+22, cy+100), (cx+22-leg_swing, cy+155), bc, 10)
    cv2.ellipse(canvas, (cx-22+leg_swing, cy+162), (14,7), 0, 0, 360, bc, -1)
    cv2.ellipse(canvas, (cx+22-leg_swing, cy+162), (14,7), 0, 0, 360, bc, -1)

    # ── Body ──
    draw_rounded_rect(canvas, cx-55, cy+8, cx+55, cy+103, bgr(18,18,30), 12)
    draw_rounded_rect(canvas, cx-55, cy+8, cx+55, cy+103, bc, 12, 2)
    # Chest light
    pulse = 0.5 + math.sin(t * 2) * 0.5
    chest_r = int(cr * pulse)
    chest_g = int(cg * pulse)
    chest_b = int(cb * pulse)
    cv2.circle(canvas, (cx, cy+55), 10, bgr(chest_r, chest_g, chest_b), -1)
    cv2.circle(canvas, (cx-3, cy+52), 3, (255,255,255), -1)

    # ── Arms ──
    if emotion == "happy":
        wave = int(math.sin(t * 3) * 15)
        cv2.line(canvas, (cx-55, cy+22), (cx-95, cy-8+wave),  bc, 9)
        cv2.line(canvas, (cx+55, cy+22), (cx+95, cy-8-wave),  bc, 9)
        cv2.circle(canvas, (cx-95, cy-8+wave), 9, bc, -1)
        cv2.circle(canvas, (cx+95, cy-8-wave), 9, bc, -1)
    elif emotion == "sad":
        cv2.line(canvas, (cx-55, cy+30), (cx-85, cy+88), bc, 9)
        cv2.line(canvas, (cx+55, cy+30), (cx+85, cy+88), bc, 9)
        cv2.circle(canvas, (cx-85, cy+88), 8, bc, -1)
        cv2.circle(canvas, (cx+85, cy+88), 8, bc, -1)
    elif emotion == "angry":
        punch = int(math.sin(t * 4) * 8)
        cv2.line(canvas, (cx-55, cy+28), (cx-100+punch, cy+18), bc, 9)
        cv2.line(canvas, (cx+55, cy+28), (cx+100-punch, cy+18), bc, 9)
        cv2.circle(canvas, (cx-100+punch, cy+18), 10, bc, -1)
        cv2.circle(canvas, (cx+100-punch, cy+18), 10, bc, -1)
    elif emotion == "fear":
        tr = int(math.sin(t * 9) * 4)
        cv2.line(canvas, (cx-55, cy+28), (cx-82+tr, cy+50), bc, 9)
        cv2.line(canvas, (cx+55, cy+28), (cx+82+tr, cy+50), bc, 9)
        cv2.circle(canvas, (cx-82+tr, cy+50), 8, bc, -1)
        cv2.circle(canvas, (cx+82+tr, cy+50), 8, bc, -1)
    else:
        cv2.line(canvas, (cx-55, cy+28), (cx-88, cy+48), bc, 9)
        cv2.line(canvas, (cx+55, cy+28), (cx+88, cy+48), bc, 9)
        cv2.circle(canvas, (cx-88, cy+48), 8, bc, -1)
        cv2.circle(canvas, (cx+88, cy+48), 8, bc, -1)

    # ── Neck ──
    draw_rounded_rect(canvas, cx-12, cy-10, cx+12, cy+10, bgr(22,22,35), 4)
    draw_rounded_rect(canvas, cx-12, cy-10, cx+12, cy+10, bc, 4, 1)

    # ── Head ──
    draw_rounded_rect(canvas, cx-68, cy-82, cx+68, cy+8, bgr(18,18,30), 16)
    draw_rounded_rect(canvas, cx-68, cy-82, cx+68, cy+8, bc, 16, 3)

    # Scan line
    scan_y = cy - 82 + int((t * 28) % 90)
    scan_overlay = canvas.copy()
    cv2.rectangle(scan_overlay, (cx-68, scan_y), (cx+68, scan_y+5),
                  bgr(cr//6, cg//6, cb//6), -1)
    cv2.addWeighted(scan_overlay, 0.5, canvas, 0.5, 0, canvas)

    # ── Antenna ──
    cv2.line(canvas, (cx, cy-82), (cx, cy-108), bc, 3)
    ant_pulse = 0.5 + math.sin(t * 2.5) * 0.5
    ant_c = bgr(int(cr*ant_pulse), int(cg*ant_pulse), int(cb*ant_pulse))
    cv2.circle(canvas, (cx, cy-112), 8, ant_c, -1)
    cv2.circle(canvas, (cx-2, cy-114), 3, (255,255,255), -1)

    # ── Eyes ──
    ey = cy - 48
    el, er = cx-24, cx+24

    if emotion == "happy":
        cv2.ellipse(canvas, (el, ey+4), (16,8), 0, 180, 360, bc, 3)
        cv2.ellipse(canvas, (er, ey+4), (16,8), 0, 180, 360, bc, 3)
    elif emotion == "sad":
        cv2.ellipse(canvas, (el, ey-4), (16,8), 0,   0, 180, bc, 3)
        cv2.ellipse(canvas, (er, ey-4), (16,8), 0,   0, 180, bc, 3)
        # Sad brows
        cv2.line(canvas, (el-16, ey-22), (el+8, ey-14), bc, 3)
        cv2.line(canvas, (er-8,  ey-14), (er+16, ey-22), bc, 3)
    elif emotion == "angry":
        cv2.ellipse(canvas, (el, ey), (16,9), 15, 0, 360, bc, -1)
        cv2.ellipse(canvas, (er, ey), (16,9),-15, 0, 360, bc, -1)
        cv2.circle(canvas, (el, ey), 6, bgr(10,10,20), -1)
        cv2.circle(canvas, (er, ey), 6, bgr(10,10,20), -1)
        # V brows
        cv2.line(canvas, (el-20, ey-18), (el+10, ey-8), bc, 3)
        cv2.line(canvas, (er+20, ey-18), (er-10, ey-8), bc, 3)
    elif emotion == "surprise":
        cv2.circle(canvas, (el, ey), 18, bc, -1)
        cv2.circle(canvas, (er, ey), 18, bc, -1)
        cv2.circle(canvas, (el, ey), 9,  bgr(10,10,20), -1)
        cv2.circle(canvas, (er, ey), 9,  bgr(10,10,20), -1)
        cv2.circle(canvas, (el-4, ey-4), 4, (255,255,255), -1)
        cv2.circle(canvas, (er-4, ey-4), 4, (255,255,255), -1)
    elif emotion == "fear":
        tr = int(math.sin(t * 10) * 2)
        cv2.circle(canvas, (el+tr, ey), 17, bc, -1)
        cv2.circle(canvas, (er+tr, ey), 17, bc, -1)
        cv2.circle(canvas, (el+tr+3, ey+3), 9, bgr(10,10,20), -1)
        cv2.circle(canvas, (er+tr-3, ey+3), 9, bgr(10,10,20), -1)
    elif emotion == "disgust":
        cv2.ellipse(canvas, (el, ey), (16,6), 0, 0, 360, bc, -1)
        cv2.ellipse(canvas, (er, ey), (16,6), 0, 0, 360, bc, -1)
        cv2.line(canvas, (el-16, ey-17), (el+12, ey-11), bc, 3)
        cv2.line(canvas, (er-12, ey-13), (er+14, ey-19), bc, 3)
    else:
        cv2.circle(canvas, (el, ey), 14, bc, -1)
        cv2.circle(canvas, (er, ey), 14, bc, -1)
        cv2.circle(canvas, (el, ey),  6, bgr(10,10,20), -1)
        cv2.circle(canvas, (er, ey),  6, bgr(10,10,20), -1)
        cv2.circle(canvas, (el-3, ey-3), 3, (255,255,255), -1)
        cv2.circle(canvas, (er-3, ey-3), 3, (255,255,255), -1)

    # ── Mouth ──
    my = cy - 20
    if emotion == "happy":
        pts = np.array([[cx-28,my],[cx,my+22],[cx+28,my]], np.int32)
        cv2.polylines(canvas, [pts], False, bc, 3)
    elif emotion == "sad":
        pts = np.array([[cx-24,my+16],[cx,my+4],[cx+24,my+16]], np.int32)
        cv2.polylines(canvas, [pts], False, bc, 3)
    elif emotion == "angry":
        pts = np.array([[cx-24,my+12],[cx,my+5],[cx+24,my+12]], np.int32)
        cv2.polylines(canvas, [pts], False, bc, 3)
        cv2.rectangle(canvas,(cx-18,my+6),(cx+18,my+14),
                      bgr(cr//3,cg//3,cb//3),-1)
    elif emotion == "surprise":
        cv2.ellipse(canvas,(cx,my+10),(16,14),0,0,360,bc,-1)
        cv2.ellipse(canvas,(cx,my+10),(16,14),0,0,360,bc,2)
    elif emotion == "fear":
        tr = int(math.sin(t*10)*2)
        pts = np.array([[cx-22+tr,my+8],[cx,my+18+tr],[cx+22+tr,my+8]],np.int32)
        cv2.polylines(canvas,[pts],False,bc,3)
    elif emotion == "disgust":
        cv2.line(canvas,(cx-22,my+10),(cx+4,my+10),bc,3)
        cv2.line(canvas,(cx+4,my+10),(cx+22,my+4),bc,3)
    else:
        cv2.line(canvas,(cx-20,my+10),(cx+20,my+10),bc,3)

    # ── Tears (sad) ──
    if emotion == "sad":
        if np.random.rand() < 0.06:
            tears.append({"x":float(el),"y":float(ey+10),"vy":1.5,"life":1.0})
            tears.append({"x":float(er),"y":float(ey+10),"vy":1.5,"life":1.0})
    else:
        tears.clear()

    alive = []
    for tr2 in tears:
        tr2["y"]  += tr2["vy"]; tr2["vy"] += 0.1; tr2["life"] -= 0.015
        if tr2["life"] > 0:
            alpha = tr2["life"]
            tx,ty = int(tr2["x"]),int(tr2["y"])
            if 0<=tx<BOT_W and 0<=ty<BOT_H:
                cv2.ellipse(canvas,(tx,ty),(3,5),0,0,360,
                            bgr(int(85*alpha),int(119*alpha),int(255*alpha)),-1)
            alive.append(tr2)
    tears[:] = alive

    # ── Particles (happy) ──
    if emotion == "happy" and np.random.rand() < 0.18:
        for _ in range(6):
            particles.append({
                "x": float(cx + np.random.randint(-80,80)),
                "y": float(cy + np.random.randint(-60,40)),
                "vx": float(np.random.uniform(-2.5,2.5)),
                "vy": float(np.random.uniform(-3,-1)),
                "life": 1.0,
                "decay": float(np.random.uniform(0.02,0.04)),
                "size": int(np.random.randint(3,7))
            })
    if emotion != "happy":
        particles.clear()

    alive_p = []
    for p in particles:
        p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["vy"]+=0.06; p["life"]-=p["decay"]
        if p["life"]>0:
            px,py = int(p["x"]),int(p["y"])
            if 0<=px<BOT_W and 0<=py<BOT_H:
                a = p["life"]
                cv2.circle(canvas,(px,py),p["size"],
                           bgr(int(cr*a),int(cg*a),int(cb*a)),-1)
            alive_p.append(p)
    particles[:] = alive_p

    # ── Sweat (fear) ──
    if emotion == "fear":
        sw = int(math.sin(t*2)*3)
        cv2.circle(canvas,(cx+72,cy-60+sw),4,bgr(100,40,200),-1)
        cv2.circle(canvas,(cx+78,cy-44+sw),3,bgr(100,40,200),-1)

    # ── Anger marks ──
    if emotion == "angry":
        ak = bgr(180,30,50)
        cv2.line(canvas,(cx-90+shake_x,cy-55),(cx-80+shake_x,cy-68),ak,2)
        cv2.line(canvas,(cx-84+shake_x,cy-55),(cx-74+shake_x,cy-68),ak,2)
        cv2.line(canvas,(cx+80+shake_x,cy-55),(cx+90+shake_x,cy-68),ak,2)
        cv2.line(canvas,(cx+74+shake_x,cy-55),(cx+84+shake_x,cy-68),ak,2)


def draw_bot_panel(emotion, t):
    canvas = np.full((BOT_H, BOT_W, 3), (12, 10, 18), dtype=np.uint8)
    draw_bot(canvas, emotion, t)

    c  = col(emotion)
    bc = bgr(*c)
    info = PALETTE.get(emotion, PALETTE["neutral"])

    # Emotion label top
    cv2.rectangle(canvas, (0,0),(BOT_W,44), bgr(10,8,16),-1)
    label = emotion.upper()
    lsz   = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    cv2.putText(canvas, label,
                (BOT_W//2 - lsz[0]//2, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, bc, 2)

    # Response text bottom
    cv2.rectangle(canvas,(0,BOT_H-70),(BOT_W,BOT_H),bgr(10,8,16),-1)
    cv2.line(canvas,(0,BOT_H-70),(BOT_W,BOT_H-70),bc,1)
    resp = info[1]
    rsz  = cv2.getTextSize(resp, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    cv2.putText(canvas, resp,
                (BOT_W//2 - rsz[0]//2, BOT_H-45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, bc, 1)
    action = "BOT: " + info[2]
    asz    = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(canvas, action,
                (BOT_W//2 - asz[0]//2, BOT_H-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,200), 1)

    # Divider line left
    cv2.line(canvas,(0,0),(0,BOT_H),bc,2)

    return canvas


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════
def run_app():
	start_voice_detection()
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

	print("Starting... Press Q to quit")

	while True:
	    ret, frame = cap.read()
	    if not ret:
	        break

	    frame_count += 1
	    now   = time.time()
	    fps   = 1.0 / (now - fps_time + 1e-6)
	    fps_time = now
	    anim_t  += 0.045

	    # Detection every N frames
	    if frame_count % DETECT_EVERY_N_FRAMES == 0:
	        t2 = threading.Thread(target=detect_emotion, args=(frame.copy(),))
	        t2.daemon = True
	        t2.start()

	    with lock:
	        emotion = current_emotion
	        region  = current_region.copy()
	        scores  = emotion_scores.copy()

	    c_info = PALETTE.get(emotion, PALETTE["neutral"])
	    color  = c_info[0]
	    bc     = bgr(*color)

	    # ── Resize camera feed ──
	    cam_display = cv2.resize(frame, (CAM_W, CAM_H))

	    # ── Face box ──
	    if region and region.get("w",0)>0:
	        x,y,w,h = region["x"],region["y"],region["w"],region["h"]
	        cv2.rectangle(cam_display,(x,y),(x+w,y+h),bc,2)
	        cv2.putText(cam_display, emotion.upper(),(x,y-10),
	                    cv2.FONT_HERSHEY_SIMPLEX,0.7,bc,2)

	    # ── Top bar on camera ──
	    cv2.rectangle(cam_display,(0,0),(CAM_W,100),(15,12,22),-1)
	    cv2.putText(cam_display, f"Emotion: {emotion.upper()}",(12,36),
	                cv2.FONT_HERSHEY_SIMPLEX,0.9,bc,2)
	    cv2.putText(cam_display, c_info[1],(12,68),
	                cv2.FONT_HERSHEY_SIMPLEX,0.55,bc,1)
	    cv2.putText(cam_display, f"FPS: {fps:.1f}",(CAM_W-110,30),
	                cv2.FONT_HERSHEY_SIMPLEX,0.55,(80,220,80),1)

	    # ── Confidence mini bars ──
	    if scores:
	        yp = 115
	        for emo,sc in sorted(scores.items(),key=lambda x:-x[1])[:5]:
	            blen = int(sc*1.5)
	            bc2  = bgr(*col(emo)) if emo==emotion else (60,60,60)
	            cv2.rectangle(cam_display,(12,yp),(12+blen,yp+12),bc2,-1)
	            cv2.putText(cam_display,f"{emo[:5]}",(12,yp+10),
	                        cv2.FONT_HERSHEY_SIMPLEX,0.32,(200,200,200),1)
	            yp += 18

	    # ── Bottom bar ──
	    voice_emo = get_voice_emotion()
	    cv2.rectangle(cam_display,(0,CAM_H-36),(CAM_W,CAM_H),(15,12,22),-1)
	    cv2.line(cam_display,(0,CAM_H-36),(CAM_W,CAM_H-36),bc,1)
	    elapsed = int(time.time()-start_time)
	    cv2.putText(cam_display,
	                f"Face:{emotion}  Voice:{voice_emo}  Time:{elapsed}s",
	                (10,CAM_H-12),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,220,220),1)

	    # ── Bot panel ──
	    bot_panel = draw_bot_panel(emotion, anim_t)
	    # Resize bot to match camera height
	    bot_panel = cv2.resize(bot_panel,(BOT_W, CAM_H))

	    # ── Combine side by side ──
	    combined = np.hstack([cam_display, bot_panel])

	    cv2.imshow("Emotion-Aware Bot", combined)

	    if cv2.waitKey(1) & 0xFF == ord("q"):
	        break

	cap.release()
	cv2.destroyAllWindows()
	print(f"Done! Log: {CSV_FILE}")

	# ── Timeline ──
	try:
	    import matplotlib.pyplot as plt
	    times,emotions_log=[],[]
	    with open(CSV_FILE) as f:
	        for row in csv.DictReader(f):
	            times.append(float(row["timestamp"]))
	            emotions_log.append(row["emotion"])
	    elist = list(PALETTE.keys())
	    yv = [elist.index(e) if e in elist else 0 for e in emotions_log]
	    plt.figure(figsize=(14,4))
	    plt.plot(times,yv,marker='o',color='cyan',linewidth=1.5,markersize=4)
	    plt.yticks(range(len(elist)),elist)
	    plt.xlabel("Time (s)"); plt.title("Emotion Timeline")
	    plt.grid(True,alpha=0.3); plt.tight_layout()
	    plt.savefig("emotion_timeline.png"); plt.savefig("emotion_timeline.png")
	    print("Timeline saved!")
	except Exception as e:
	    print(f"Timeline error: {e}")

if __name__ == "__main__":
	run_app()
