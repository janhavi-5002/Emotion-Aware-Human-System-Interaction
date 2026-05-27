from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import csv, os, base64, threading, time, io
import numpy as np
from deepface import DeepFace
from voice_emotion import start_voice_detection, get_voice_emotion

app = Flask(__name__)
CORS(app)
CSV_FILE = "emotion_log.csv"

# ─── State ────────────────────────────────────────────────────
current_emotion   = "neutral"
current_confidence = 0.0
current_scores    = {}
start_time        = time.time()
lock              = threading.Lock()

with open(CSV_FILE, "w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "emotion", "confidence"])

def log_csv(emotion, conf):
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([round(time.time() - start_time, 2), emotion, round(conf, 2)])

# ─── Emotion detection from base64 frame ──────────────────────
def analyze_frame(b64_data):
    global current_emotion, current_confidence, current_scores
    try:
        img_bytes = base64.b64decode(b64_data.split(",")[1])
        import cv2
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)
        emotion   = result[0]["dominant_emotion"]
        scores    = result[0]["emotion"]
        confidence = scores.get(emotion, 0)
        with lock:
            current_emotion    = emotion
            current_confidence = confidence
            current_scores     = scores
        log_csv(emotion, confidence)
    except Exception as e:
        pass

# ─── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if data and "frame" in data:
        t = threading.Thread(target=analyze_frame, args=(data["frame"],))
        t.daemon = True
        t.start()
    with lock:
        return jsonify({
            "emotion":    current_emotion,
            "confidence": round(current_confidence, 2),
            "scores":     {k: round(v, 2) for k, v in current_scores.items()},
            "voice":      get_voice_emotion(),
            "elapsed":    round(time.time() - start_time, 1)
        })

@app.route("/history")
def history():
    try:
        with open(CSV_FILE) as f:
            rows = list(csv.DictReader(f))
        return jsonify(rows[-60:])
    except:
        return jsonify([])

# ─── HTML ─────────────────────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html>
<head>
<title>EMOTION.SYS</title>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#04040f;font-family:'Share Tech Mono',monospace;color:#00ffcc;min-height:100vh;overflow-x:hidden}
.scanlines{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:9;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.04) 2px,rgba(0,0,0,0.04) 4px)}
.scan-beam{position:fixed;top:-3px;left:0;width:100%;height:3px;background:linear-gradient(90deg,transparent,rgba(0,255,204,0.07),transparent);animation:sb 7s linear infinite;z-index:10;pointer-events:none}
@keyframes sb{0%{top:-3px}100%{top:100vh}}
header{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;border-bottom:1px solid rgba(0,255,204,0.08);position:relative;z-index:5}
header::after{content:'';position:absolute;bottom:0;left:0;width:100%;height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,204,0.5),transparent)}
.logo{font-family:'Orbitron',monospace;font-size:18px;font-weight:900;letter-spacing:6px;color:#00ffcc}
.logo span{color:#ff2d78}
.hdr-r{display:flex;align-items:center;gap:18px}
.live-b{display:flex;align-items:center;gap:6px;font-size:10px;letter-spacing:2px;color:rgba(0,255,204,0.5)}
.ldot{width:7px;height:7px;border-radius:50%;background:#00ff88;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.25;transform:scale(0.6)}}
#clk{font-size:12px;letter-spacing:3px;color:rgba(0,255,204,0.25);font-family:'Orbitron',monospace}
.main{display:grid;grid-template-columns:1fr 1fr 320px;gap:12px;padding:14px 20px;align-items:start}
.panel{background:#080818;border:1px solid rgba(0,255,204,0.08);border-radius:3px;padding:14px;position:relative;overflow:hidden}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,204,0.35),transparent)}
.ptitle{font-size:9px;letter-spacing:4px;color:rgba(0,255,204,0.28);margin-bottom:12px;text-transform:uppercase}
.cn{position:absolute;width:9px;height:9px;border-color:rgba(0,255,204,0.2);border-style:solid}
.tl{top:4px;left:4px;border-width:1px 0 0 1px}
.tr{top:4px;right:4px;border-width:1px 1px 0 0}
.bl{bottom:4px;left:4px;border-width:0 0 1px 1px}
.br{bottom:4px;right:4px;border-width:0 1px 1px 0}

/* Camera panel */
.cam-wrap{position:relative;width:100%;aspect-ratio:4/3;background:#02020a;border:1px solid rgba(0,255,204,0.06);border-radius:2px;overflow:hidden}
#video{width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}
#faceOverlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.cam-status{display:flex;align-items:center;gap:6px;margin-top:8px;font-size:10px;letter-spacing:2px;color:rgba(0,255,204,0.35)}
.cam-dot{width:6px;height:6px;border-radius:50%;background:#ff3344}
.cam-dot.active{background:#00ff88;animation:pulse 0.8s infinite}
.fps-badge{margin-left:auto;font-size:9px;color:rgba(0,255,204,0.2);letter-spacing:2px}

/* Bot panel */
.bot-canvas-wrap{display:flex;justify-content:center;margin-bottom:10px}
#botC{background:#06060f;border:1px solid rgba(0,255,204,0.06);border-radius:3px}
.emo-name{font-family:'Orbitron',monospace;font-size:26px;font-weight:900;letter-spacing:5px;text-align:center;transition:color 0.5s,text-shadow 0.5s;margin-bottom:6px}
.emo-resp{font-size:11px;color:#444;letter-spacing:1px;text-align:center;margin-bottom:8px}
.action-pill{display:block;text-align:center;padding:6px 10px;border:1px solid rgba(255,230,0,0.15);border-radius:2px;font-size:10px;letter-spacing:2px;color:#ffe600;transition:all 0.4s;margin-bottom:10px}
.hist-row{display:flex;gap:4px;flex-wrap:wrap;justify-content:center;margin-bottom:10px}
.hchip{font-size:8px;padding:2px 7px;border-radius:2px;letter-spacing:1px;border:1px solid currentColor;opacity:0.5}
.tl-wrap{display:flex;align-items:flex-end;gap:2px;height:40px;margin-top:10px;overflow:hidden;border-top:1px solid rgba(0,255,204,0.05);padding-top:8px}
.tlbar{min-width:8px;border-radius:1px 1px 0 0;transition:height 0.4s,background 0.4s}

/* Right panel */
.bar-row{display:flex;align-items:center;gap:7px;margin-bottom:8px}
.blbl{font-size:9px;letter-spacing:1px;width:54px;text-transform:capitalize;transition:color 0.4s}
.btrack{flex:1;height:6px;background:#0a0a1a;border:1px solid rgba(255,255,255,0.03);border-radius:1px;overflow:hidden}
.bfill{height:100%;border-radius:1px;transition:width 1s ease}
.bval{font-size:9px;width:28px;text-align:right;color:#2a2a3a}
.conf-big{font-family:'Orbitron',monospace;font-size:36px;font-weight:900;letter-spacing:3px;margin-top:10px;padding-top:10px;border-top:1px solid rgba(0,255,204,0.07);transition:color 0.4s,text-shadow 0.4s;color:#00ffcc}
.conf-lbl{font-size:8px;color:rgba(0,255,204,0.18);letter-spacing:4px;margin-top:3px}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:12px}
.sbox{background:#04040e;border:1px solid rgba(0,255,204,0.06);border-radius:2px;padding:8px 10px}
.sval{font-family:'Orbitron',monospace;font-size:17px;font-weight:700;color:#00ffcc}
.slbl{font-size:7px;color:#222;letter-spacing:2px;margin-top:2px;text-transform:uppercase}
.voice-row{display:flex;align-items:center;gap:8px;margin-top:12px;padding-top:10px;border-top:1px solid rgba(0,255,204,0.07)}
.vdot{width:5px;height:5px;border-radius:50%;background:#ff7700;animation:pulse 0.7s infinite}
.vemo{font-family:'Orbitron',monospace;font-size:12px;font-weight:700;letter-spacing:3px;transition:color 0.4s;color:#ff7700}
.wave{display:flex;align-items:center;gap:2px;height:24px;margin-top:6px}
.wbar{width:3px;border-radius:2px;background:rgba(255,119,0,0.3);transition:height 0.1s}

.glitch{animation:glitch 0.08s steps(2) forwards}
@keyframes glitch{0%{transform:translateX(-3px)}50%{transform:translateX(3px)}100%{transform:translateX(0)}}

@media(max-width:1100px){.main{grid-template-columns:1fr 1fr}.right-col{grid-column:1/-1}}
@media(max-width:700px){.main{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="scanlines"></div><div class="scan-beam"></div>

<header>
  <div class="logo">EMOTION<span>.</span>SYS</div>
  <div class="hdr-r">
    <div class="live-b"><span class="ldot"></span>LIVE</div>
    <div id="clk">00:00:00</div>
  </div>
</header>

<div class="main">

  <!-- CAMERA -->
  <div class="panel">
    <div class="cn tl"></div><div class="cn tr"></div><div class="cn bl"></div><div class="cn br"></div>
    <div class="ptitle">camera.feed / face.detect</div>
    <div class="cam-wrap">
      <video id="video" autoplay muted playsinline></video>
      <canvas id="faceOverlay"></canvas>
    </div>
    <div class="cam-status">
      <div class="cam-dot" id="camDot"></div>
      <span id="camTxt">Initializing...</span>
      <span class="fps-badge" id="fpsBadge">-- FPS</span>
    </div>
  </div>

  <!-- BOT -->
  <div class="panel">
    <div class="cn tl"></div><div class="cn tr"></div><div class="cn bl"></div><div class="cn br"></div>
    <div class="ptitle">bot.render / emotion.state</div>
    <div class="bot-canvas-wrap">
      <canvas id="botC" width="220" height="270"></canvas>
    </div>
    <div class="emo-name" id="emoName" style="color:#aaaaaa">NEUTRAL</div>
    <div class="emo-resp" id="emoResp">Ready to assist you.</div>
    <div class="action-pill" id="emoAction">BOT: Standing by.</div>
    <div class="hist-row" id="histRow"></div>
    <div class="tl-wrap" id="tlWrap"></div>
  </div>

  <!-- RIGHT PANEL -->
  <div class="panel" style="grid-row:1/3">
    <div class="cn tl"></div><div class="cn tr"></div><div class="cn bl"></div><div class="cn br"></div>
    <div class="ptitle">confidence.matrix</div>
    <div id="barsWrap"></div>
    <div class="conf-big" id="confBig">0%</div>
    <div class="conf-lbl">OVERALL CONFIDENCE</div>

    <div class="stat-grid">
      <div class="sbox"><div class="sval" id="sCnt">0</div><div class="slbl">Detections</div></div>
      <div class="sbox"><div class="sval" id="sTop" style="font-size:11px">----</div><div class="slbl">Top Emotion</div></div>
      <div class="sbox"><div class="sval" id="sChg">0</div><div class="slbl">Changes</div></div>
      <div class="sbox"><div class="sval" id="sConf">0%</div><div class="slbl">Avg Conf</div></div>
    </div>

    <div class="voice-row">
      <div class="vdot"></div>
      <div class="vemo" id="vEmo">NEUTRAL</div>
    </div>
    <div class="wave" id="wave"></div>

    <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(0,255,204,0.07)">
      <div style="font-size:8px;color:rgba(0,255,204,0.18);letter-spacing:3px;margin-bottom:4px">SESSION</div>
      <div style="font-family:'Orbitron',monospace;font-size:18px;letter-spacing:3px;color:rgba(0,255,204,0.35)" id="sessT">00:00</div>
    </div>
  </div>

</div>

<script>
const ECFG={
  happy:   {color:'#00dd77',resp:'You look happy!',       action:'Dancing & celebrating!'},
  sad:     {color:'#5577ff',resp:'Are you okay?',          action:'Offering comfort...'},
  angry:   {color:'#ff3344',resp:'Take a deep breath.',    action:'Backing away.'},
  neutral: {color:'#aaaaaa',resp:'Ready to assist you.',   action:'Standing by.'},
  surprise:{color:'#ffe600',resp:'Wow, surprised!',        action:'Looking around!'},
  fear:    {color:'#bb44ff',resp:"Don't worry!",           action:'Moving closer.'},
  disgust: {color:'#ff7700',resp:'Something wrong?',       action:'Pausing...'},
};

// ── Camera setup ──
const video = document.getElementById('video');
const fov   = document.getElementById('faceOverlay');
const fovCtx= fov.getContext('2d');
let camReady = false;
let lastFps = 0, fpsFrames = 0, fpsTime = Date.now();

navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false})
  .then(stream=>{
    video.srcObject = stream;
    video.onloadedmetadata = ()=>{
      fov.width  = video.videoWidth;
      fov.height = video.videoHeight;
      camReady = true;
      document.getElementById('camDot').classList.add('active');
      document.getElementById('camTxt').textContent = 'Camera active';
    };
  })
  .catch(e=>{
    document.getElementById('camTxt').textContent = 'Camera error: '+e.message;
  });

// ── Frame capture & send ──
const offscreen = document.createElement('canvas');
offscreen.width = 320; offscreen.height = 240;
const offCtx = offscreen.getContext('2d');
let analyzing = false;

function captureAndSend(){
  if(!camReady || analyzing) return;
  analyzing = true;
  offCtx.drawImage(video, 0, 0, 320, 240);
  const b64 = offscreen.toDataURL('image/jpeg', 0.7);

  fetch('/analyze',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({frame: b64})
  })
  .then(r=>r.json())
  .then(data=>{
    analyzing = false;
    const emo = (data.emotion||'neutral').toLowerCase();
    const conf = data.confidence||0;
    const scores = data.scores||{};
    const voice  = data.voice||'neutral';
    updateAll(emo, conf, scores, voice);
    fpsFrames++;
    const now = Date.now();
    if(now - fpsTime >= 2000){
      lastFps = Math.round(fpsFrames*1000/(now-fpsTime));
      document.getElementById('fpsBadge').textContent = lastFps+' FPS';
      fpsFrames=0; fpsTime=now;
    }
  })
  .catch(()=>{ analyzing=false; });
}

setInterval(captureAndSend, 500);

// ── Face overlay ──
function drawFaceBox(x,y,w,h,emo){
  if(!camReady) return;
  fovCtx.clearRect(0,0,fov.width,fov.height);
  if(!w||w<=0) return;
  // Mirror x because video is mirrored
  const mx = fov.width - x - w;
  const col = ECFG[emo]?.color||'#00ffcc';
  fovCtx.strokeStyle = col;
  fovCtx.lineWidth   = 2;
  // Corner brackets
  const cs = 18;
  fovCtx.beginPath();
  [[mx,y],[mx+w,y],[mx,y+h],[mx+w,y+h]].forEach(([cx,cy])=>{
    const dx = cx===mx?1:-1, dy = cy===y?1:-1;
    fovCtx.moveTo(cx,cy); fovCtx.lineTo(cx+dx*cs,cy);
    fovCtx.moveTo(cx,cy); fovCtx.lineTo(cx,cy+dy*cs);
  });
  fovCtx.stroke();
  // Label
  fovCtx.fillStyle = col;
  fovCtx.font = 'bold 13px Share Tech Mono,monospace';
  fovCtx.letterSpacing = '2px';
  fovCtx.fillText(emo.toUpperCase(), mx, y-8);
}

// ── Bot canvas ──
const BC = document.getElementById('botC');
const bctx = BC.getContext('2d');
const BW=220, BH=270;
let animT=0, particles=[], tears=[], shakeX=0, shakeTick=0, curEmo='neutral';

function h2r(h){return[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]}
function rrect(c,x,y,w,h,r){c.beginPath();c.moveTo(x+r,y);c.lineTo(x+w-r,y);c.arcTo(x+w,y,x+w,y+r,r);c.lineTo(x+w,y+h-r);c.arcTo(x+w,y+h,x+w-r,y+h,r);c.lineTo(x+r,y+h);c.arcTo(x,y+h,x,y+h-r,r);c.lineTo(x,y+r);c.arcTo(x,y,x+r,y,r);c.closePath()}

function drawBot(emo){
  const cfg=ECFG[emo]||ECFG.neutral;
  const col=cfg.color; const[r,g,b]=h2r(col);
  bctx.clearRect(0,0,BW,BH); animT+=0.05;
  if(emo==='angry'){shakeTick++;shakeX=Math.round(Math.sin(shakeTick*0.9)*5);}
  else{shakeX=0;shakeTick=0;}
  const bA=emo==='happy'?10:emo==='fear'?4:3, bS=emo==='happy'?1.9:0.7;
  const by=Math.round(Math.sin(animT*bS)*bA);
  const cx=BW/2+shakeX, cy=BH/2-15+by;

  // glow
  const grd=bctx.createRadialGradient(cx,cy,8,cx,cy,120);
  grd.addColorStop(0,`rgba(${r},${g},${b},0.1)`); grd.addColorStop(1,'transparent');
  bctx.fillStyle=grd; bctx.fillRect(0,0,BW,BH);

  // legs
  bctx.strokeStyle=col; bctx.lineWidth=8; bctx.lineCap='round';
  const ls=emo==='happy'?Math.round(Math.sin(animT*3)*11):0;
  bctx.beginPath();bctx.moveTo(cx-18,cy+72);bctx.lineTo(cx-18+ls,cy+115);bctx.stroke();
  bctx.beginPath();bctx.moveTo(cx+18,cy+72);bctx.lineTo(cx+18-ls,cy+115);bctx.stroke();
  bctx.fillStyle=col;
  bctx.beginPath();bctx.ellipse(cx-18+ls,cy+121,11,5,0,0,Math.PI*2);bctx.fill();
  bctx.beginPath();bctx.ellipse(cx+18-ls,cy+121,11,5,0,0,Math.PI*2);bctx.fill();

  // body
  bctx.fillStyle='#0d0d22'; bctx.strokeStyle=col; bctx.lineWidth=2;
  rrect(bctx,cx-44,cy+6,88,68,11); bctx.fill(); bctx.stroke();
  bctx.strokeStyle=`rgba(${r},${g},${b},0.12)`; bctx.lineWidth=1;
  for(let i=0;i<3;i++){bctx.beginPath();bctx.moveTo(cx-28,cy+18+i*16);bctx.lineTo(cx+28,cy+18+i*16);bctx.stroke();}
  const pp=0.4+Math.sin(animT*2)*0.6;
  bctx.fillStyle=`rgba(${Math.round(r*pp)},${Math.round(g*pp)},${Math.round(b*pp)},1)`;
  bctx.beginPath();bctx.arc(cx,cy+40,8,0,Math.PI*2);bctx.fill();
  bctx.fillStyle='rgba(255,255,255,0.85)';bctx.beginPath();bctx.arc(cx-2,cy+38,2,0,Math.PI*2);bctx.fill();

  // arms
  bctx.strokeStyle=col; bctx.lineWidth=7; bctx.lineCap='round';
  if(emo==='happy'){const w2=Math.round(Math.sin(animT*3)*12);bctx.beginPath();bctx.moveTo(cx-44,cy+16);bctx.lineTo(cx-76,cy-4+w2);bctx.stroke();bctx.beginPath();bctx.moveTo(cx+44,cy+16);bctx.lineTo(cx+76,cy-4-w2);bctx.stroke();bctx.fillStyle=col;bctx.beginPath();bctx.arc(cx-76,cy-4+w2,7,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(cx+76,cy-4-w2,7,0,Math.PI*2);bctx.fill();}
  else if(emo==='sad'){bctx.beginPath();bctx.moveTo(cx-44,cy+20);bctx.lineTo(cx-70,cy+66);bctx.stroke();bctx.beginPath();bctx.moveTo(cx+44,cy+20);bctx.lineTo(cx+70,cy+66);bctx.stroke();bctx.fillStyle=col;bctx.beginPath();bctx.arc(cx-70,cy+66,6,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(cx+70,cy+66,6,0,Math.PI*2);bctx.fill();}
  else if(emo==='angry'){const pu=Math.round(Math.sin(animT*4)*6);bctx.beginPath();bctx.moveTo(cx-44,cy+18);bctx.lineTo(cx-80+pu,cy+10);bctx.stroke();bctx.beginPath();bctx.moveTo(cx+44,cy+18);bctx.lineTo(cx+80-pu,cy+10);bctx.stroke();bctx.fillStyle=col;bctx.beginPath();bctx.arc(cx-80+pu,cy+10,8,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(cx+80-pu,cy+10,8,0,Math.PI*2);bctx.fill();}
  else if(emo==='fear'){const tr=Math.round(Math.sin(animT*9)*3);bctx.beginPath();bctx.moveTo(cx-44,cy+18);bctx.lineTo(cx-70+tr,cy+38);bctx.stroke();bctx.beginPath();bctx.moveTo(cx+44,cy+18);bctx.lineTo(cx+70+tr,cy+38);bctx.stroke();bctx.fillStyle=col;bctx.beginPath();bctx.arc(cx-70+tr,cy+38,6,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(cx+70+tr,cy+38,6,0,Math.PI*2);bctx.fill();}
  else{bctx.beginPath();bctx.moveTo(cx-44,cy+18);bctx.lineTo(cx-74,cy+34);bctx.stroke();bctx.beginPath();bctx.moveTo(cx+44,cy+18);bctx.lineTo(cx+74,cy+34);bctx.stroke();bctx.fillStyle=col;bctx.beginPath();bctx.arc(cx-74,cy+34,6,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(cx+74,cy+34,6,0,Math.PI*2);bctx.fill();}

  // neck + head
  bctx.fillStyle='#111'; bctx.strokeStyle=col; bctx.lineWidth=1.5;
  rrect(bctx,cx-10,cy-7,20,14,4); bctx.fill(); bctx.stroke();
  bctx.fillStyle='#0d0d22'; bctx.strokeStyle=col; bctx.lineWidth=2.5;
  rrect(bctx,cx-55,cy-70,110,80,15); bctx.fill(); bctx.stroke();
  const sy2=cy-70+Math.round((animT*22)%80);
  bctx.fillStyle=`rgba(${r},${g},${b},0.04)`; bctx.fillRect(cx-55,sy2,110,5);
  bctx.strokeStyle=col; bctx.lineWidth=2;
  bctx.beginPath();bctx.moveTo(cx,cy-70);bctx.lineTo(cx,cy-92);bctx.stroke();
  const ap=0.4+Math.sin(animT*2.5)*0.6;
  bctx.fillStyle=`rgba(${r},${g},${b},${ap})`;bctx.beginPath();bctx.arc(cx,cy-96,6,0,Math.PI*2);bctx.fill();
  bctx.fillStyle='white';bctx.beginPath();bctx.arc(cx-1,cy-98,1.8,0,Math.PI*2);bctx.fill();

  // eyes
  const ey=cy-39, el=cx-18, er=cx+18;
  if(emo==='happy'){bctx.strokeStyle=col;bctx.lineWidth=2.5;bctx.beginPath();bctx.arc(el,ey+3,12,Math.PI,0);bctx.stroke();bctx.beginPath();bctx.arc(er,ey+3,12,Math.PI,0);bctx.stroke();}
  else if(emo==='sad'){bctx.strokeStyle=col;bctx.lineWidth=2.5;bctx.beginPath();bctx.arc(el,ey-2,12,0,Math.PI);bctx.stroke();bctx.beginPath();bctx.arc(er,ey-2,12,0,Math.PI);bctx.stroke();bctx.lineWidth=2;bctx.beginPath();bctx.moveTo(el-12,ey-16);bctx.lineTo(el+6,ey-10);bctx.stroke();bctx.beginPath();bctx.moveTo(er-6,ey-10);bctx.lineTo(er+12,ey-16);bctx.stroke();}
  else if(emo==='angry'){bctx.fillStyle=col;bctx.beginPath();bctx.ellipse(el,ey,13,7,0.28,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.ellipse(er,ey,13,7,-0.28,0,Math.PI*2);bctx.fill();bctx.fillStyle='#06060f';bctx.beginPath();bctx.arc(el,ey,5,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er,ey,5,0,Math.PI*2);bctx.fill();bctx.strokeStyle=col;bctx.lineWidth=2.5;bctx.beginPath();bctx.moveTo(el-16,ey-14);bctx.lineTo(el+7,ey-7);bctx.stroke();bctx.beginPath();bctx.moveTo(er+16,ey-14);bctx.lineTo(er-7,ey-7);bctx.stroke();}
  else if(emo==='surprise'){bctx.fillStyle=col;bctx.beginPath();bctx.arc(el,ey,14,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er,ey,14,0,Math.PI*2);bctx.fill();bctx.fillStyle='#06060f';bctx.beginPath();bctx.arc(el,ey,7,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er,ey,7,0,Math.PI*2);bctx.fill();bctx.fillStyle='white';bctx.beginPath();bctx.arc(el-3,ey-3,3,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er-3,ey-3,3,0,Math.PI*2);bctx.fill();}
  else if(emo==='fear'){const tr=Math.round(Math.sin(animT*10)*1.5);bctx.fillStyle=col;bctx.beginPath();bctx.arc(el+tr,ey,14,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er+tr,ey,14,0,Math.PI*2);bctx.fill();bctx.fillStyle='#06060f';bctx.beginPath();bctx.arc(el+tr+2,ey+2,6,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er+tr-2,ey+2,6,0,Math.PI*2);bctx.fill();}
  else if(emo==='disgust'){bctx.fillStyle=col;bctx.beginPath();bctx.ellipse(el,ey,13,5,0,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.ellipse(er,ey,13,5,0,0,Math.PI*2);bctx.fill();bctx.strokeStyle=col;bctx.lineWidth=2;bctx.beginPath();bctx.moveTo(el-12,ey-13);bctx.lineTo(el+9,ey-8);bctx.stroke();bctx.beginPath();bctx.moveTo(er-9,ey-10);bctx.lineTo(er+11,ey-15);bctx.stroke();}
  else{bctx.fillStyle=col;bctx.beginPath();bctx.arc(el,ey,12,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er,ey,12,0,Math.PI*2);bctx.fill();bctx.fillStyle='#06060f';bctx.beginPath();bctx.arc(el,ey,5,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er,ey,5,0,Math.PI*2);bctx.fill();bctx.fillStyle='white';bctx.beginPath();bctx.arc(el-2,ey-2,2.5,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(er-2,ey-2,2.5,0,Math.PI*2);bctx.fill();}

  // mouth
  const my=cy-14; bctx.strokeStyle=col; bctx.lineWidth=2.5; bctx.lineCap='round';
  if(emo==='happy'){bctx.beginPath();bctx.moveTo(cx-20,my);bctx.quadraticCurveTo(cx,my+17,cx+20,my);bctx.stroke();}
  else if(emo==='sad'){bctx.beginPath();bctx.moveTo(cx-18,my+12);bctx.quadraticCurveTo(cx,my+3,cx+18,my+12);bctx.stroke();}
  else if(emo==='angry'){bctx.beginPath();bctx.moveTo(cx-18,my+9);bctx.quadraticCurveTo(cx,my+3,cx+18,my+9);bctx.stroke();}
  else if(emo==='surprise'){bctx.fillStyle=`rgba(${r},${g},${b},0.5)`;bctx.beginPath();bctx.ellipse(cx,my+8,12,10,0,0,Math.PI*2);bctx.fill();bctx.stroke();}
  else if(emo==='fear'){const tr=Math.round(Math.sin(animT*9)*1.5);bctx.beginPath();bctx.moveTo(cx-16+tr,my+7);bctx.quadraticCurveTo(cx,my+14,cx+16+tr,my+7);bctx.stroke();}
  else if(emo==='disgust'){bctx.beginPath();bctx.moveTo(cx-18,my+9);bctx.lineTo(cx+2,my+9);bctx.stroke();bctx.beginPath();bctx.moveTo(cx+2,my+9);bctx.lineTo(cx+18,my+4);bctx.stroke();}
  else{bctx.beginPath();bctx.moveTo(cx-16,my+9);bctx.lineTo(cx+16,my+9);bctx.stroke();}

  // tears, particles, extras
  if(emo==='sad'){if(Math.random()<0.05){tears.push({x:el,y:ey+9,vy:1.5,life:1});tears.push({x:er,y:ey+9,vy:1.5,life:1});}}else tears=[];
  tears=tears.filter(t=>{t.y+=t.vy;t.vy+=0.08;t.life-=0.018;if(t.life>0&&t.y<BH){const[r2,g2,b2]=h2r('#5577ff');bctx.fillStyle=`rgba(${r2},${g2},${b2},${t.life*0.8})`;bctx.beginPath();bctx.ellipse(t.x,t.y,2,3.5,0,0,Math.PI*2);bctx.fill();return true;}return false;});
  if(emo==='happy'&&Math.random()<0.14){for(let i=0;i<4;i++)particles.push({x:cx+(Math.random()-0.5)*110,y:cy+(Math.random()-0.5)*70,vx:(Math.random()-0.5)*2.8,vy:-1.8-Math.random()*2.2,life:1,decay:0.024+Math.random()*0.02,sz:2+Math.random()*3.5});}
  if(emo!=='happy')particles=[];
  particles=particles.filter(p=>{p.x+=p.vx;p.y+=p.vy;p.vy+=0.06;p.life-=p.decay;if(p.life>0&&p.x>=0&&p.x<BW&&p.y>=0&&p.y<BH){bctx.fillStyle=`rgba(${r},${g},${b},${p.life})`;bctx.beginPath();bctx.arc(p.x,p.y,p.sz,0,Math.PI*2);bctx.fill();return true;}return false;});
  if(emo==='fear'){const sw=Math.round(Math.sin(animT*2)*3);bctx.fillStyle='rgba(187,68,255,0.5)';bctx.beginPath();bctx.arc(cx+58,cy-44+sw,3.5,0,Math.PI*2);bctx.fill();bctx.beginPath();bctx.arc(cx+64,cy-32+sw,3,0,Math.PI*2);bctx.fill();}
  if(emo==='angry'){bctx.strokeStyle='rgba(255,50,70,0.6)';bctx.lineWidth=1.8;[[cx-72,cx-65],[cx-68,cx-61]].forEach(([x1,x2])=>{bctx.beginPath();bctx.moveTo(x1+shakeX,cy-40);bctx.lineTo(x2+shakeX,cy-50);bctx.stroke();});[[cx+65,cx+72],[cx+61,cx+68]].forEach(([x1,x2])=>{bctx.beginPath();bctx.moveTo(x1+shakeX,cy-40);bctx.lineTo(x2+shakeX,cy-50);bctx.stroke();});}
}

// ── UI updates ──
let detCount=0, emoChanges=0, confSum=0, emoFreq={}, hist=[], tlHist=[];
let sessStart=Date.now(), lastEmo='neutral';

function updateAll(emo, conf, scores, voice){
  if(emo!==lastEmo){emoChanges++;lastEmo=emo;}
  curEmo=emo; detCount++; confSum+=conf; emoFreq[emo]=(emoFreq[emo]||0)+1;
  const cfg=ECFG[emo]||ECFG.neutral;

  // bot info
  const ne=document.getElementById('emoName');
  ne.textContent=emo.toUpperCase(); ne.style.color=cfg.color;
  ne.style.textShadow=`0 0 30px ${cfg.color}55`;
  ne.classList.add('glitch'); setTimeout(()=>ne.classList.remove('glitch'),100);
  document.getElementById('emoResp').textContent=cfg.resp;
  const ap=document.getElementById('emoAction');
  ap.textContent='BOT: '+cfg.action; ap.style.borderColor=cfg.color+'25'; ap.style.color=cfg.color;

  // history chips
  hist.push(emo); if(hist.length>8) hist.shift();
  document.getElementById('histRow').innerHTML=hist.map((e,i)=>{
    const c=ECFG[e]?.color||'#888';
    return`<span class="hchip" style="color:${c};border-color:${c};opacity:${0.2+(i/hist.length)*0.8}">${e}</span>`;
  }).join('');

  // timeline
  tlHist.push({emo,conf}); if(tlHist.length>45) tlHist.shift();
  document.getElementById('tlWrap').innerHTML=tlHist.map(({emo:e2,conf:c2})=>{
    const c=ECFG[e2]?.color||'#333';
    return`<div class="tlbar" style="height:${Math.round(4+(c2/100)*34)}px;background:${c};opacity:0.65;min-width:8px"></div>`;
  }).join('');

  // bars
  document.getElementById('barsWrap').innerHTML=Object.entries(ECFG).map(([e,c2])=>{
    const isA=e===emo, pct=isA?Math.round(conf):scores[e]?Math.round(scores[e]):Math.round(Math.random()*10);
    return`<div class="bar-row"><span class="blbl" style="color:${isA?c2.color:'#2a2a3a'}">${e}</span><div class="btrack"><div class="bfill" style="width:${pct}%;background:${c2.color}${isA?'':'44'}"></div></div><span class="bval">${pct}%</span></div>`;
  }).join('');
  const cb=document.getElementById('confBig');
  cb.textContent=Math.round(conf)+'%'; cb.style.color=cfg.color; cb.style.textShadow=`0 0 20px ${cfg.color}44`;

  // stats
  const top=Object.entries(emoFreq).sort((a,b)=>b[1]-a[1])[0];
  document.getElementById('sCnt').textContent=detCount;
  const st=document.getElementById('sTop'); st.textContent=(top?top[0].slice(0,5):'----').toUpperCase(); st.style.color=top?ECFG[top[0]]?.color:'#aaa';
  document.getElementById('sChg').textContent=emoChanges;
  document.getElementById('sConf').textContent=Math.round(confSum/detCount)+'%';

  // voice
  const ve=document.getElementById('vEmo'); ve.textContent=voice.toUpperCase(); ve.style.color=ECFG[voice]?.color||'#ff7700';

  // face box (if region available — placeholder here)
  drawFaceBox(0,0,0,0,emo);
}

function updateWave(){
  document.getElementById('wave').innerHTML=
    Array.from({length:16},()=>`<div class="wbar" style="height:${Math.round(3+Math.random()*20)}px"></div>`).join('');
}

function loop(){
  drawBot(curEmo);
  const e=Math.floor((Date.now()-sessStart)/1000);
  document.getElementById('sessT').textContent=String(Math.floor(e/60)).padStart(2,'0')+':'+String(e%60).padStart(2,'0');
  document.getElementById('clk').textContent=new Date().toTimeString().slice(0,8);
  requestAnimationFrame(loop);
}

// init
document.getElementById('barsWrap').innerHTML=Object.entries(ECFG).map(([e,c])=>
  `<div class="bar-row"><span class="blbl" style="color:#2a2a3a">${e}</span><div class="btrack"><div class="bfill" style="width:0%;background:${c.color}44"></div></div><span class="bval">0%</span></div>`
).join('');

setInterval(updateWave, 130);
loop();
</script>
</body>
</html>'''

if __name__ == '__main__':
    print("Starting voice detection...")
    start_voice_detection()
    print("Cyberpunk Emotion Bot: http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)
