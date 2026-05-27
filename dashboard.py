import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, Counter
import csv
import time
import os

# ─── Config ───────────────────────────────────────
CSV_FILE = "emotion_log.csv"
MAX_POINTS = 50

emotion_list = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]
colors_map = {
    "happy":    "#00FF00",
    "sad":      "#6464FF",
    "angry":    "#FF0000",
    "neutral":  "#CCCCCC",
    "surprise": "#00FFFF",
    "fear":     "#AA00FF",
    "disgust":  "#FF8800",
}

# ─── Data buffers ─────────────────────────────────
time_buf = deque(maxlen=MAX_POINTS)
emotion_buf = deque(maxlen=MAX_POINTS)
last_line_count = 0

# ─── Setup figure ─────────────────────────────────
fig = plt.figure(figsize=(14, 7), facecolor='#111111')
fig.suptitle("🧠 Emotion-Aware Bot — Live Dashboard", 
             color='white', fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(2, 2, (1, 2))  # timeline top
ax2 = fig.add_subplot(2, 2, 3)       # pie chart
ax3 = fig.add_subplot(2, 2, 4)       # bar chart

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

def read_new_data():
    global last_line_count
    rows = []
    if not os.path.exists(CSV_FILE):
        return rows
    with open(CSV_FILE, "r") as f:
        all_rows = list(csv.DictReader(f))
        new_rows = all_rows[last_line_count:]
        last_line_count = len(all_rows)
        return new_rows

def animate(i):
    new_rows = read_new_data()
    for row in new_rows:
        try:
            time_buf.append(float(row["timestamp"]))
            emotion_buf.append(row["emotion"])
        except:
            pass

    if not emotion_buf:
        return

    # ── Timeline chart ──
    ax1.clear()
    ax1.set_facecolor('#1a1a1a')
    y_vals = [emotion_list.index(e) if e in emotion_list else 0 for e in emotion_buf]
    point_colors = [colors_map.get(e, "#FFFFFF") for e in emotion_buf]

    ax1.plot(list(time_buf), y_vals, color='#444444', linewidth=1)
    ax1.scatter(list(time_buf), y_vals, c=point_colors, s=40, zorder=5)
    ax1.set_yticks(range(len(emotion_list)))
    ax1.set_yticklabels(emotion_list, color='white', fontsize=9)
    ax1.set_xlabel("Time (seconds)", color='white')
    ax1.set_title("📈 Emotion Timeline", color='white', fontsize=11)
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15)

    # Highlight current emotion
    if emotion_buf:
        current = emotion_buf[-1]
        ax1.axhline(y=emotion_list.index(current),
                   color=colors_map.get(current, 'white'),
                   alpha=0.3, linestyle='--')

    # ── Pie chart ──
    ax2.clear()
    ax2.set_facecolor('#1a1a1a')
    counts = Counter(emotion_buf)
    labels = list(counts.keys())
    sizes = list(counts.values())
    pie_colors = [colors_map.get(l, "#FFFFFF") for l in labels]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=pie_colors,
        autopct='%1.0f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 8}
    )
    for at in autotexts:
        at.set_color('black')
        at.set_fontweight('bold')
    ax2.set_title("🥧 Emotion Distribution", color='white', fontsize=11)

    # ── Bar chart ──
    ax3.clear()
    ax3.set_facecolor('#1a1a1a')
    all_counts = {e: 0 for e in emotion_list}
    all_counts.update(Counter(emotion_buf))

    bars = ax3.bar(
        list(all_counts.keys()),
        list(all_counts.values()),
        color=[colors_map.get(e, "#FFFFFF") for e in all_counts.keys()]
    )
    ax3.set_title("📊 Emotion Frequency", color='white', fontsize=11)
    ax3.tick_params(colors='white')
    ax3.set_xticklabels(list(all_counts.keys()), rotation=30,
                        ha='right', color='white', fontsize=8)

    # Highlight current emotion bar
    if emotion_buf:
        current = emotion_buf[-1]
        for bar, emo in zip(bars, all_counts.keys()):
            if emo == current:
                bar.set_edgecolor('white')
                bar.set_linewidth(2)

    # ── Stats text ──
    if emotion_buf:
        most_common = Counter(emotion_buf).most_common(1)[0][0]
        fig.suptitle(
            f"🧠 Emotion-Aware Bot  |  Current: {emotion_buf[-1].upper()}  "
            f"|  Most Common: {most_common.upper()}  |  Samples: {len(emotion_buf)}",
            color='white', fontsize=11, fontweight='bold'
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

# ─── Run ──────────────────────────────────────────
ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
plt.tight_layout()
plt.show()