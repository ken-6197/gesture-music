"""
Gesture Piano  —  keyboard/organ sound, fixed finger counting
MediaPipe Tasks API  (mediapipe >= 0.10.x)
"""

import cv2
import numpy as np
import pygame
import time
import urllib.request
import sys
from pathlib import Path
from collections import deque

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe import Image, ImageFormat

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

def download_model():
    print("Downloading hand-landmarker model (~8 MB)...", end="", flush=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size = MODEL_PATH.stat().st_size
        if size < 100_000:
            MODEL_PATH.unlink()
            raise RuntimeError(f"File too small ({size} bytes).")
        print(f" done ({size // 1_000_000} MB)")
    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Download manually from:\n  {MODEL_URL}")
        print(f"Place next to this script as: hand_landmarker.task")
        sys.exit(1)

if not MODEL_PATH.exists():
    download_model()
else:
    print(f"Model found: {MODEL_PATH.name}")

# ---------------------------------------------------------------------------
# Keyboard / organ sound  (slower attack than piano, no harsh beat)
# ---------------------------------------------------------------------------
pygame.mixer.init(frequency=44100, size=-16, channels=2)

def create_keyboard_tone(frequency, duration=2.0, volume=0.32):
    """
    Electric-keyboard / organ tone:
      - Short but not instant attack (~40 ms)  -> removes piano 'click'
      - Gentle exponential decay over the full duration  -> sustains
      - Clean harmonics without beating: 2nd harmonic at 0.4, 3rd at 0.15
        (no 4th / 5th which cause the fast tremolo / beating you heard)
    """
    sr   = 44100
    n    = int(sr * duration)
    t    = np.linspace(0, duration, n, endpoint=False)

    # Harmonics
    wave  = 1.00 * np.sin(2 * np.pi * frequency * t)
    wave += 0.40 * np.sin(2 * np.pi * frequency * 2 * t)
    wave += 0.15 * np.sin(2 * np.pi * frequency * 3 * t)

    # Envelope: 40 ms attack ramp, then slow exponential decay (tau = 1.8 s)
    attack_n = int(0.04 * sr)
    envelope = np.exp(-t / 1.8)                      # slow decay
    envelope[:attack_n] *= np.linspace(0, 1, attack_n)  # smooth attack

    wave = wave * envelope * volume
    wave16 = (wave * 32767).clip(-32768, 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.stack([wave16, wave16], axis=1))
    return pygame.sndarray.make_sound(stereo)

NOTE_FREQS  = {0: 261.63, 1: 293.66, 2: 329.63, 3: 349.23, 4: 392.00, 5: 440.00}
NOTE_NAMES  = {0: "C", 1: "D", 2: "E", 3: "F", 4: "G", 5: "A"}
NOTE_COLORS = {
    0: (80,  80,  255),   # blue   — C
    1: (80,  180, 255),   # cyan   — D
    2: (60,  210,  90),   # green  — E
    3: (200, 210,  30),   # yellow — F
    4: (255, 140,  20),   # orange — G
    5: (255,  60,  60),   # red    — A
}

print("Loading keyboard sounds...")
sounds = {n: create_keyboard_tone(f) for n, f in NOTE_FREQS.items()}
for n in range(6):
    print(f"  ok  {NOTE_NAMES[n]}  ({NOTE_FREQS[n]} Hz)")
print("All sounds ready!\n")

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
WRIST     = 0
THUMB_TIP = 4;  THUMB_IP  = 3;  THUMB_MCP = 2
INDEX_MCP = 5;  INDEX_PIP = 6;  INDEX_TIP = 8
MID_MCP   = 9;  MID_PIP   = 10; MID_TIP   = 12
RING_MCP  = 13; RING_PIP  = 14; RING_TIP  = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_TIP = 20

# ---------------------------------------------------------------------------
# Finger counting
# ---------------------------------------------------------------------------
def count_fingers(lms):
    """
    Returns 0-5.

    Four fingers: tip.y < pip.y - margin  (tip clearly above middle joint)
    Thumb:        horizontal spread from index MCP  > threshold
                  AND the thumb tip must be above its own IP joint
                  (prevents a resting / bent thumb from firing)
    """
    def y(i): return lms[i].y
    def x(i): return lms[i].x

    count = 0

    # Four fingers — margin 0.02 prevents borderline bent fingers
    for tip, pip in [(INDEX_TIP, INDEX_PIP),
                     (MID_TIP,   MID_PIP),
                     (RING_TIP,  RING_PIP),
                     (PINKY_TIP, PINKY_PIP)]:
        if y(tip) < y(pip) - 0.02:
            count += 1

    # Thumb — two conditions must BOTH be true:
    #   1. TIP is horizontally far from INDEX_MCP (spread out)
    #   2. TIP is above (lower y) than THUMB_IP  (not curled under)
    thumb_spread = abs(x(THUMB_TIP) - x(INDEX_MCP))
    thumb_up     = y(THUMB_TIP) < y(THUMB_IP)        # tip above IP joint

    if thumb_spread > 0.07 and thumb_up:
        count += 1

    return min(count, 5)

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
lm_options = HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.55,
)
landmarker = HandLandmarker.create_from_options(lm_options)

# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand(frame, lms, color=(0, 230, 100)):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        r = 6 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (px, py), r, color, -1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=" * 55)
print("GESTURE PIANO")
for i in range(6):
    print(f"  {i} finger{'s' if i != 1 else ' '} -> {NOTE_NAMES[i]}")
print("Press Q to quit")
print("=" * 55)

finger_history = deque(maxlen=7)
last_played    = None
last_play_time = 0.0
PLAY_DELAY     = 0.6        # seconds before re-triggering same note
current_color  = (130, 130, 130)
frame_ts_ms    = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    frame_ts_ms += 30

    result        = landmarker.detect_for_video(mp_image, frame_ts_ms)
    hand_detected = bool(result.hand_landmarks)
    finger_count  = 0

    if hand_detected:
        lms = result.hand_landmarks[0]
        draw_hand(frame, lms)

        raw = count_fingers(lms)
        finger_history.append(raw)
        finger_count = max(set(finger_history), key=finger_history.count)

        # Label above wrist — only after history stabilises
        if len(finger_history) >= 3:
            wx = int(lms[WRIST].x * frame.shape[1])
            wy = int(lms[WRIST].y * frame.shape[0])
            label = f"{finger_count} finger{'s' if finger_count != 1 else ''}"
            cv2.putText(frame, label, (wx - 50, wy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        NOTE_COLORS.get(finger_count, (200, 200, 200)),
                        2, cv2.LINE_AA)

    # --- Sound ---
    now = time.time()
    if hand_detected and finger_count in sounds:
        current_color = NOTE_COLORS[finger_count]
        if finger_count != last_played:
            sounds[finger_count].play()
            last_played    = finger_count
            last_play_time = now
            print(f"  {finger_count} -> {NOTE_NAMES[finger_count]}")
        elif (now - last_play_time) > PLAY_DELAY:
            sounds[finger_count].play()
            last_play_time = now
        # Status text — ASCII only, no dashes or special chars
        status = f"PLAYING: {finger_count} finger{'s' if finger_count != 1 else ''}  [{NOTE_NAMES[finger_count]}]"
    else:
        status        = "Show your hand"
        last_played   = None
        current_color = (130, 130, 130)

    # --- HUD panel ---
    ov = frame.copy()
    cv2.rectangle(ov, (10, 10), (420, 75), (0, 0, 0), -1)
    frame = cv2.addWeighted(ov, 0.55, frame, 0.45, 0)
    cv2.putText(frame, status, (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, current_color, 2, cv2.LINE_AA)

    # --- Virtual keyboard ---
    kw  = 80
    kx0 = (frame.shape[1] - 6 * kw) // 2
    ky  = frame.shape[0] - 80
    for i in range(6):
        kx  = kx0 + i * kw
        active = hand_detected and finger_count == i
        col = NOTE_COLORS[i] if active else (50, 50, 50)
        if active:
            cv2.rectangle(frame, (kx-3, ky-3), (kx+kw-2, ky+53), (255, 255, 255), 2)
        cv2.rectangle(frame, (kx, ky), (kx+kw-5, ky+50), col, -1)
        cv2.rectangle(frame, (kx, ky), (kx+kw-5, ky+50), (140, 140, 140), 1)
        cv2.putText(frame, NOTE_NAMES[i],
                    (kx + kw//2 - 10, ky + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(i),
                    (kx + kw//2 - 7, ky + 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Down-arrow above active key
    if hand_detected and 0 <= finger_count <= 5:
        ax = kx0 + finger_count * kw + kw//2 - 8
        cv2.putText(frame, "v", (ax, ky - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 120), 2, cv2.LINE_AA)

    cv2.putText(frame, "Q quit",
                (frame.shape[1] - 85, frame.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.imshow("Gesture Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("Thanks for playing!")