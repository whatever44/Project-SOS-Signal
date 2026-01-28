import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
import time
import csv
import threading

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

# ================== LOAD MODELS ==================

svm_model = joblib.load("stage1_linear_svm_trigger.pkl")
dl_model = tf.keras.models.load_model("stage2_sos_private.h5")

# ================== MEDIAPIPE ==================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ================== FASTAPI ==================

app = FastAPI()

camera = cv2.VideoCapture(0)

sequence_buffer = []
SOS_FLAG = False
LOCK = threading.Lock()

# ================== METRICS LOGGING ==================

confidence_log = []
time_log = []
frame_log = []
frame_count = 0

prev_time = time.time()
fps = 0

# ================== FEATURE FUNCTIONS ==================

def extract_ratio_features(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    def dist(a, b):
        return np.linalg.norm(pts[a][:2] - pts[b][:2])

    palm = dist(0, 9) + 1e-6

    features = [
        dist(4, 8) / palm,
        dist(8, 12) / palm,
        dist(12, 16) / palm,
        dist(16, 20) / palm,
        dist(4, 20) / palm,
        dist(0, 8) / palm,
        dist(0, 12) / palm,
        dist(0, 16) / palm,
        dist(0, 20) / palm,
        dist(5, 17) / palm
    ]
    return np.array(features)


def landmarks_to_vector(landmarks):
    vec = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    if vec.shape[0] == 63:
        vec = np.pad(vec, (0, 63), 'constant')  # pad to 126
    return vec


# ================== VIDEO STREAM ==================

def gen_frames():
    global sequence_buffer, SOS_FLAG, frame_count, fps, prev_time

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        trigger = False

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # ---- STAGE 1 : SVM TRIGGER ----
            ratio_feat = extract_ratio_features(hand.landmark).reshape(1, -1)
            trigger = svm_model.predict(ratio_feat)[0] == 1

            # ---- STAGE 2 BUFFERING ----
            if trigger:
                seq_feat = landmarks_to_vector(hand.landmark)
                with LOCK:
                    sequence_buffer.append(seq_feat)

                if len(sequence_buffer) == 90:
                    seq = np.array(sequence_buffer).reshape(1, 90, 126)

                    start = time.time()
                    prob = dl_model.predict(seq, verbose=0)[0][0]
                    infer_time = time.time() - start

                    SOS_FLAG = prob > 0.5

                    confidence_log.append(prob)
                    time_log.append(infer_time)
                    frame_log.append(frame_count)

                    sequence_buffer = []

        # ---- FPS CALCULATION ----
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # ---- UI OVERLAY ----
        color = (0, 0, 255) if SOS_FLAG else (0, 255, 0)
        text = "SOS DETECTED" if SOS_FLAG else "Monitoring..."

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# ================== API ROUTES ==================

@app.get("/")
def root():
    return {"status": "SOS Gesture Detection Running"}


@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/sos_status")
def sos_status():
    return JSONResponse({"sos": bool(SOS_FLAG)})


@app.get("/download_metrics")
def download_metrics():
    with open("metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "confidence", "inference_time_sec"])
        for fno, c, t in zip(frame_log, confidence_log, time_log):
            writer.writerow([fno, c, t])

    return {"status": "metrics saved to metrics.csv"}

