import numpy as np
import sys
import os
import time
import threading
import urllib.request
import cv2
import mediapipe as mp
from collections import deque
from threading import Thread
from tf_keras.models import load_model
import tensorflow as tf
import json

sys.path.append("..")

_BASE          = os.path.join(os.path.dirname(__file__), '..')
_MP_MODEL_PATH = os.path.join(_BASE, 'models', 'hand_landmarker.task')
_MP_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
                  'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
_LM_MODEL_PATH = os.path.join(_BASE, 'models', 'landmark_model.keras')
_LM_LABELS_PATH= os.path.join(_BASE, 'models', 'landmark_labels.json')

_HandLandmarker        = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_BaseOptions           = mp.tasks.BaseOptions
_RunningMode           = mp.tasks.vision.RunningMode

# Populated at runtime when landmark model is loaded
LABEL_MAP = {}


def extract_landmark_features(hand_lms):
    """63-value normalised feature vector from 21 MediaPipe landmarks."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
    pts -= pts[0]                                       # centre on wrist
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts  /= scale
    return pts.flatten()

# ── Skeleton ───────────────────────────────────────────────────────────────────
_SKELETON = [
    (0,  1,  (200,200,200)), (0,  5,  (200,200,200)), (5,  9,  (200,200,200)),
    (9,  13, (200,200,200)), (13, 17, (200,200,200)), (0,  17, (200,200,200)),
    (1, 2, (255,200, 30)), (2, 3, (255,200, 30)), (3, 4, (255,200, 30)),
    (5, 6, ( 50,220,100)), (6, 7, ( 50,220,100)), (7, 8, ( 50,220,100)),
    (9,  10,(  0,200,255)), (10,11,(  0,200,255)), (11,12,(  0,200,255)),
    (13, 14,(255,130,  0)), (14,15,(255,130,  0)), (15,16,(255,130,  0)),
    (17, 18,(220, 50,200)), (18,19,(220, 50,200)), (19,20,(220, 50,200)),
]
_FINGERTIPS = {4, 8, 12, 16, 20}


# ── Async landmark predictor ───────────────────────────────────────────────────
class AsyncPredictor:
    """
    Background-thread inference with temporal probability smoothing.
    Uses only the landmark model (models/landmark_model.keras).
    Call ready() to check if a model is loaded before using predictions.
    """

    def __init__(self, smooth_frames=12, min_confidence=0.60):
        self._min_conf  = min_confidence
        self._prob_buf  = deque(maxlen=smooth_frames)

        self._lm_model  = None
        self._lm_labels = None
        self._load_landmark_model()

        self._pending  = None
        self._result   = (None, 0.0)
        self._lock     = threading.Lock()
        self._event    = threading.Event()

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _load_landmark_model(self):
        global LABEL_MAP
        if os.path.exists(_LM_MODEL_PATH) and os.path.exists(_LM_LABELS_PATH):
            try:
                self._lm_model  = load_model(_LM_MODEL_PATH)
                with open(_LM_LABELS_PATH, encoding='utf-8') as f:
                    self._lm_labels = json.load(f)
                LABEL_MAP = {i: w for i, w in enumerate(self._lm_labels)}
                print(f'>  Landmark model loaded — {len(self._lm_labels)} words: '
                      f'{", ".join(self._lm_labels)}')
            except Exception as e:
                print(f'[Predictor] Could not load landmark model: {e}')

    def ready(self):
        """True if a landmark model is loaded and predictions are available."""
        return self._lm_model is not None

    def submit_landmarks(self, feat_63):
        """Submit a 63-value landmark feature vector for inference."""
        if self._lm_model is None:
            return
        with self._lock:
            self._pending = feat_63.copy()
        self._event.set()

    def get_result(self):
        with self._lock:
            return self._result

    def reset(self):
        with self._lock:
            self._prob_buf.clear()
            self._result = (None, 0.0)

    def _worker(self):
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                data = self._pending
            if data is None or self._lm_model is None:
                continue

            try:
                feat  = tf.constant(data.reshape(1, -1))
                probs = self._lm_model(feat, training=False).numpy()[0]
            except Exception:
                continue

            with self._lock:
                self._prob_buf.append(probs)
                smoothed   = np.mean(self._prob_buf, axis=0)
                idx        = int(np.argmax(smoothed))
                confidence = float(smoothed[idx])
                name = self._lm_labels[idx] if idx < len(self._lm_labels) else None
                self._result = (name, confidence) if confidence >= self._min_conf \
                               else (None, confidence)


def _ensure_model():
    path = os.path.abspath(_MP_MODEL_PATH)
    if not os.path.exists(path):
        print('> Downloading hand_landmarker.task (~7 MB)...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(_MP_MODEL_URL, path)
        print(f'>  Done: {path}')
    return path


def load_inference_graph():
    print('> ====== loading MediaPipe HandLandmarker')
    model_path = _ensure_model()
    options = _HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = _HandLandmarker.create_from_options(options)
    print('>  ====== MediaPipe HandLandmarker loaded.')
    return landmarker, None


def detect_objects(image_np, landmarker, _sess):
    ih, iw    = image_np.shape[:2]
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    timestamp = int(time.monotonic() * 1000)
    result    = landmarker.detect_for_video(mp_image, timestamp)

    if not result.hand_landmarks:
        return np.array([[0.,0.,0.,0.]]), np.array([0.]), [], []

    boxes, scores, landmarks_px, raw_lms = [], [], [], []
    for i, hand_lms in enumerate(result.hand_landmarks):
        confidence = 1.0
        if result.handedness and i < len(result.handedness):
            confidence = result.handedness[i][0].score

        xs = [lm.x for lm in hand_lms]
        ys = [lm.y for lm in hand_lms]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_min = max(0., x_min - 0.05);  x_max = min(1., x_max + 0.05)
        y_min = max(0., y_min - 0.08);  y_max = min(1., y_max + 0.08)

        if (x_max - x_min) < 0.02 or (y_max - y_min) < 0.02:
            continue

        boxes.append([y_min, x_min, y_max, x_max])
        scores.append(confidence)
        landmarks_px.append([(int(lm.x * iw), int(lm.y * ih)) for lm in hand_lms])
        raw_lms.append(hand_lms)

    if not boxes:
        return np.array([[0.,0.,0.,0.]]), np.array([0.]), [], []

    return np.array(boxes), np.array(scores), landmarks_px, raw_lms


def draw_skeleton(image_np, landmarks_px):
    """Draw colour-coded finger skeleton on an RGB image (in-place)."""
    for hand_pts in landmarks_px:
        if len(hand_pts) < 21:
            continue
        for a, b, color in _SKELETON:
            cv2.line(image_np, hand_pts[a], hand_pts[b], color, 2, cv2.LINE_AA)
        for idx, pt in enumerate(hand_pts):
            r = 7 if idx in _FINGERTIPS else 5
            cv2.circle(image_np, pt, r, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(image_np, pt, r, ( 30, 30, 30),  1, cv2.LINE_AA)


def draw_box_on_image(predictor, num_hands_detect, score_thresh, scores, boxes,
                      im_width, im_height, image_np, landmarks_px=None,
                      raw_landmarks=None):
    """
    Draws bounding boxes for all detected hands.
    Submits landmark features from the primary (highest-score) hand for recognition.
    Returns (left, bottom, label_name, confidence) for the primary hand.
    """
    detected = [i for i in range(min(num_hands_detect, len(scores)))
                if scores[i] > score_thresh]

    if not detected:
        predictor.reset()
        return 0, 0, None, 0.0

    result_left, result_bottom = 0, 0
    for i in detected:
        p1 = int(boxes[i][1] * im_width)
        p2 = int(boxes[i][0] * im_height)
        p3 = int(boxes[i][3] * im_width)
        p4 = int(boxes[i][2] * im_height)

        # Different box colours: primary=green, secondary=cyan
        color = (77, 255, 9) if i == detected[0] else (0, 220, 220)
        cv2.rectangle(image_np, (p1, p2), (p3, p4), color, 2, cv2.LINE_AA)

        if i == detected[0]:
            result_left, result_bottom = p1, p4
            if raw_landmarks and i < len(raw_landmarks):
                feat = extract_landmark_features(raw_landmarks[i])
                predictor.submit_landmarks(feat)

    label_name, confidence = predictor.get_result()
    return result_left, result_bottom, label_name, confidence


def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


class WebcamVideoStream:
    def __init__(self, src, width, height):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):  return self.frame
    def size(self):  return self.stream.get(3), self.stream.get(4)
    def stop(self):  self.stopped = True
