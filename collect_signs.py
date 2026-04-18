"""
collect_signs.py — Record landmark training data for any new Bangla sign word.

Usage:
    python collect_signs.py --word আপনি --samples 120
    python collect_signs.py --word ধন্যবাদ --samples 120

Controls while recording:
    SPACE  = start / stop recording this word
    Q      = quit without saving
"""

import os, sys, argparse, json, time
import numpy as np
import cv2
import mediapipe as mp

_MODEL_PATH = os.path.join('models', 'hand_landmarker.task')
_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
               'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')

_HandLandmarker        = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_BaseOptions           = mp.tasks.BaseOptions
_RunningMode           = mp.tasks.vision.RunningMode

SAVE_DIR = os.path.join('dataset', 'landmarks')
os.makedirs(SAVE_DIR, exist_ok=True)


def ensure_model():
    if not os.path.exists(_MODEL_PATH):
        import urllib.request
        print('Downloading hand_landmarker.task ...')
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def extract_features(hand_lms):
    """
    Return a 63-value normalised feature vector from 21 MediaPipe landmarks.
    Centred on wrist, scaled by hand span — position & size invariant.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
    pts -= pts[0]                                          # centre on wrist
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts /= scale
    return pts.flatten()                                   # shape (63,)


def draw_skeleton(frame, hand_pts, color=(0, 255, 100)):
    CONNECTIONS = [
        (0,1),(0,5),(5,9),(9,13),(13,17),(0,17),
        (1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
        (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),
        (17,18),(18,19),(19,20),
    ]
    for a, b in CONNECTIONS:
        cv2.line(frame, hand_pts[a], hand_pts[b], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(hand_pts):
        r = 7 if i in {4,8,12,16,20} else 5
        cv2.circle(frame, pt, r, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, ( 40, 40, 40),  1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word',    required=True, help='Bangla word to record')
    parser.add_argument('--samples', type=int, default=120,
                        help='Number of samples to collect (default 120)')
    args = parser.parse_args()

    word     = args.word.strip()
    n_target = args.samples
    save_path = os.path.join(SAVE_DIR, f'{word}.npy')

    # Load existing samples so we can append
    existing = []
    if os.path.exists(save_path):
        existing = list(np.load(save_path, allow_pickle=True))
        print(f'  Found {len(existing)} existing samples for "{word}"')

    options = _HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=ensure_model()),
        running_mode=_RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = _HandLandmarker.create_from_options(options)
    cap        = cv2.VideoCapture(0)
    cv2.namedWindow('Collect Signs', cv2.WINDOW_NORMAL)

    recording  = False
    collected  = []
    ts         = 0

    print(f'\n  Word: "{word}"  |  Target: {n_target} samples')
    print('  Press SPACE to start recording, SPACE again to stop, Q to quit.\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        ts += 33
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        rgb.flags.writeable = True

        h, w = frame.shape[:2]
        hand_pts = None
        feat     = None

        if result.hand_landmarks:
            lms      = result.hand_landmarks[0]
            hand_pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
            feat     = extract_features(lms)
            col      = (0,200,80) if recording else (200,200,200)
            draw_skeleton(frame, hand_pts, col)

        # Record
        if recording and feat is not None:
            collected.append(feat)

        # HUD
        total = len(existing) + len(collected)
        pct   = min(total / n_target, 1.0)
        bar_w = int(w * 0.6 * pct)

        cv2.rectangle(frame, (20, h-50), (20+int(w*0.6), h-30), (50,50,50), -1)
        cv2.rectangle(frame, (20, h-50), (20+bar_w,       h-30),
                      (0,200,80) if recording else (100,100,200), -1)
        cv2.putText(frame, f'{word}  {total}/{n_target}',
                    (20, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        status = 'RECORDING...' if recording else 'SPACE=record  Q=quit'
        color  = (0,80,255) if recording else (200,200,200)
        cv2.putText(frame, status, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        if total >= n_target:
            cv2.putText(frame, 'DONE! Press Q to save.',
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,150), 2)

        cv2.imshow('Collect Signs', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = not recording
            if recording:
                print(f'  Recording started...')
            else:
                print(f'  Paused — {len(collected)} new samples so far')

        elif key == ord('q') or total >= n_target:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    all_samples = existing + collected
    if all_samples:
        np.save(save_path, np.array(all_samples))
        print(f'\n  Saved {len(all_samples)} samples → {save_path}')
        print(f'  Run  python train_landmark_model.py  to retrain the model.')
    else:
        print('\n  No samples collected.')


if __name__ == '__main__':
    main()
