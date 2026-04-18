import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import detector_utils as detector_utils
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import datetime, argparse, threading, tempfile, json, glob
from collections import deque
# ── TTS ────────────────────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init()
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ── tkinter for word-name popup ────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False

detection_graph, sess = detector_utils.load_inference_graph()
predictor          = detector_utils.AsyncPredictor(smooth_frames=20, min_confidence=0.75)

CONFIRM_FRAMES   = 45          # ~1.5 s at 30 fps — hold sign longer before adding
COOLDOWN_FRAMES  = 40          # pause between words
MIN_CONFIRM_CONF = 0.75        # higher confidence required
PANEL_W          = 320
RECORD_TARGET    = 400          # samples per new word
LANDMARK_DIR     = os.path.join('dataset', 'landmarks')
os.makedirs(LANDMARK_DIR, exist_ok=True)

# colours (RGB)
BG      = (18, 18, 28);   BG_CARD = (28, 28, 45)
YELLOW  = (255,215, 50);  GREEN   = ( 55,220,110)
BLUE    = (100,150,255);  ORANGE  = (255,150, 50)
RED     = (255, 80, 80);  WHITE   = (230,230,235)
GRAY    = (110,110,125);  DIVIDER = ( 45, 45, 65)
REC_RED = (200, 40, 40)

# ── voice ──────────────────────────────────────────────────────────────────────
_speaking   = False
_speak_lock = threading.Lock()

def speak(text):
    if not TTS_AVAILABLE or not text.strip():
        return
    def _run():
        global _speaking
        with _speak_lock:
            _speaking = True
            try:
                tts = gTTS(text=text, lang='bn', slow=False)
                tmp = tempfile.mktemp(suffix='.mp3')
                tts.save(tmp); pygame.mixer.music.load(tmp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(80)
                pygame.mixer.music.unload(); os.remove(tmp)
            except Exception as e:
                print(f'[TTS] {e}')
            finally:
                _speaking = False
    threading.Thread(target=_run, daemon=True).start()


# ── training thread ────────────────────────────────────────────────────────────
_training       = False
_train_result   = ''   # status message after training

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _run_training():
    global _training, _train_result, predictor
    _training = True
    _train_result = 'প্রশিক্ষণ চলছে...'
    try:
        import subprocess, sys
        res = subprocess.run(
            [sys.executable, 'train_landmark_model.py'],
            cwd=_SCRIPT_DIR,          # always run from project root
            text=True)                # output goes to console — visible to user
        if res.returncode == 0:
            new_pred = detector_utils.AsyncPredictor(
                smooth_frames=20, min_confidence=0.75)
            predictor = new_pred
            words = list(detector_utils.LABEL_MAP.values())
            _train_result = f'লোড হয়েছে: {", ".join(words)}'
        else:
            _train_result = f'ত্রুটি (কোড {res.returncode}) — কনসোল দেখুন'
    except Exception as e:
        _train_result = f'ত্রুটি: {e}'
        print(f'[Training] {e}')
    finally:
        _training = False


def start_training():
    threading.Thread(target=_run_training, daemon=True).start()


# ── word-name popup ────────────────────────────────────────────────────────────
def ask_word_name():
    """Open a small tkinter dialog and return the typed word (or None)."""
    if not _TK_AVAILABLE:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()
    word = simpledialog.askstring(
        'নতুন ইশারা',
        'নতুন শব্দটি লিখুন (e.g. ধন্যবাদ):',
        parent=root)
    root.destroy()
    return word.strip() if word else None


# panel-local y-ranges for clickable buttons
_REC_BTN_Y   = (58, 104)
_RESET_BTN_Y = (0, 0)

# ── panel renderers ────────────────────────────────────────────────────────────
def draw_detect_panel(width, height, fonts, label_name, confidence,
                      stable_count, cooldown, sentence, speaking):
    global _REC_BTN_Y, _RESET_BTN_Y
    font_title, font_lg, font_md, font_sm, font_xs = fonts
    panel = np.full((height, width, 3), BG, dtype=np.uint8)
    img   = Image.fromarray(panel)
    draw  = ImageDraw.Draw(img)

    # header
    draw.rectangle([(0,0),(width,48)], fill=(25,25,55))
    draw.text((12,10), 'বাংলা ইশারা ভাষা', font=font_title, fill=YELLOW)
    y = 58

    # ── prominent Record button ────────────────────────────────────────────────
    BTN_H = 46
    draw.rounded_rectangle([(6,y),(width-6,y+BTN_H)], radius=10, fill=(170,55,15))
    draw.rounded_rectangle([(8,y+2),(width-8,y+BTN_H//2)], radius=8, fill=(200,80,30))
    draw.text((14, y+14), '+ নতুন ইশারা রেকর্ড করুন', font=font_sm, fill=(255,220,180))
    draw.text((width-38, y+28), '[R]', font=font_xs, fill=(255,190,130))
    _REC_BTN_Y = (y, y+BTN_H)
    y += BTN_H + 6

    # ── Reset button ───────────────────────────────────────────────────────────
    RST_H = 36
    draw.rounded_rectangle([(6,y),(width-6,y+RST_H)], radius=8, fill=(120,20,20))
    draw.rounded_rectangle([(8,y+2),(width-8,y+RST_H//2)], radius=6, fill=(160,35,35))
    draw.text((14, y+10), 'রিসেট — বাক্য মুছুন', font=font_sm, fill=(255,180,180))
    draw.text((width-38, y+20), '[C]', font=font_xs, fill=(255,140,140))
    _RESET_BTN_Y = (y, y+RST_H)
    y += RST_H + 10

    # ── no-model prompt ────────────────────────────────────────────────────────
    if not predictor.ready():
        draw.rounded_rectangle([(6,y),(width-6,y+120)], radius=12, fill=(35,25,10))
        draw.text((14, y+12), 'কোনো মডেল নেই', font=font_md, fill=ORANGE)
        draw.text((14, y+44), 'উপরের বাটনে ক্লিক করুন', font=font_xs, fill=(200,170,120))
        draw.text((14, y+64), 'বা R চাপুন — ইশারা রেকর্ড', font=font_xs, fill=(200,170,120))
        draw.text((14, y+84), 'করুন এবং মডেল তৈরি করুন।', font=font_xs, fill=(200,170,120))
        y += 130
        # still show sentence and controls below
    else:
        # current sign card
        draw.text((12,y), 'শনাক্তকৃত ইশারা', font=font_xs, fill=GRAY); y += 20
        draw.rounded_rectangle([(6,y),(width-6,y+80)], radius=10, fill=BG_CARD)
        if label_name:
            draw.text((width//2-40,y+8), label_name, font=font_lg, fill=GREEN)
            cw = int((width-32)*min(confidence,1.0))
            draw.rounded_rectangle([(14,y+56),(width-14,y+68)], radius=4, fill=(40,40,58))
            if cw > 0:
                draw.rounded_rectangle([(14,y+56),(14+cw,y+68)], radius=4, fill=GREEN)
            draw.text((14,y+70), f'{int(confidence*100)}% নিশ্চিত', font=font_xs, fill=GREEN)
        else:
            draw.text((20,y+28), 'হাত দেখান...', font=font_md, fill=GRAY)
        y += 90

        # progress bar
        if cooldown > 0:
            progress, bar_color = 1.0, BLUE
            status_msg, msg_color = 'যোগ হয়েছে!', BLUE
        elif label_name:
            progress   = stable_count/CONFIRM_FRAMES
            bar_color  = GREEN
            secs       = round((CONFIRM_FRAMES-stable_count)/30, 1)
            status_msg, msg_color = f'ধরে রাখুন... {secs}s', GREEN
        else:
            progress, bar_color   = 0.0, GRAY
            status_msg, msg_color = 'হাত স্থির রাখুন', GRAY

        bx0, bx1 = 10, width-10
        bf = int((bx1-bx0)*progress)
        draw.rounded_rectangle([(bx0,y),(bx1,y+16)], radius=8, fill=(40,40,58))
        if bf > 0:
            draw.rounded_rectangle([(bx0,y),(bx0+bf,y+16)], radius=8, fill=bar_color)
        y += 22
        draw.text((12,y), status_msg, font=font_xs, fill=msg_color); y += 22
        draw.line([(10,y),(width-10,y)], fill=DIVIDER, width=1); y += 10

    # sentence
    draw.text((12,y), 'তৈরি হচ্ছে বাক্য', font=font_xs, fill=GRAY); y += 20
    SENT_H = 88
    draw.rounded_rectangle([(6,y),(width-6,y+SENT_H)], radius=10, fill=BG_CARD)
    if sentence:
        words = sentence[-8:]
        draw.text((14,y+10), ' '.join(words[:4]), font=font_md, fill=YELLOW)
        if len(words)>4:
            draw.text((14,y+44), ' '.join(words[4:]), font=font_md, fill=YELLOW)
        draw.text((width-72,y+70), f'{len(sentence)} শব্দ', font=font_xs, fill=GRAY)
    else:
        draw.text((14,y+30), '(এখনো খালি)', font=font_sm, fill=GRAY)
    y += SENT_H+10

    # speaking / training banner
    if speaking:
        draw.rounded_rectangle([(6,y),(width-6,y+26)], radius=8, fill=(30,60,30))
        draw.text((14,y+5), 'বলছে...', font=font_xs, fill=GREEN); y += 32
    if _training:
        draw.rounded_rectangle([(6,y),(width-6,y+26)], radius=8, fill=(40,30,10))
        draw.text((14,y+5), _train_result, font=font_xs, fill=ORANGE); y += 32
    elif _train_result and not _training:
        draw.rounded_rectangle([(6,y),(width-6,y+26)], radius=8, fill=(20,40,20))
        draw.text((14,y+5), _train_result, font=font_xs, fill=GREEN); y += 32

    draw.line([(10,y),(width-10,y)], fill=DIVIDER, width=1); y += 10

    # known words
    draw.text((12,y), 'পরিচিত শব্দ', font=font_xs, fill=GRAY); y += 18
    known = list(detector_utils.LABEL_MAP.values())
    if known:
        for idx, word in enumerate(known):
            wx = 14 + (idx%2)*(width//2)
            wy = y  + (idx//2)*22
            active = (word == label_name)
            draw.text((wx,wy), ('▶ ' if active else '• ')+word,
                      font=font_xs, fill=(GREEN if active else WHITE))
        y += (len(known)//2+1)*22+6
    else:
        draw.text((14,y), 'কোনো শব্দ নেই — R চাপুন', font=font_xs, fill=GRAY); y += 22

    # controls
    draw.line([(10,y),(width-10,y)], fill=DIVIDER, width=1); y += 8
    ctls = [
        ('[V]', 'বাক্য বলুন',   ORANGE if TTS_AVAILABLE else GRAY),
        ('[C]', 'মুছুন',         RED),
        ('[B]', 'শেষ শব্দ বাদ', (200,120,50)),
        ('[Q]', 'বন্ধ',          GRAY),
    ]
    for lbl, desc, color in ctls:
        draw.text((14,y), lbl,  font=font_xs, fill=color)
        draw.text((52,y), desc, font=font_xs, fill=WHITE); y += 20

    return np.array(img)


def draw_record_panel(width, height, fonts, word, collected, target, rec_active):
    font_title, font_lg, font_md, font_sm, font_xs = fonts
    panel = np.full((height, width, 3), BG, dtype=np.uint8)
    img   = Image.fromarray(panel)
    draw  = ImageDraw.Draw(img)

    # header
    hdr_col = (80,20,20) if rec_active else (25,25,55)
    draw.rectangle([(0,0),(width,48)], fill=hdr_col)
    draw.text((12,10), 'নতুন ইশারা রেকর্ড', font=font_title,
              fill=(255,100,100) if rec_active else YELLOW)
    y = 60

    # word badge
    draw.text((12,y), 'শব্দ:', font=font_xs, fill=GRAY); y += 18
    draw.rounded_rectangle([(6,y),(width-6,y+60)], radius=10, fill=BG_CARD)
    draw.text((width//2-40, y+10), word, font=font_lg,
              fill=(255,100,100) if rec_active else GREEN)
    y += 70

    # progress
    draw.text((12,y), 'নমুনা সংগ্রহ', font=font_xs, fill=GRAY); y += 18
    pct = min(collected/target, 1.0)
    bx0, bx1 = 10, width-10
    bf = int((bx1-bx0)*pct)
    draw.rounded_rectangle([(bx0,y),(bx1,y+20)], radius=10, fill=(40,40,58))
    if bf > 0:
        col = (200,40,40) if rec_active else (55,180,90)
        draw.rounded_rectangle([(bx0,y),(bx0+bf,y+20)], radius=10, fill=col)
    y += 26
    draw.text((12,y), f'{collected} / {target} নমুনা', font=font_sm,
              fill=(255,100,100) if rec_active else WHITE); y += 28

    # big status
    draw.rounded_rectangle([(6,y),(width-6,y+60)], radius=10, fill=BG_CARD)
    if rec_active:
        draw.text((14,y+12), '● রেকর্ড হচ্ছে...', font=font_md, fill=(255,80,80))
    else:
        draw.text((14,y+12), 'প্রস্তুত', font=font_md, fill=GRAY)
    y += 70

    draw.line([(10,y),(width-10,y)], fill=DIVIDER, width=1); y += 14

    # tips
    tips = [
        'হাত ফ্রেমের মাঝে রাখুন',
        'আলো পর্যাপ্ত কিনা দেখুন',
        'বিভিন্ন কোণ থেকে রেকর্ড করুন',
        'ধীরে ধীরে ইশারা করুন',
    ]
    draw.text((12,y), 'টিপস:', font=font_xs, fill=GRAY); y += 18
    for tip in tips:
        draw.text((14,y), f'• {tip}', font=font_xs, fill=(150,150,170)); y += 18
    y += 8

    draw.line([(10,y),(width-10,y)], fill=DIVIDER, width=1); y += 10

    # controls
    ctls = [
        ('[SPACE]', 'রেকর্ড শুরু/বন্ধ', (255,180,50)),
        ('[Q]',     'সংরক্ষণ করুন',      GREEN),
        ('[ESC]',   'বাতিল করুন',         RED),
    ]
    for lbl, desc, color in ctls:
        draw.text((14,y), lbl,  font=font_xs, fill=color)
        draw.text((72,y), desc, font=font_xs, fill=WHITE); y += 22

    return np.array(img)


def overlay_detect(image_np, label_name, confidence, left, bottom,
                   iw, ih, stable_count, cooldown, speaking, font_lg, font_xs):
    img  = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    if label_name:
        draw.text((max(int(left),4), max(int(bottom)-50,4)),
                  label_name, font=font_lg, fill=GREEN)
    ax, ay   = iw//2, ih-18
    progress = stable_count/CONFIRM_FRAMES if cooldown==0 else 1.0
    sweep    = int(360*progress)
    ring_col = BLUE if cooldown>0 else GREEN
    r = 14
    draw.ellipse([(ax-r,ay-r),(ax+r,ay+r)], outline=(50,50,60), width=4)
    if sweep > 0:
        draw.arc([(ax-r,ay-r),(ax+r,ay+r)], start=-90, end=-90+sweep,
                 fill=ring_col, width=4)
    if speaking:
        draw.rectangle([(0,0),(iw,28)], fill=(10,40,10))
        draw.text((8,4), 'বলছে...', font=font_xs, fill=GREEN)
    return np.array(img)


def overlay_record(image_np, word, collected, target, rec_active, iw, ih, font_md):
    img  = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)

    # coloured border
    border_col = (220,40,40) if rec_active else (50,150,80)
    thickness  = 6
    draw.rectangle([(0,0),(iw-1,ih-1)], outline=border_col, width=thickness)

    # top status bar
    bar_col = (100,15,15) if rec_active else (15,50,30)
    draw.rectangle([(0,0),(iw,44)], fill=bar_col)
    status = f'● রেকর্ড  "{word}"' if rec_active else f'প্রস্তুত  "{word}"'
    draw.text((12,8), status, font=font_md,
              fill=(255,100,100) if rec_active else (100,255,150))

    # progress bar at bottom
    pct = min(collected/target, 1.0)
    bw  = int(iw * pct)
    draw.rectangle([(0,ih-20),(iw,ih)], fill=(30,30,40))
    if bw > 0:
        col = (200,40,40) if rec_active else (40,180,80)
        draw.rectangle([(0,ih-20),(bw,ih)], fill=col)
    draw.text((iw//2-40, ih-18), f'{collected}/{target}', font=font_md,
              fill=(255,255,255))

    return np.array(img)


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', dest='score_thresh', type=float, default=0.4)
    parser.add_argument('-src', dest='video_source',             default=0)
    parser.add_argument('-ds',  dest='display',     type=int,    default=1)
    args = parser.parse_args()

    cap       = cv2.VideoCapture(int(args.video_source))
    im_width  = int(cap.get(3))
    im_height = int(cap.get(4))

    fontpath   = './fonts/SolaimanLipi_22-02-2012.ttf'
    font_title = ImageFont.truetype(fontpath, 20)
    font_lg    = ImageFont.truetype(fontpath, 36)
    font_md    = ImageFont.truetype(fontpath, 22)
    font_sm    = ImageFont.truetype(fontpath, 18)
    font_xs    = ImageFont.truetype(fontpath, 14)
    fonts      = (font_title, font_lg, font_md, font_sm, font_xs)

    win_title = 'Bangla Sign Language Detector'
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, im_width + PANEL_W, im_height)

    # ── mouse callback for clickable Record button ──────────────────────────────
    _btn_clicked   = [False]
    _reset_clicked = [False]

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= im_width:
                py = y
                if _REC_BTN_Y[0] <= py <= _REC_BTN_Y[1]:
                    _btn_clicked[0] = True
                elif _RESET_BTN_Y[0] <= py <= _RESET_BTN_Y[1]:
                    _reset_clicked[0] = True

    cv2.setMouseCallback(win_title, _on_mouse)

    # ── app state ──────────────────────────────────────────────────────────────
    APP_MODE     = 'detect'    # 'detect' | 'record'

    # detect-mode state
    sentence     = []
    stable_label = None
    stable_count = 0
    cooldown     = 0

    # record-mode state
    rec_word      = ''
    rec_samples   = []
    rec_active    = False      # True = currently capturing frames

    while True:
        ret, image_np = cap.read()
        if not ret:
            break
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except Exception:
            continue

        boxes, scores, landmarks_px, raw_lms = detector_utils.detect_objects(
            image_np, detection_graph, sess)
        detector_utils.draw_skeleton(image_np, landmarks_px)

        # ── RECORD MODE ────────────────────────────────────────────────────────
        if APP_MODE == 'record':
            # Capture landmark features when actively recording
            if rec_active and raw_lms:
                feat = detector_utils.extract_landmark_features(raw_lms[0])
                rec_samples.append(feat)

            cam_out = overlay_record(
                image_np, rec_word, len(rec_samples), RECORD_TARGET,
                rec_active, im_width, im_height, font_md)

            panel = draw_record_panel(
                PANEL_W, im_height, fonts,
                rec_word, len(rec_samples), RECORD_TARGET, rec_active)

            combined = np.concatenate(
                [cv2.cvtColor(cam_out, cv2.COLOR_RGB2BGR),
                 panel[:,:,::-1]], axis=1)
            cv2.imshow(win_title, combined)

            key = cv2.waitKey(1) & 0xFF

            # Auto-finish when target reached
            if len(rec_samples) >= RECORD_TARGET:
                rec_active = False
                key = ord('q')   # force save

            if key == ord(' '):
                rec_active = not rec_active

            elif key == ord('q'):
                # Save samples
                if rec_samples:
                    save_path = os.path.join(LANDMARK_DIR, f'{rec_word}.npy')
                    existing  = []
                    if os.path.exists(save_path):
                        existing = list(np.load(save_path, allow_pickle=True))
                    all_data = existing + rec_samples
                    np.save(save_path, np.array(all_data))
                    print(f'Saved {len(all_data)} samples → {save_path}')

                    # Prompt to train
                    if _TK_AVAILABLE:
                        r2 = tk.Tk(); r2.withdraw(); r2.attributes('-topmost', True)
                        do_train = messagebox.askyesno(
                            'প্রশিক্ষণ',
                            f'{len(all_data)} নমুনা সংরক্ষিত হয়েছে।\n'
                            'এখন মডেল প্রশিক্ষণ দিতে চান?',
                            parent=r2)
                        r2.destroy()
                        if do_train:
                            start_training()

                APP_MODE   = 'detect'
                rec_word   = ''
                rec_samples = []
                rec_active  = False
                predictor.reset()

            elif key == 27:   # ESC — cancel
                APP_MODE   = 'detect'
                rec_word   = ''
                rec_samples = []
                rec_active  = False

        # ── DETECT MODE ────────────────────────────────────────────────────────
        else:
            left, bottom, label_name, confidence = detector_utils.draw_box_on_image(
                predictor, 1, args.score_thresh, scores, boxes,
                im_width, im_height, image_np,
                landmarks_px=landmarks_px, raw_landmarks=raw_lms)

            # Accumulation
            if cooldown > 0:
                cooldown -= 1; stable_label = None; stable_count = 0
            elif label_name and confidence >= MIN_CONFIRM_CONF:
                stable_count = stable_count+1 if label_name==stable_label else 1
                stable_label = label_name
                if stable_count >= CONFIRM_FRAMES:
                    sentence.append(label_name)
                    stable_label=None; stable_count=0; cooldown=COOLDOWN_FRAMES
            else:
                stable_label=None; stable_count=0

            cam_out = overlay_detect(
                image_np, label_name, confidence, left, bottom,
                im_width, im_height, stable_count, cooldown,
                _speaking, font_lg, font_xs)

            panel = draw_detect_panel(
                PANEL_W, im_height, fonts,
                label_name, confidence, stable_count, cooldown,
                sentence, _speaking)

            combined = np.concatenate(
                [cv2.cvtColor(cam_out, cv2.COLOR_RGB2BGR),
                 panel[:,:,::-1]], axis=1)
            cv2.imshow(win_title, combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r') or _btn_clicked[0]:
                _btn_clicked[0] = False
                # Enter recording mode
                word = ask_word_name()
                if word:
                    rec_word    = word
                    rec_samples = []
                    rec_active  = False
                    APP_MODE    = 'record'
                    predictor.reset()

            elif key == ord('v') and sentence and not _speaking:
                speak(' '.join(sentence))

            elif key == ord('c') or _reset_clicked[0]:
                _reset_clicked[0] = False
                sentence.clear(); stable_label=None; stable_count=0; cooldown=0
                predictor.reset()

            elif key == ord('b') and sentence:
                sentence.pop()

    cap.release()
    cv2.destroyAllWindows()
    if TTS_AVAILABLE:
        pygame.mixer.quit()
