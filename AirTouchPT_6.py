import cv2
import mediapipe as mp
import time
import pyautogui
import numpy as np
import math
import threading
import queue
import speech_recognition as sr
import os
import warnings
import tempfile
import pygame
import pickle
import sqlite3
from datetime import datetime
from gtts import gTTS
from PIL import ImageFont, ImageDraw, Image

# --- Environment ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# =============================================================
# 1. 한글 텍스트 렌더링 (PIL) — 속도 최적화 유지
# =============================================================
def _load_korean_font(size):
    candidates = [
        "malgun.ttf", "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/System/Library/Fonts/AppleGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path): return ImageFont.truetype(path, size)
    return ImageFont.load_default()

_font_cache = {}
def get_font(size):
    if size not in _font_cache: _font_cache[size] = _load_korean_font(size)
    return _font_cache[size]

def draw_text_batch(frame, items):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, pos, size, color in items:
        draw.text(pos, text, font=get_font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =============================================================
# 2. Database System (유지)
# =============================================================
def init_db():
    with sqlite3.connect('fitness_records.db') as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE)')
        c.execute('''CREATE TABLE IF NOT EXISTS workout_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_name TEXT, date TEXT, exercise_type TEXT, count INTEGER)''')
        conn.commit()

def save_workout(exercise_type, count, user_name):
    if count > 0:
        with sqlite3.connect('fitness_records.db') as conn:
            c = conn.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            c.execute('INSERT INTO workout_logs (user_name, date, exercise_type, count) VALUES (?, ?, ?, ?)',
                      (user_name, now, exercise_type, count))
            conn.commit()
            print(f"[DB] Saved — {user_name} | {exercise_type} | {count} reps")

def get_user_summary(user_name):
    with sqlite3.connect('fitness_records.db') as conn:
        c = conn.cursor()
        c.execute('''SELECT exercise_type, SUM(count), COUNT(*), MAX(date)
                     FROM workout_logs WHERE user_name = ? GROUP BY exercise_type''', (user_name,))
        rows = c.fetchall()
    return {row[0]: {'total': row[1], 'sessions': row[2], 'last': row[3][:10] if row[3] else '-'} for row in rows}

def announce_user_summary(user_name):
    summary = get_user_summary(user_name)
    if not summary:
        speak(f"{user_name}님, 아직 운동 기록이 없습니다. 첫 운동을 시작해보세요.")
        return
    parts = [f"{ex} 총 {d['total']}회" for ex, d in summary.items()]
    speak(f"{user_name}님, 환영합니다. 기존 기록: {', '.join(parts)}.")

init_db()

# =============================================================
# 3. ML 모델 로드 (개별 학습 모델 방식 복구)
# =============================================================
squat_model = None
pushup_model = None

try:
    with open('squat_model.pkl', 'rb') as f:
        squat_model = pickle.load(f)
    print("[Model] squat_model.pkl loaded.")
except: pass

try:
    with open('pushup_model.pkl', 'rb') as f:
        pushup_model = pickle.load(f)
    print("[Model] pushup_model.pkl loaded.")
except: pass

if not squat_model and not pushup_model:
    print("[Model] No custom models found — angle fallback active.")

# =============================================================
# 4. 각도 계산 (폴백용)
# =============================================================
def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# =============================================================
# 5. TTS System (유지)
# =============================================================
pygame.mixer.init()
speech_queue = queue.Queue(maxsize=1)

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None: break
        tmp_path = None
        try:
            tts = gTTS(text=text, lang='ko')
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f: tmp_path = f.name
            tts.save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.05)
            pygame.mixer.music.unload()
        except: pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
            speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    print(f"[TTS] {text}")
    while not speech_queue.empty():
        try: speech_queue.get_nowait(); speech_queue.task_done()
        except queue.Empty: break
    try: speech_queue.put_nowait(text)
    except queue.Full: pass

# =============================================================
# 6. 자세 교정 피드백 (유지)
# =============================================================
last_feedback_msg, last_feedback_time, FEEDBACK_COOLDOWN = "", 0, 4.0
def get_posture_feedback(lm_list):
    ls_y, rs_y = lm_list[11].y, lm_list[12].y
    lh_y, rh_y = lm_list[23].y, lm_list[24].y
    lk_x, rk_x = lm_list[25].x, lm_list[26].x
    la_x, ra_x = lm_list[27].x, lm_list[28].x
    if ls_y - rs_y > 0.04:  return "오른쪽으로 기울었습니다."
    if ls_y - rs_y < -0.04: return "왼쪽으로 기울었습니다."
    if lh_y - rh_y > 0.04:  return "엉덩이가 왼쪽으로 기울었습니다."
    if lh_y - rh_y < -0.04: return "엉덩이가 오른쪽으로 기울었습니다."
    if (lk_x - la_x) > 0.06: return "왼쪽 무릎이 안으로 쏠렸습니다."
    if (ra_x - rk_x) > 0.06: return "오른쪽 무릎이 안으로 쏠렸습니다."
    return None

# =============================================================
# 7. 음성인식 (유지 - 원하는 종목 추가 가능)
# =============================================================
EXERCISE_MAP = {
    "스쿼트": "Squat",   "푸시업": "Pushup",  "팔굽혀펴기": "Pushup", 
    "풀업":   "Pullup",  "턱걸이": "Pullup", 
    "싯업":   "Situp",   "윗몸":   "Situp", 
    "점핑":   "JumpingJack", "점핑잭": "JumpingJack"
}
KOR_NAME_MAP = {
    "Squat": "스쿼트", "Pushup": "푸시업", "Pullup": "풀업", 
    "Situp": "윗몸일으키기", "JumpingJack": "점핑잭"
}

def listen_to_voice(r, source):
    try: audio = r.listen(source, timeout=4, phrase_time_limit=3); return r.recognize_google(audio, language='ko-KR')
    except: return None

def trigger_voice_recognition():
    global ex_sub_mode, ex_counter, ex_stage, voice_listening, current_user, user_summary_cache
    voice_listening = True; r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("명령을 말씀하세요.")
        r.adjust_for_ambient_noise(source, duration=0.5)
        cmd = listen_to_voice(r, source)
        if not cmd: speak("인식하지 못했습니다."); voice_listening = False; return
        print(f"[Voice] Recognized: '{cmd}'")
        
        if "등록" in cmd or "가입" in cmd:
            speak("등록하실 이름을 말씀해 주세요."); name = listen_to_voice(r, source)
            if name:
                name = name.replace(" ", "")
                with sqlite3.connect('fitness_records.db') as conn:
                    try:
                        conn.cursor().execute("INSERT INTO users (name) VALUES (?)", (name,)); conn.commit()
                        current_user = name; user_summary_cache = get_user_summary(current_user); speak(f"{name}님, 등록 완료.")
                    except: speak("이미 등록된 이름입니다.")
            else: speak("이름을 인식하지 못했습니다.")
        elif "로그인" in cmd:
            speak("이름을 말씀해 주세요."); name = listen_to_voice(r, source)
            if name:
                name = name.replace(" ", "")
                with sqlite3.connect('fitness_records.db') as conn:
                    result = conn.cursor().execute("SELECT name FROM users WHERE name=?", (name,)).fetchone()
                if result: current_user = result[0]; user_summary_cache = get_user_summary(current_user); announce_user_summary(current_user)
                else: speak("등록되지 않은 사용자입니다.")
            else: speak("이름을 인식하지 못했습니다.")
        elif "로그아웃" in cmd:
            if current_user != "Guest": speak(f"{current_user}님, 로그아웃."); current_user = "Guest"; user_summary_cache = {}
            else: speak("현재 로그인된 사용자가 없습니다.")
        elif "종료" in cmd or "그만" in cmd:
            if ex_sub_mode != "Waiting" and ex_counter > 0: save_workout(ex_sub_mode, ex_counter, current_user); user_summary_cache = get_user_summary(current_user)
            ex_sub_mode = "Waiting"; ex_counter = 0; speak("기록 저장 완료.")
        else:
            matched = next((mode for key, mode in EXERCISE_MAP.items() if key in cmd), None)
            if matched:
                if ex_sub_mode != "Waiting" and ex_counter > 0: save_workout(ex_sub_mode, ex_counter, current_user); user_summary_cache = get_user_summary(current_user)
                ex_sub_mode = matched; ex_counter = 0; ex_stage = "up"
                speak(f"{current_user}님의 {KOR_NAME_MAP.get(matched, matched)} 시작합니다.")
            else: speak("알 수 없는 명령입니다.")
        voice_listening = False

# =============================================================
# 8. MediaPipe & 상태 변수 (유지)
# =============================================================
mp_hands_mod, mp_pose_mod, mp_draw = mp.solutions.hands, mp.solutions.pose, mp.solutions.drawing_utils
hands_detector = mp_hands_mod.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose_detector  = mp_pose_mod.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

PT_MODE, last_mode_switch_t = False, 0
ex_sub_mode, ex_counter, ex_stage = "Waiting", 0, 'up'
voice_listening, current_user, user_summary_cache = False, "Guest", {}
sw, sh, prev_x, prev_y = pyautogui.size()[0], pyautogui.size()[1], 0, 0

# =============================================================
# 9. Main Loop
# =============================================================
cap = cv2.VideoCapture(0)
WIN_NAME = 'Custom AI PT Master'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)

speak("오디오 피티 시작. 스페이스바를 눌러주세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_res = hands_detector.process(rgb_frame)
    pose_res  = pose_detector.process(rgb_frame)

    rect = cv2.getWindowImageRect(WIN_NAME)
    win_w, win_h = rect[2], rect[3]
    if win_w > 0 and win_h > 0: frame = cv2.resize(frame, (win_w, win_h))
    h, w, _ = frame.shape

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        if ex_sub_mode != "Waiting" and ex_counter > 0: save_workout(ex_sub_mode, ex_counter, current_user)
        break
    if key == ord(' ') and time.time() - last_mode_switch_t > 2.0:
        PT_MODE = not PT_MODE; last_mode_switch_t = time.time()
        speak(f"바디 모드. 접속자: {current_user}" if PT_MODE else "핸드 모드 전환.")
    if PT_MODE and key == ord('q') and not voice_listening:
        threading.Thread(target=trigger_voice_recognition, daemon=True).start()

    text_items = []

    # ── [Mode 1] 핸드 모드 ──
    if not PT_MODE:
        text_items.append(("핸드 모드  |  [SPACE] 바디 모드", (20, 15), 26, (0, 255, 0)))
        if hand_res.multi_hand_landmarks:
            lm = hand_res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands_mod.HAND_CONNECTIONS)
            tx, ty = np.interp(lm.landmark[8].x, [0.2, 0.8], [0, sw]), np.interp(lm.landmark[8].y, [0.2, 0.8], [0, sh])
            cx, cy = prev_x + (tx - prev_x) * 0.2, prev_y + (ty - prev_y) * 0.2
            pyautogui.moveTo(cx, cy); prev_x, prev_y = cx, cy
            if math.sqrt((lm.landmark[12].x - lm.landmark[4].x)**2 + (lm.landmark[12].y - lm.landmark[4].y)**2) < 0.05: pyautogui.click()

    # ── [Mode 2] 바디 PT 모드 ──
    else:
        text_items.append((f"사용자: {current_user}", (20, 10), 26, (255, 255, 255)))
        status_txt = "듣는 중..." if voice_listening else f"운동: {ex_sub_mode}"
        text_items.append((status_txt, (20, 48), 28, (0, 0, 255) if voice_listening else (0, 165, 255)))
        text_items.append(("[Q] 음성 명령  |  [SPACE] 핸드 모드", (20, h - 35), 22, (200, 200, 200)))

        if user_summary_cache:
            rx = w - 330
            text_items.append(("[ 운동 기록 ]", (rx, 50), 22, (200, 200, 200)))
            for i, (ex, data) in enumerate(user_summary_cache.items()):
                text_items.append((f"{ex}: 총 {data['total']}회 / {data['sessions']}세션", (rx, 50 + 38*(i+1)), 20, (180, 255, 180)))
                text_items.append((f"  마지막: {data['last']}", (rx, 50 + 38*(i+1) + 20), 18, (150, 150, 255)))

        if pose_res.pose_landmarks and ex_sub_mode != "Waiting":
            lm_obj  = pose_res.pose_landmarks
            lm_list = lm_obj.landmark
            mp_draw.draw_landmarks(frame, lm_obj, mp_pose_mod.POSE_CONNECTIONS)
            text_items.append((f"횟수: {ex_counter}", (20, 92), 36, (0, 255, 255)))

            prediction = None
            
            # 🚨 [복구 완료] 직관적인 개별 모델(132 좌표) 추론 방식
            active_model = None
            if ex_sub_mode == "Squat": active_model = squat_model
            elif ex_sub_mode == "Pushup": active_model = pushup_model
            
            if active_model:
                # 33개 관절의 x, y, z, visibility를 1차원 배열(132개)로 쫙 폅니다.
                pose_row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in lm_list]).flatten())
                
                try:
                    prediction = active_model.predict([pose_row])[0]
                    # 직접 학습한 모델이므로 확률이 아닌 확실한 예측 결과 텍스트를 바로 사용합니다.
                    
                    pred_lower = prediction.lower().replace(" ", "")
                    
                    if "_down" in pred_lower and ex_stage == "up":
                        ex_stage = "down"
                        speak("내려갔습니다.")
                    elif "_up" in pred_lower and ex_stage == "down":
                        ex_stage = "up"
                        ex_counter += 1
                        speak(f"{ex_counter}회")

                    text_items.append((f"AI: {prediction}", (20, 143), 24, (255, 100, 255)))
                except Exception as e:
                    text_items.append((f"모델 오류: {e}", (20, 143), 24, (100, 100, 200)))

            # 모델이 없거나 스쿼트 모델이 없을 때의 각도 기반 폴백
            if prediction is None and ex_sub_mode == "Squat":
                hip   = [lm_list[23].x, lm_list[23].y]
                knee  = [lm_list[25].x, lm_list[25].y]
                ankle = [lm_list[27].x, lm_list[27].y]
                angle = get_angle(hip, knee, ankle)
                text_items.append((f"각도: {int(angle)}", (20, 143), 24, (200, 200, 200)))
                if angle < 95 and ex_stage == "up":  ex_stage = "down"
                if angle > 160 and ex_stage == "down":
                    ex_stage = "up"; ex_counter += 1; speak(f"{ex_counter}회")

            if ex_sub_mode == "Squat" and ex_stage == "down":
                feedback = get_posture_feedback(lm_list)
                now = time.time()
                if feedback:
                    if feedback != last_feedback_msg or (now - last_feedback_time) > FEEDBACK_COOLDOWN:
                        speak(feedback); last_feedback_msg = feedback; last_feedback_time = now
                    text_items.append((f"! {feedback}", (20, 200), 24, (0, 0, 255)))
                else:
                    text_items.append(("Good posture", (20, 200), 24, (0, 255, 0)))
                    last_feedback_msg = ""

    frame = draw_text_batch(frame, text_items)
    cv2.imshow(WIN_NAME, frame)

pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
