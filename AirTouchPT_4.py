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
from gtts import gTTS
import sqlite3
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

# --- Environment ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# =============================================================
# 한글 텍스트 렌더링 (PIL 사용)
# =============================================================
# cv2.putText()는 영문 폰트만 지원 → 한글은 ??? 로 깨짐
# PIL로 한글을 그린 뒤 numpy array로 변환해서 cv2 frame에 합성
# 폰트 파일 우선순위: malgun.ttf(맑은 고딕) → NanumGothic → 시스템 기본
# =============================================================

def _load_korean_font(size):
    candidates = [
        "malgun.ttf",                                 # Windows 맑은 고딕
        "malgunbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
        "/System/Library/Fonts/AppleGothic.ttf",            # Mac
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

# 자주 쓰는 크기 캐싱
_font_cache = {}
def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = _load_korean_font(size)
    return _font_cache[size]

def put_text_kr(frame, text, pos, size=28, color=(255, 255, 255)):
    """
    한글/영문 텍스트를 frame에 그려서 반환
    pos  : (x, y) - 텍스트 좌측 상단 기준
    size : 폰트 크기 (px)
    color: BGR → RGB 자동 변환
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(size)
    rgb_color = (color[2], color[1], color[0])  # BGR → RGB
    draw.text(pos, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =============================================================
# Database System
# =============================================================
def init_db():
    with sqlite3.connect('fitness_records.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS workout_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                date TEXT,
                exercise_type TEXT,
                count INTEGER
            )
        ''')
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
        c.execute('''
            SELECT exercise_type, SUM(count), COUNT(*), MAX(date)
            FROM workout_logs
            WHERE user_name = ?
            GROUP BY exercise_type
        ''', (user_name,))
        rows = c.fetchall()
    summary = {}
    for exercise_type, total, sessions, last_date in rows:
        summary[exercise_type] = {
            'total'   : total,
            'sessions': sessions,
            'last'    : last_date[:10] if last_date else '-'
        }
    return summary

def announce_user_summary(user_name):
    summary = get_user_summary(user_name)
    if not summary:
        print(f"[DB] {user_name} — no records yet")
        speak(f"{user_name}님, 아직 운동 기록이 없습니다. 첫 운동을 시작해보세요.")
        return
    print(f"[DB] ── {user_name} record summary ──")
    tts_parts = []
    for ex, data in summary.items():
        print(f"     {ex}: total {data['total']} reps / {data['sessions']} sessions / last {data['last']}")
        tts_parts.append(f"{ex} 총 {data['total']}회")
    speak(f"{user_name}님, 환영합니다. 기존 기록: {', '.join(tts_parts)}.")

init_db()

# =============================================================
# ML Model Load
# =============================================================
try:
    with open('squat_model.pkl', 'rb') as f:
        squat_model = pickle.load(f)
    print("[Model] squat_model.pkl loaded.")
except:
    squat_model = None
    print("[Model] squat_model.pkl not found — angle fallback active.")

try:
    with open('pushup_model.pkl', 'rb') as f:
        pushup_model = pickle.load(f)
    print("[Model] pushup_model.pkl loaded.")
except:
    pushup_model = None

# =============================================================
# TTS System - gTTS + pygame
# =============================================================
pygame.mixer.init()
speech_queue = queue.Queue(maxsize=1)

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        tmp_path = None
        try:
            tts = gTTS(text=text, lang='ko')
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                tmp_path = f.name
            tts.save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    print(f"[TTS] {text}")
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            break
    try:
        speech_queue.put_nowait(text)
    except queue.Full:
        pass

# --- Variables ---
PT_MODE = False
last_mode_switch_time = 0
ex_sub_mode = "Waiting"
ex_counter = 0
ex_stage = 'up'
voice_listening = False
current_user = "Guest"
user_summary_cache = {}

last_feedback_msg = ""
last_feedback_time = 0
FEEDBACK_COOLDOWN = 4.0

# --- MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# =============================================================
# Angle-based fallback
# =============================================================
def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# =============================================================
# Posture Feedback
# =============================================================
def check_squat_posture(lm):
    ls_y = lm[11].y; rs_y = lm[12].y
    lh_y = lm[23].y; rh_y = lm[24].y
    lk_x = lm[25].x; rk_x = lm[26].x
    la_x = lm[27].x; ra_x = lm[28].x

    if ls_y - rs_y > 0.04: return "오른쪽으로 기울었습니다."
    if ls_y - rs_y < -0.04: return "왼쪽으로 기울었습니다."
    if lh_y - rh_y > 0.04: return "엉덩이가 왼쪽으로 기울었습니다."
    if lh_y - rh_y < -0.04: return "엉덩이가 오른쪽으로 기울었습니다."
    if (lk_x - la_x) > 0.06: return "왼쪽 무릎이 안으로 쏠렸습니다."
    if (ra_x - rk_x) > 0.06: return "오른쪽 무릎이 안으로 쏠렸습니다."
    return None

def handle_posture_feedback(lm, frame):
    global last_feedback_msg, last_feedback_time
    feedback = check_squat_posture(lm)
    now = time.time()
    if feedback:
        if feedback != last_feedback_msg or (now - last_feedback_time) > FEEDBACK_COOLDOWN:
            speak(feedback)
            last_feedback_msg = feedback
            last_feedback_time = now
        frame = put_text_kr(frame, f"! {feedback}", (20, 200), size=26, color=(0, 0, 255))
    else:
        frame = put_text_kr(frame, "Good posture", (20, 200), size=26, color=(0, 255, 0))
        last_feedback_msg = ""
    return frame

# =============================================================
# 화면 우측 상단 — 유저 누적 기록 표시
# =============================================================
def draw_user_summary(frame, summary):
    if not summary:
        return frame
    x = frame.shape[1] - 320
    y = 50
    frame = put_text_kr(frame, "[ 운동 기록 ]", (x, y), size=22, color=(200, 200, 200))
    for i, (ex, data) in enumerate(summary.items()):
        line1 = f"{ex}: 총 {data['total']}회 / {data['sessions']}세션"
        line2 = f"  마지막: {data['last']}"
        frame = put_text_kr(frame, line1, (x, y + 35 * (i + 1)),      size=20, color=(180, 255, 180))
        frame = put_text_kr(frame, line2, (x, y + 35 * (i + 1) + 20), size=18, color=(150, 150, 255))
    return frame

# =============================================================
# Voice Recognition
# =============================================================
def listen_to_voice(r, source):
    try:
        audio = r.listen(source, timeout=4, phrase_time_limit=3)
        return r.recognize_google(audio, language='ko-KR')
    except:
        return None

def trigger_voice_recognition():
    global ex_sub_mode, ex_counter, ex_stage, voice_listening
    global current_user, user_summary_cache
    voice_listening = True
    r = sr.Recognizer()

    with sr.Microphone() as source:
        speak("명령을 말씀하세요.")
        r.adjust_for_ambient_noise(source, duration=0.5)
        cmd = listen_to_voice(r, source)

        if not cmd:
            speak("인식하지 못했습니다.")
            voice_listening = False
            return

        print(f"[Voice] Recognized: '{cmd}'")

        if "등록" in cmd or "가입" in cmd:
            speak("등록하실 이름을 말씀해 주세요.")
            name = listen_to_voice(r, source)
            if name:
                name = name.replace(" ", "")
                with sqlite3.connect('fitness_records.db') as conn:
                    c = conn.cursor()
                    try:
                        c.execute("INSERT INTO users (name) VALUES (?)", (name,))
                        conn.commit()
                        current_user = name
                        user_summary_cache = get_user_summary(current_user)
                        speak(f"{name}님, 등록 완료. 첫 운동을 시작해보세요.")
                        print(f"[DB] New user: {name}")
                    except sqlite3.IntegrityError:
                        speak("이미 등록된 이름입니다.")
            else:
                speak("이름을 인식하지 못했습니다.")

        elif "로그인" in cmd:
            speak("이름을 말씀해 주세요.")
            name = listen_to_voice(r, source)
            if name:
                name = name.replace(" ", "")
                with sqlite3.connect('fitness_records.db') as conn:
                    c = conn.cursor()
                    c.execute("SELECT name FROM users WHERE name=?", (name,))
                    result = c.fetchone()
                if result:
                    current_user = result[0]
                    user_summary_cache = get_user_summary(current_user)
                    announce_user_summary(current_user)
                    print(f"[DB] Logged in: {current_user}")
                else:
                    speak("등록되지 않은 사용자입니다.")
            else:
                speak("이름을 인식하지 못했습니다.")

        elif "로그아웃" in cmd:
            if current_user != "Guest":
                speak(f"{current_user}님, 로그아웃 되었습니다.")
                current_user = "Guest"
                user_summary_cache = {}
            else:
                speak("현재 로그인된 사용자가 없습니다.")

        elif "스쿼트" in cmd:
            if ex_sub_mode != "Waiting" and ex_counter > 0:
                save_workout(ex_sub_mode, ex_counter, current_user)
                user_summary_cache = get_user_summary(current_user)
            ex_sub_mode = "Squat"; ex_counter = 0; ex_stage = "up"
            speak(f"{current_user}님의 스쿼트를 시작합니다.")

        elif "푸시업" in cmd or "팔굽혀" in cmd:
            if ex_sub_mode != "Waiting" and ex_counter > 0:
                save_workout(ex_sub_mode, ex_counter, current_user)
                user_summary_cache = get_user_summary(current_user)
            ex_sub_mode = "Pushup"; ex_counter = 0; ex_stage = "up"
            speak(f"{current_user}님의 푸시업을 시작합니다.")

        elif "종료" in cmd or "그만" in cmd:
            if ex_sub_mode != "Waiting" and ex_counter > 0:
                save_workout(ex_sub_mode, ex_counter, current_user)
                user_summary_cache = get_user_summary(current_user)
            ex_sub_mode = "Waiting"; ex_counter = 0
            speak(f"운동 종료. {current_user}님의 기록을 저장했습니다.")

        else:
            speak("알 수 없는 명령입니다.")

        voice_listening = False

# =============================================================
# Main Loop
# =============================================================
cap = cv2.VideoCapture(0)
sw, sh = pyautogui.size()
prev_x, prev_y = 0, 0

# WINDOW_NORMAL: 사용자가 창을 마우스로 드래그해서 크기 조절 가능
# 창 크기가 바뀌면 매 프레임에서 실제 창 크기를 읽어 frame을 리사이즈
WIN_NAME = 'Barrier-Free AI PT'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)  # 초기 창 크기

print("[System] Starting — Barrier-Free AI PT")
speak("오디오 피티 시작. 스페이스바를 눌러 모드를 바꾸세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_res = hands.process(rgb_frame)
    pose_res = pose.process(rgb_frame)

    # 현재 창 크기 읽기 → frame을 창 크기에 맞게 리사이즈
    # getWindowImageRect()는 (x, y, w, h) 반환
    rect = cv2.getWindowImageRect(WIN_NAME)
    win_w, win_h = rect[2], rect[3]
    if win_w > 0 and win_h > 0:
        frame = cv2.resize(frame, (win_w, win_h))
    h, w, _ = frame.shape

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        if ex_sub_mode != "Waiting" and ex_counter > 0:
            save_workout(ex_sub_mode, ex_counter, current_user)
        break

    if key == ord(' ') and time.time() - last_mode_switch_time > 2.0:
        PT_MODE = not PT_MODE
        last_mode_switch_time = time.time()
        if PT_MODE:
            speak(f"바디 모드. 현재 접속자는 {current_user}님 입니다.")
        else:
            speak("핸드 모드 전환.")

    if PT_MODE and key == ord('q') and not voice_listening:
        threading.Thread(target=trigger_voice_recognition, daemon=True).start()

    # --- [Mode 1] Hand Control ---
    if not PT_MODE:
        frame = put_text_kr(frame, "핸드 모드  |  [SPACE] 바디 모드 전환", (20, 15), size=26, color=(0, 255, 0))
        if hand_res.multi_hand_landmarks:
            # 랜드마크를 리사이즈된 frame 크기에 맞게 스케일 조정해서 그리기
            landmarks = hand_res.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            idx = landmarks.landmark[8]
            tx = np.interp(idx.x, [0.2, 0.8], [0, sw])
            ty = np.interp(idx.y, [0.2, 0.8], [0, sh])
            cx = prev_x + (tx - prev_x) * 0.2
            cy = prev_y + (ty - prev_y) * 0.2
            pyautogui.moveTo(cx, cy)
            prev_x, prev_y = cx, cy

            c_dist = math.sqrt((landmarks.landmark[12].x - landmarks.landmark[4].x) ** 2 +
                               (landmarks.landmark[12].y - landmarks.landmark[4].y) ** 2)
            if c_dist < 0.05:
                pyautogui.click()

    # --- [Mode 2] Body Motion ---
    else:
        frame = put_text_kr(frame, f"사용자: {current_user}", (20, 10), size=26, color=(255, 255, 255))

        if voice_listening:
            frame = put_text_kr(frame, "듣는 중...", (20, 50), size=28, color=(0, 0, 255))
        else:
            frame = put_text_kr(frame, f"PT 모드: {ex_sub_mode}", (20, 50), size=28, color=(0, 165, 255))

        frame = put_text_kr(frame, "[Q] 음성 명령  |  [SPACE] 핸드 모드", (20, h - 35), size=22, color=(200, 200, 200))

        # 우측 상단 기록 표시
        frame = draw_user_summary(frame, user_summary_cache)

        if pose_res.pose_landmarks and ex_sub_mode in ["Squat", "Pushup"]:
            lm = pose_res.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame = put_text_kr(frame, f"횟수: {ex_counter}", (20, 95), size=32, color=(255, 255, 0))

            active_model = squat_model if ex_sub_mode == "Squat" else pushup_model

            if active_model:
                pose_row = list(np.array([
                    [lm_.x, lm_.y, lm_.z, lm_.visibility] for lm_ in lm
                ]).flatten())
                prediction = active_model.predict([pose_row])[0]
                frame = put_text_kr(frame, f"AI: {prediction}", (20, 145), size=26, color=(255, 100, 255))

                if ex_sub_mode == "Squat":
                    if prediction == "Squat_Down" and ex_stage == "up":
                        ex_stage = "down"
                    elif prediction == "Squat_Up" and ex_stage == "down":
                        ex_stage = "up"; ex_counter += 1
                        speak(f"{ex_counter}회")
                    if ex_stage == "down":
                        frame = handle_posture_feedback(lm, frame)

                elif ex_sub_mode == "Pushup":
                    if prediction == "Pushup_Down" and ex_stage == "up":
                        ex_stage = "down"
                    elif prediction == "Pushup_Up" and ex_stage == "down":
                        ex_stage = "up"; ex_counter += 1
                        speak(f"{ex_counter}회")
            else:
                # 각도 폴백
                hip   = [lm[23].x, lm[23].y]
                knee  = [lm[25].x, lm[25].y]
                ankle = [lm[27].x, lm[27].y]
                angle = get_angle(hip, knee, ankle)
                frame = put_text_kr(frame, f"각도: {int(angle)}", (20, 145), size=26, color=(200, 200, 200))
                if angle < 95 and ex_stage == "up":
                    ex_stage = "down"
                if angle > 160 and ex_stage == "down":
                    ex_stage = "up"; ex_counter += 1
                    speak(f"{ex_counter}회")
                if ex_stage == "down":
                    frame = handle_posture_feedback(lm, frame)

    cv2.imshow(WIN_NAME, frame)

pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
