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

# --- Environment ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

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
    """
    유저의 전체 운동 기록 집계
    반환: { 'Squat': {'total': 150, 'sessions': 5, 'last': '2024-01-15'}, ... }
    재실행 후에도 DB에서 읽어오므로 기록이 유지됨
    """
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
    """로그인 시 콘솔 + TTS로 기록 안내"""
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
# speak() 호출마다 큐를 비우고 최신 텍스트 1개만 유지
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
user_summary_cache = {}  # 로그인 시 갱신, 화면 표시용

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

    if ls_y - rs_y > 0.04: return "오른쪽으로 기울었습니다. 부상 위험이 있습니다."
    if ls_y - rs_y < -0.04: return "왼쪽으로 기울었습니다. 다칠 위험이 있습니다."
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
        cv2.putText(frame, f"! {feedback}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Good posture", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        last_feedback_msg = ""

# =============================================================
# 화면 우측 상단 — 유저 누적 기록 표시
# =============================================================
def draw_user_summary(frame, summary):
    """
    재실행 후에도 DB에서 읽은 기록을 화면에 표시
    summary: get_user_summary() 반환값 딕셔너리
    """
    if not summary:
        return
    x = frame.shape[1] - 290
    y = 50
    cv2.putText(frame, "[ Record ]", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    for i, (ex, data) in enumerate(summary.items()):
        line1 = f"{ex}: {data['total']} reps / {data['sessions']} sessions"
        line2 = f"  Last: {data['last']}"
        cv2.putText(frame, line1, (x, y + 28 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)
        cv2.putText(frame, line2, (x, y + 28 * (i + 1) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 150, 255), 1)

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
                    # ★ 핵심: 로그인 시 DB에서 기록 읽어서 캐시 갱신 + TTS 안내
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
                user_summary_cache = get_user_summary(current_user)  # 저장 후 캐시 즉시 갱신
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

# --- Main Loop ---
cap = cv2.VideoCapture(0)
sw, sh = pyautogui.size()
prev_x, prev_y = 0, 0

print("[System] Starting — Barrier-Free AI PT")
speak("오디오 피티 시작. 스페이스바를 눌러 모드를 바꾸세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_res = hands.process(rgb_frame)
    pose_res = pose.process(rgb_frame)

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
        cv2.putText(frame, "HAND MODE  |  [SPACE] Body Mode", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if hand_res.multi_hand_landmarks:
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
        cv2.putText(frame, f"User: {current_user}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        status_txt = "LISTENING..." if voice_listening else f"PT: {ex_sub_mode}"
        color = (0, 0, 255) if voice_listening else (0, 165, 255)
        cv2.putText(frame, status_txt, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "[Q] Voice  |  [SPACE] Hand Mode", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # ★ 화면 우측 상단에 DB에서 읽은 누적 기록 표시
        draw_user_summary(frame, user_summary_cache)

        if pose_res.pose_landmarks and ex_sub_mode in ["Squat", "Pushup"]:
            lm = pose_res.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Count: {ex_counter}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            active_model = squat_model if ex_sub_mode == "Squat" else pushup_model

            if active_model:
                pose_row = list(np.array([
                    [lm_.x, lm_.y, lm_.z, lm_.visibility] for lm_ in lm
                ]).flatten())
                prediction = active_model.predict([pose_row])[0]
                cv2.putText(frame, f"AI: {prediction}", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)

                if ex_sub_mode == "Squat":
                    if prediction == "Squat_Down" and ex_stage == "up":
                        ex_stage = "down"
                    elif prediction == "Squat_Up" and ex_stage == "down":
                        ex_stage = "up"; ex_counter += 1
                        speak(f"{ex_counter}회")
                    if ex_stage == "down":
                        handle_posture_feedback(lm, frame)

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
                cv2.putText(frame, f"Angle: {int(angle)}", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                if angle < 95 and ex_stage == "up":
                    ex_stage = "down"
                if angle > 160 and ex_stage == "down":
                    ex_stage = "up"; ex_counter += 1
                    speak(f"{ex_counter}회")
                if ex_stage == "down":
                    handle_posture_feedback(lm, frame)

    cv2.imshow('Barrier-Free AI PT', frame)

pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
