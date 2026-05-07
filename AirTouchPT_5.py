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
# 1. 한글 텍스트 렌더링 (PIL 사용)
# =============================================================
def _load_korean_font(size):
    candidates = ["malgun.ttf", "C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
    for path in candidates:
        if os.path.exists(path): return ImageFont.truetype(path, size)
    return ImageFont.load_default()

_font_cache = {}
def get_font(size):
    if size not in _font_cache: _font_cache[size] = _load_korean_font(size)
    return _font_cache[size]

def put_text_kr(frame, text, pos, size=28, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(size)
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =============================================================
# 2. Database System (기존과 동일)
# =============================================================
def init_db():
    with sqlite3.connect('fitness_records.db') as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE)')
        c.execute('CREATE TABLE IF NOT EXISTS workout_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_name TEXT, date TEXT, exercise_type TEXT, count INTEGER)')
        conn.commit()

def save_workout(exercise_type, count, user_name):
    if count > 0:
        with sqlite3.connect('fitness_records.db') as conn:
            c = conn.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            c.execute('INSERT INTO workout_logs (user_name, date, exercise_type, count) VALUES (?, ?, ?, ?)',
                      (user_name, now, exercise_type, count))
            conn.commit()

def get_user_summary(user_name):
    with sqlite3.connect('fitness_records.db') as conn:
        c = conn.cursor()
        c.execute('SELECT exercise_type, SUM(count), COUNT(*), MAX(date) FROM workout_logs WHERE user_name = ? GROUP BY exercise_type', (user_name,))
        rows = c.fetchall()
    return {row[0]: {'total': row[1], 'sessions': row[2], 'last': row[3][:10]} for row in rows}

init_db()

# =============================================================
# 3. 새로운 5종 통합 모델 로드 (핵심 수정부)
# =============================================================
try:
    with open('multi_fitness_model.pkl', 'rb') as f:
        multi_model = pickle.load(f)
    with open('feature_columns.txt', 'r') as f:
        feature_cols = f.read().split(',')
    print("[Model] 5종 통합 모델 로드 완료.")
except Exception as e:
    print(f"[Error] 모델 로드 실패: {e}")
    multi_model = None

# 거리 계산 함수 (모델용)
def calculate_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2) * 100

def get_dist_features(lm):
    l = lm.landmark
    dist_map = {
        'left_shoulder_left_wrist': calculate_dist(l[11], l[15]),
        'right_shoulder_right_wrist': calculate_dist(l[12], l[16]),
        'left_hip_left_ankle': calculate_dist(l[23], l[27]),
        'right_hip_right_ankle': calculate_dist(l[24], l[28]),
        'left_hip_left_wrist': calculate_dist(l[23], l[15]),
        'right_hip_right_wrist': calculate_dist(l[24], l[16]),
        'left_shoulder_left_ankle': calculate_dist(l[11], l[27]),
        'right_shoulder_right_ankle': calculate_dist(l[12], l[28]),
        'left_hip_right_wrist': calculate_dist(l[23], l[16]),
        'right_hip_left_wrist': calculate_dist(l[24], l[15]),
        'left_elbow_right_elbow': calculate_dist(l[13], l[14]),
        'left_knee_right_knee': calculate_dist(l[25], l[26]),
        'left_wrist_right_wrist': calculate_dist(l[15], l[16]),
        'left_ankle_right_ankle': calculate_dist(l[27], l[28]),
        'left_hip_avg_left_wrist_left_ankle': (calculate_dist(l[23], l[15]) + calculate_dist(l[23], l[27])) / 2,
        'right_hip_avg_right_wrist_right_ankle': (calculate_dist(l[24], l[16]) + calculate_dist(l[24], l[28])) / 2
    }
    return [dist_map[col] for col in feature_cols]

# =============================================================
# 4. TTS & Voice System (기존과 동일)
# =============================================================
pygame.mixer.init()
speech_queue = queue.Queue(maxsize=1)

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            tts = gTTS(text=text, lang='ko')
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f: tmp_path = f.name
            tts.save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.05)
            pygame.mixer.music.unload()
            os.remove(tmp_path)
        except: pass
        finally: speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()
def speak(text):
    while not speech_queue.empty():
        try: speech_queue.get_nowait(); speech_queue.task_done()
        except: break
    speech_queue.put(text)

# =============================================================
# 5. 자세 피드백 (스쿼트 전용)
# =============================================================
def check_posture(lm):
    ls, rs, lh, rh = lm[11], lm[12], lm[23], lm[24]
    if ls.y - rs.y > 0.05: return "오른쪽으로 기울었습니다."
    if ls.y - rs.y < -0.05: return "왼쪽으로 기울었습니다."
    return None

# =============================================================
# 6. Main Loop
# =============================================================
cap = cv2.VideoCapture(0)
sw, sh = pyautogui.size()
prev_x, prev_y = 0, 0
PT_MODE = False
ex_sub_mode = "Waiting" # Squat, Pushup, Pullup, Situp, JumpingJack
ex_counter = 0
ex_stage = 'up'
current_user = "Guest"
voice_listening = False
user_summary_cache = {}

WIN_NAME = 'Ultimate AI Trainer'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

# 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe 처리
    mp_pose = mp.solutions.pose
    pose_res = mp_pose.Pose(min_detection_confidence=0.5).process(rgb_frame)
    mp_hands = mp.solutions.hands
    hand_res = mp_hands.Hands(max_num_hands=1).process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    if key == ord(' '):
        PT_MODE = not PT_MODE
        speak("바디 모드" if PT_MODE else "핸드 모드")

    # --- 핸드 모드 ---
    if not PT_MODE:
        frame = put_text_kr(frame, "핸드 모드 (Space로 전환)", (20, 20), color=(0,255,0))
        if hand_res.multi_hand_landmarks:
            landmarks = hand_res.multi_hand_landmarks[0]
            idx = landmarks.landmark[8]
            tx = np.interp(idx.x, [0.2, 0.8], [0, sw])
            ty = np.interp(idx.y, [0.2, 0.8], [0, sh])
            pyautogui.moveTo(prev_x + (tx-prev_x)*0.2, prev_y + (ty-prev_y)*0.2)
            prev_x, prev_y = tx, ty
            if math.sqrt((landmarks.landmark[12].x - landmarks.landmark[4].x)**2 + (landmarks.landmark[12].y - landmarks.landmark[4].y)**2) < 0.05:
                pyautogui.click()

    # --- 바디/PT 모드 ---
    else:
        frame = put_text_kr(frame, f"사용자: {current_user} | 운동: {ex_sub_mode}", (20, 20))
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp_pose.POSE_CONNECTIONS)
            
            if multi_model and ex_sub_mode != "Waiting":
                # 거리 데이터 추출 및 예측
                features = get_dist_features(lm)
                prob = multi_model.predict_proba([features])[0]
                idx = np.argmax(prob)
                prediction = multi_model.classes_[idx]
                conf = prob[idx]

                # 카운팅 로직 (동적으로 정답 이름 분석)
                # 예: prediction이 'squats_down' 이고 현재 운동이 'Squat'인 경우
                current_ex_lower = ex_sub_mode.lower().replace(" ", "")
                if current_ex_lower in prediction:
                    if "_down" in prediction and ex_stage == "up" and conf > 0.7:
                        ex_stage = "down"
                    elif "_up" in prediction and ex_stage == "down" and conf > 0.7:
                        ex_stage = "up"
                        ex_counter += 1
                        speak(f"{ex_counter}회")
                
                frame = put_text_kr(frame, f"동작: {prediction} ({conf*100:.0f}%)", (20, 60), color=(255,255,0))
                frame = put_text_kr(frame, f"횟수: {ex_counter}", (20, 100), size=40, color=(0,255,255))

        # Q키 음성인식 (기존 로직 유지)
        if key == ord('q'):
             # (이 부분에 기존의 trigger_voice_recognition 스레드 실행 로직 동일하게 삽입)
             # 편의상 '스쿼트', '푸시업' 등 음성 명령으로 ex_sub_mode를 바꾼다고 가정
             pass

    cv2.imshow(WIN_NAME, frame)

cap.release()
cv2.destroyAllWindows()
