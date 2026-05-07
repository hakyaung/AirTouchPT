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
import difflib
import re
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
# 2. Database System
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
        speak(f"{user_name}님, 환영합니다. 첫 운동을 시작해보세요.")
        return
    parts = [f"{ex} 총 {d['total']}회" for ex, d in summary.items()]
    speak(f"{user_name}님, 환영합니다. 기존 기록: {', '.join(parts)}.")

init_db()

# =============================================================
# 3. ML 모델 로드
# =============================================================
squat_model = None
pushup_model = None
legraise_model = None
pullup_model = None
situp_model = None
barbellcurl_model = None
hammercurl_model = None
sidelateralraise_model = None

try:
    with open('squat_model.pkl', 'rb') as f: squat_model = pickle.load(f)
    print("[Model] squat_model.pkl loaded.")
except: pass

try:
    with open('pushup_model.pkl', 'rb') as f: pushup_model = pickle.load(f)
    print("[Model] pushup_model.pkl loaded.")
except: pass

try:
    with open('legraise_model.pkl', 'rb') as f: legraise_model = pickle.load(f)
    print("[Model] legraise_model.pkl loaded. (3D)")
except: pass

try:
    with open('pullup_model.pkl', 'rb') as f: pullup_model = pickle.load(f)
    print("[Model] pullup_model.pkl loaded. (3D)")
except: pass

try:
    with open('situp_model.pkl', 'rb') as f: situp_model = pickle.load(f)
    print("[Model] situp_model.pkl loaded. (3D)")
except: pass

try:
    with open('barbellcurl_model.pkl', 'rb') as f: barbellcurl_model = pickle.load(f)
    print("[Model] barbellcurl_model.pkl loaded. (3D)")
except: pass

try:
    with open('hammercurl_model.pkl', 'rb') as f: hammercurl_model = pickle.load(f)
    print("[Model] hammercurl_model.pkl loaded. (3D)")
except: pass

try:
    with open('sidelateralraise_model.pkl', 'rb') as f: sidelateralraise_model = pickle.load(f)
    print("[Model] sidelateralraise_model.pkl loaded. (3D)")
except: pass

# =============================================================
# 4. 수학 유틸리티
# =============================================================
def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# =============================================================
# 5. TTS System
# =============================================================
pygame.mixer.init()
speech_queue = queue.Queue(maxsize=1)
voice_listening = False

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
    global voice_listening
    if voice_listening: return
    print(f"[TTS] {text}")
    while not speech_queue.empty():
        try: speech_queue.get_nowait(); speech_queue.task_done()
        except queue.Empty: break
    try: speech_queue.put_nowait(text)
    except queue.Full: pass

# =============================================================
# 6. 자세 교정 모듈
# =============================================================
last_feedback_msg, last_feedback_time, FEEDBACK_COOLDOWN = "", 0, 4.0
min_knee_dist = 0.0  

def squat_feedback(lm_list, stage):
    global min_knee_dist
    lk = [lm_list[25].x, lm_list[25].y]
    rk = [lm_list[26].x, lm_list[26].y]
    ls_y, rs_y = lm_list[11].y, lm_list[12].y
    current_knee_dist = get_dist(lk, rk)

    if stage == "down":
        min_knee_dist = current_knee_dist 
    elif stage == "up" and min_knee_dist > 0:
        if current_knee_dist < (min_knee_dist * 0.55):
            return "무릎이 안으로 모입니다. 무릎을 바깥으로 벌려주세요."

    if abs(ls_y - rs_y) > 0.07:  return "상체가 한쪽으로 기울었습니다. 중심을 잡으세요."
    return None

def pushup_feedback(lm_list):
    ls, rs = [lm_list[11].x, lm_list[11].y], [lm_list[12].x, lm_list[12].y]
    le, re = [lm_list[13].x, lm_list[13].y], [lm_list[14].x, lm_list[14].y]
    lw, rw = [lm_list[15].x, lm_list[15].y], [lm_list[16].x, lm_list[16].y]
    
    l_angle = get_angle(ls, le, lw)
    r_angle = get_angle(rs, re, rw)
    if abs(l_angle - r_angle) > 35:
        return "양팔의 굽힘이 다릅니다. 좌우 균형을 맞춰주세요."

    lh, rh = [lm_list[23].x, lm_list[23].y], [lm_list[24].x, lm_list[24].y]
    la, ra = [lm_list[27].x, lm_list[27].y], [lm_list[28].x, lm_list[28].y]
    
    hip_angle_left = get_angle(ls, lh, la)
    hip_angle_right = get_angle(rs, rh, ra)
    if hip_angle_left < 145 or hip_angle_right < 145:
        return "엉덩이가 너무 내려갔습니다. 복부에 힘을 주세요."
    return None

def pullup_feedback(lm_list):
    ls_y, rs_y = lm_list[11].y, lm_list[12].y
    if abs(ls_y - rs_y) > 0.06:
        return "한쪽 어깨가 너무 올라갔습니다. 양쪽 어깨 높이를 맞춰주세요."
    return None

def sidelateralraise_feedback(lm_list):
    ls_y, rs_y = lm_list[11].y, lm_list[12].y
    lw_y, rw_y = lm_list[15].y, lm_list[16].y
    
    if abs(ls_y - rs_y) > 0.06:
        return "어깨가 한쪽으로 기울었습니다. 양쪽 어깨를 평행하게 맞춰주세요."
    if abs(lw_y - rw_y) > 0.10:
        return "양손의 높이가 다릅니다. 균형을 맞춰서 들어올려주세요."
    return None

# =============================================================
# 7. 초고속 STT + 스마트 자연어 처리 모듈
# =============================================================
EXERCISE_MAP = {
    "스쿼트": "Squat",   "푸시업": "Pushup",  "팔굽혀펴기": "Pushup", 
    "풀업":   "Pullup",  "턱걸이": "Pullup", 
    "싯업":   "Situp",   "윗몸일으키기": "Situp", 
    "점핑잭": "JumpingJack", "레그레이즈": "LegRaise", "다리올리기": "LegRaise",
    "바벨컬": "BarbellCurl", "바벨": "BarbellCurl", "이두": "BarbellCurl", "이두컬": "BarbellCurl",
    "해머컬": "HammerCurl", "해머": "HammerCurl", "망치": "HammerCurl",
    "사이드레터럴레이즈": "SideLateralRaise", "사레레": "SideLateralRaise", "사이드": "SideLateralRaise", "어깨운동": "SideLateralRaise"
}
KOR_NAME_MAP = {
    "Squat": "스쿼트", "Pushup": "푸시업", "Pullup": "풀업", 
    "Situp": "윗몸일으키기", "JumpingJack": "점핑잭", "LegRaise": "레그레이즈", 
    "BarbellCurl": "바벨컬", "HammerCurl": "해머컬", "SideLateralRaise": "사이드레터럴레이즈"
}

def listen_to_voice(r, source):
    try:
        audio = r.listen(source, timeout=4, phrase_time_limit=4)
        return r.recognize_google(audio, language='ko-KR')
    except: return None

def trigger_voice_recognition():
    global ex_sub_mode, ex_counter, ex_stage, voice_listening, current_user, user_summary_cache, min_knee_dist
    
    voice_listening = True
    r = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            cmd = listen_to_voice(r, source)
            
        voice_listening = False 
        if not cmd:
            speak("인식하지 못했습니다.")
            return 
            
        print(f"[Voice] 인식된 문장: '{cmd}'")
        cmd_clean = cmd.replace(" ", "") 

        if "등록" in cmd or "가입" in cmd:
            name = re.sub(r'(등록|가입|해줘|시켜줘|으로|이라는|이름|할게)', '', cmd_clean)
            if name:
                with sqlite3.connect('fitness_records.db') as conn:
                    try:
                        conn.cursor().execute("INSERT INTO users (name) VALUES (?)", (name,))
                        conn.commit()
                        current_user = name
                        user_summary_cache = get_user_summary(current_user)
                        speak(f"{name}님, 환영합니다. 등록되었습니다.")
                    except: speak(f"{name}님은 이미 등록된 이름입니다.")
            else: speak("누구 이름으로 등록할지 말씀해주세요.")

        elif "로그인" in cmd or "접속" in cmd or "로그" in cmd:
            name = re.sub(r'(로그인|접속|로그|해줘|시켜줘|으로|이라는|이름|할게)', '', cmd_clean)
            if name:
                with sqlite3.connect('fitness_records.db') as conn:
                    result = conn.cursor().execute("SELECT name FROM users WHERE name=?", (name,)).fetchone()
                if result:
                    current_user = result[0]
                    user_summary_cache = get_user_summary(current_user)
                    announce_user_summary(current_user)
                else: speak(f"{name}님은 등록되지 않은 사용자입니다.")
            else: speak("누구 이름으로 로그인할지 말씀해주세요.")

        elif "로그아웃" in cmd:
            if current_user != "Guest":
                speak(f"{current_user}님, 로그아웃 되었습니다.")
                current_user = "Guest"; user_summary_cache = {}
            else: speak("현재 로그인된 사용자가 없습니다.")
                
        elif "종료" in cmd or "그만" in cmd or "저장" in cmd:
            if ex_sub_mode != "Waiting" and ex_counter > 0:
                save_workout(ex_sub_mode, ex_counter, current_user)
                user_summary_cache = get_user_summary(current_user)
            ex_sub_mode = "Waiting"; ex_counter = 0; speak("기록을 저장했습니다.")

        else:
            matched_ex = None
            words = cmd.split()
            for w in words:
                matches = difflib.get_close_matches(w, EXERCISE_MAP.keys(), n=1, cutoff=0.5)
                if matches:
                    matched_ex = EXERCISE_MAP[matches[0]]
                    print(f"[NLP] '{w}' -> '{matches[0]}' 매칭!")
                    break
            
            if not matched_ex:
                for key in EXERCISE_MAP.keys():
                    if key in cmd_clean:
                        matched_ex = EXERCISE_MAP[key]
                        break

            if matched_ex:
                if ex_sub_mode != "Waiting" and ex_counter > 0:
                    save_workout(ex_sub_mode, ex_counter, current_user)
                    user_summary_cache = get_user_summary(current_user)
                ex_sub_mode = matched_ex
                ex_counter = 0
                ex_stage = "up"
                min_knee_dist = 0.0 
                speak(f"{current_user}님의 {KOR_NAME_MAP.get(matched_ex, matched_ex)} 시작합니다.")
            else: speak("어떤 운동을 할지 다시 말씀해주세요.")
            
    finally:
        voice_listening = False

# =============================================================
# 8. MediaPipe & 상태 변수
# =============================================================
mp_hands_mod, mp_pose_mod, mp_draw = mp.solutions.hands, mp.solutions.pose, mp.solutions.drawing_utils
hands_detector = mp_hands_mod.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose_detector  = mp_pose_mod.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

PT_MODE, last_mode_switch_t = False, 0
ex_sub_mode, ex_counter, ex_stage = "Waiting", 0, 'up'
current_user, user_summary_cache = "Guest", {}
sw, sh, prev_x, prev_y = pyautogui.size()[0], pyautogui.size()[1], 0, 0

# =============================================================
# 9. Main Loop
# =============================================================
cap = cv2.VideoCapture(0)
WIN_NAME = 'AirTouchPT'
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
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        with speech_queue.mutex:
            speech_queue.queue.clear()
        threading.Thread(target=trigger_voice_recognition, daemon=True).start()

    text_items = []

    if not PT_MODE:
        text_items.append(("핸드 모드  |  [SPACE] 바디 모드", (20, 15), 26, (0, 255, 0)))
        if hand_res.multi_hand_landmarks:
            lm = hand_res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands_mod.HAND_CONNECTIONS)
            tx, ty = np.interp(lm.landmark[8].x, [0.2, 0.8], [0, sw]), np.interp(lm.landmark[8].y, [0.2, 0.8], [0, sh])
            cx, cy = prev_x + (tx - prev_x) * 0.2, prev_y + (ty - prev_y) * 0.2
            pyautogui.moveTo(cx, cy); prev_x, prev_y = cx, cy
            if math.sqrt((lm.landmark[12].x - lm.landmark[4].x)**2 + (lm.landmark[12].y - lm.landmark[4].y)**2) < 0.05: pyautogui.click()

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
            active_model = None
            
            # 모델 맵핑
            if ex_sub_mode == "Squat": active_model = squat_model
            elif ex_sub_mode == "Pushup": active_model = pushup_model
            elif ex_sub_mode == "LegRaise": active_model = legraise_model
            elif ex_sub_mode == "Pullup": active_model = pullup_model
            elif ex_sub_mode == "Situp": active_model = situp_model
            elif ex_sub_mode == "BarbellCurl": active_model = barbellcurl_model
            elif ex_sub_mode == "HammerCurl": active_model = hammercurl_model
            elif ex_sub_mode == "SideLateralRaise": active_model = sidelateralraise_model
            
            if active_model:
                if ex_sub_mode in ["LegRaise", "Pullup", "Situp", "BarbellCurl", "HammerCurl", "SideLateralRaise"] and pose_res.pose_world_landmarks:
                    target_lm = pose_res.pose_world_landmarks.landmark
                else:
                    target_lm = lm_list
                
                pose_row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in target_lm]).flatten())
                try:
                    prob = active_model.predict_proba([pose_row])[0]
                    max_idx = np.argmax(prob)
                    confidence = prob[max_idx]
                    prediction = active_model.classes_[max_idx]
                    
                    if confidence > 0.70:
                        pred_lower = prediction.lower().replace(" ", "")
                        
                        # 물리적 각도 가드 계산 (2D 화면 좌표 활용)
                        ls, rs = [lm_list[11].x, lm_list[11].y], [lm_list[12].x, lm_list[12].y]
                        lh, rh = [lm_list[23].x, lm_list[23].y], [lm_list[24].x, lm_list[24].y]
                        la, ra = [lm_list[27].x, lm_list[27].y], [lm_list[28].x, lm_list[28].y]
                        le, re = [lm_list[13].x, lm_list[13].y], [lm_list[14].x, lm_list[14].y]
                        lw, rw = [lm_list[15].x, lm_list[15].y], [lm_list[16].x, lm_list[16].y]
                        lk, rk = [lm_list[25].x, lm_list[25].y], [lm_list[26].x, lm_list[26].y]
                        
                        left_hip_angle = get_angle(ls, lh, la)
                        right_hip_angle = get_angle(rs, rh, ra)
                        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
                        
                        left_elbow_angle = get_angle(ls, le, lw)
                        right_elbow_angle = get_angle(rs, re, rw)
                        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
                        
                        left_hip_knee_angle = get_angle(ls, lh, lk)
                        right_hip_knee_angle = get_angle(rs, rh, rk)
                        avg_hip_knee_angle = (left_hip_knee_angle + right_hip_knee_angle) / 2

                        left_arm_raise_angle = get_angle(lh, ls, le)
                        right_arm_raise_angle = get_angle(rh, rs, re)
                        avg_arm_raise_angle = (left_arm_raise_angle + right_arm_raise_angle) / 2

                        # 🌟 [수정] 스쿼트: 양쪽 무릎 각도를 모두 계산한 후 가장 확실히 접히는 무릎 선택
                        left_knee_angle = get_angle([lm_list[23].x, lm_list[23].y], [lm_list[25].x, lm_list[25].y], [lm_list[27].x, lm_list[27].y])
                        right_knee_angle = get_angle([lm_list[24].x, lm_list[24].y], [lm_list[26].x, lm_list[26].y], [lm_list[28].x, lm_list[28].y])
                        active_knee_angle = min(left_knee_angle, right_knee_angle)

                        is_physically_down = True
                        is_physically_up = True

                        # 🌟 스쿼트 ML 전용 이중 잠금장치 추가
                        if ex_sub_mode == "Squat":
                            is_physically_down = active_knee_angle < 110
                            is_physically_up = active_knee_angle > 150
                            
                        elif ex_sub_mode == "LegRaise":
                            is_physically_down = avg_hip_angle > 140  
                            is_physically_up = avg_hip_angle < 130    
                        
                        elif ex_sub_mode == "Pullup":
                            arms_up = (lw[1] < ls[1]) and (rw[1] < rs[1])
                            is_physically_down = (avg_elbow_angle > 130) and arms_up
                            is_physically_up = (avg_elbow_angle < 105) and arms_up
                            
                        elif ex_sub_mode == "Situp":
                            is_physically_down = avg_hip_knee_angle > 90
                            is_physically_up = avg_hip_knee_angle < 65
                            
                        elif ex_sub_mode in ["BarbellCurl", "HammerCurl"]:
                            is_physically_down = avg_elbow_angle > 140
                            is_physically_up = avg_elbow_angle < 70
                            
                        elif ex_sub_mode == "SideLateralRaise":
                            is_physically_down = avg_arm_raise_angle < 45
                            is_physically_up = avg_arm_raise_angle > 70

                        if "_down" in pred_lower and ex_stage == "up" and is_physically_down:
                            ex_stage = "down"
                            speak("내려갔습니다.")
                        elif "_up" in pred_lower and ex_stage == "down" and is_physically_up:
                            ex_stage = "up"
                            ex_counter += 1
                            speak(f"{ex_counter}회")

                    conf_color = (255, 100, 255) if confidence > 0.70 else (100, 100, 200)
                    text_items.append((f"AI: {prediction} ({confidence*100:.0f}%)", (20, 143), 24, conf_color))
                    
                except Exception as e:
                    text_items.append((f"모델 오류: {e}", (20, 143), 24, (100, 100, 200)))

            # 🌟 [수정] ML 모델 없이 동작하는 스쿼트 Fallback 로직 완벽 보강
            if prediction is None and ex_sub_mode == "Squat":
                left_knee_angle = get_angle([lm_list[23].x, lm_list[23].y], [lm_list[25].x, lm_list[25].y], [lm_list[27].x, lm_list[27].y])
                right_knee_angle = get_angle([lm_list[24].x, lm_list[24].y], [lm_list[26].x, lm_list[26].y], [lm_list[28].x, lm_list[28].y])
                active_knee_angle = min(left_knee_angle, right_knee_angle)

                text_items.append((f"무릎 각도: {int(active_knee_angle)}", (20, 143), 24, (200, 200, 200)))
                
                if active_knee_angle < 110 and ex_stage == "up":  
                    ex_stage = "down"
                    speak("내려갔습니다.")
                if active_knee_angle > 150 and ex_stage == "down":
                    ex_stage = "up"
                    ex_counter += 1
                    speak(f"{ex_counter}회")

            feedback = None
            if ex_sub_mode == "Squat":
                feedback = squat_feedback(lm_list, ex_stage)
            elif ex_sub_mode == "Pushup":
                feedback = pushup_feedback(lm_list)
            elif ex_sub_mode == "Pullup":
                feedback = pullup_feedback(lm_list)
            elif ex_sub_mode == "SideLateralRaise":
                feedback = sidelateralraise_feedback(lm_list)
            
            now = time.time()
            if feedback:
                if feedback != last_feedback_msg or (now - last_feedback_time) > FEEDBACK_COOLDOWN:
                    speak(feedback)
                    last_feedback_msg = feedback
                    last_feedback_time = now
                text_items.append((f"! {feedback}", (20, 200), 24, (0, 0, 255)))
            else:
                text_items.append(("Good posture", (20, 200), 24, (0, 255, 0)))

    frame = draw_text_batch(frame, text_items)
    cv2.imshow(WIN_NAME, frame)

pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
