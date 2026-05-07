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
from gtts import gTTS

# --- Environment ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# =============================================================
# TTS 시스템 - gTTS + pygame
# =============================================================
# pyttsx3 문제:
#   Windows COM 기반이라 두 번째 runAndWait()부터 무음이 되는 버그
#
# 새 구조:
#   speak(text) → 큐 비우기 → 새 텍스트 1개 넣기
#   워커        → gTTS로 mp3 생성 → pygame으로 재생 → 반복
#
#   gTTS는 매 호출마다 새 파일을 만들어 재생하므로
#   두 번째 이후 무음 버그가 원천 차단됨
#   인터넷 연결 필요 (구글 TTS API 사용)
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
    # 큐에 쌓인 것 전부 버리고 최신 텍스트 1개만 유지
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

# 자세 교정 피드백 변수
# 같은 피드백이 연속으로 나오지 않도록 마지막 피드백 내용과 시간을 기록
last_feedback_msg = ""
last_feedback_time = 0
FEEDBACK_COOLDOWN = 4.0  # 같은 피드백 최소 4초 간격

# --- MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def check_posture_feedback(lm):
    """
    스쿼트 자세 교정 감지 함수
    감지 항목:
      1. 어깨 기울기  - 좌우 어깨 y좌표 차이로 몸통 기울기 판단
      2. 엉덩이 기울기 - 좌우 엉덩이 y좌표 차이로 하체 기울기 판단
      3. 무릎 안쪽 쏠림 - 무릎 x좌표가 발목보다 지나치게 안쪽으로 모이면 감지

    좌표계 주의:
      MediaPipe x좌표는 화면이 flip된 상태 기준
      y좌표는 위쪽이 0, 아래쪽이 1
      왼쪽 어깨 y > 오른쪽 어깨 y  →  왼쪽이 더 아래  →  왼쪽으로 기울어짐

    반환값: 피드백 메시지 문자열 or None (정상)
    """
    # landmark 인덱스
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP,      R_HIP      = 23, 24
    L_KNEE,     R_KNEE     = 25, 26
    L_ANKLE,    R_ANKLE    = 27, 28

    ls_y = lm[L_SHOULDER].y;  rs_y = lm[R_SHOULDER].y
    lh_y = lm[L_HIP].y;       rh_y = lm[R_HIP].y
    lk_x = lm[L_KNEE].x;      rk_x = lm[R_KNEE].x
    la_x = lm[L_ANKLE].x;     ra_x = lm[R_ANKLE].x

    TILT_THRESHOLD  = 0.04  # 어깨/엉덩이 기울기 감지 임계값 (좌표 단위)
    KNEE_THRESHOLD  = 0.06  # 무릎 안쪽 쏠림 감지 임계값

    # 1. 어깨 기울기
    shoulder_diff = ls_y - rs_y
    if shoulder_diff > TILT_THRESHOLD:
        return "왼쪽으로 기울었습니다."
    if shoulder_diff < -TILT_THRESHOLD:
        return "오른쪽으로 기울었습니다."

    # 2. 엉덩이 기울기
    hip_diff = lh_y - rh_y
    if hip_diff > TILT_THRESHOLD:
        return "엉덩이가 왼쪽으로 기울었습니다."
    if hip_diff < -TILT_THRESHOLD:
        return "엉덩이가 오른쪽으로 기울었습니다."

    # 3. 무릎 안쪽 쏠림 (무릎이 발목보다 안쪽으로 많이 모임)
    if (lk_x - la_x) > KNEE_THRESHOLD:
        return "왼쪽 무릎이 안으로 쏠렸습니다."
    if (ra_x - rk_x) > KNEE_THRESHOLD:
        return "오른쪽 무릎이 안으로 쏠렸습니다."

    return None  # 정상

# --- Voice Recognition ---
def trigger_voice_recognition():
    global ex_sub_mode, ex_counter, voice_listening
    voice_listening = True
    r = sr.Recognizer()

    with sr.Microphone() as source:
        speak("종목을 말씀하세요.")
        try:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            cmd = r.recognize_google(audio, language='ko-KR')
            print(f"[Voice] Recognized: '{cmd}'")

            if "스쿼트" in cmd:
                ex_sub_mode = "Squat"; ex_counter = 0
                speak("스쿼트 모드 시작.")
            elif "종료" in cmd or "그만" in cmd:
                ex_sub_mode = "Waiting"
                speak("운동 종료.")
            else:
                speak(f"{cmd}? 다시 말씀해주세요.")

        except sr.UnknownValueError:
            print("[Voice] Recognition failed")
            speak("인식에 실패했습니다.")
        except sr.WaitTimeoutError:
            print("[Voice] Timeout")
            speak("시간이 초과됐습니다.")
        except Exception as e:
            print(f"[Voice] Error: {e}")
            speak("오류가 발생했습니다.")
        finally:
            voice_listening = False
            print("[Voice] Mic deactivated")

# --- Main Loop ---
cap = cv2.VideoCapture(0)
sw, sh = pyautogui.size()
prev_x, prev_y = 0, 0

speak("오디오 피티 시작. 스페이스를 눌러 모드를 바꾸세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_res = hands.process(rgb_frame)
    pose_res = pose.process(rgb_frame)

    # --- Key Input ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break  # ESC quit

    # SPACE: toggle Hand <-> Body mode
    if key == ord(' ') and time.time() - last_mode_switch_time > 2.0:
        PT_MODE = not PT_MODE
        last_mode_switch_time = time.time()
        print(f"[Mode] Switched to {'Body' if PT_MODE else 'Hand'} mode")
        speak("바디 모드 전환." if PT_MODE else "핸드 모드 전환.")

    # Q: voice recognition (Body mode only)
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

    # --- [Mode 2] Body Motion (no hand mouse) ---
    else:
        status_txt = "LISTENING..." if voice_listening else f"PT MODE: {ex_sub_mode}"
        color = (0, 0, 255) if voice_listening else (0, 165, 255)
        cv2.putText(frame, status_txt, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "[Q] Voice Command  |  [SPACE] Hand Mode", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if ex_sub_mode == "Squat":
                lm = pose_res.pose_landmarks.landmark
                hip   = [lm[23].x, lm[23].y]
                knee  = [lm[25].x, lm[25].y]
                ankle = [lm[27].x, lm[27].y]
                angle = get_angle(hip, knee, ankle)

                cv2.putText(frame, f"Count: {ex_counter}  Angle: {int(angle)}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # --- 자세 교정 피드백 (내려가는 동작 중에만 체크) ---
                if ex_stage == "down":
                    feedback = check_posture_feedback(lm)
                    now = time.time()
                    if feedback:
                        # 다른 피드백이거나 쿨다운이 지났을 때만 TTS 출력
                        if feedback != last_feedback_msg or (now - last_feedback_time) > FEEDBACK_COOLDOWN:
                            print(f"[Posture] {feedback}")
                            speak(feedback)
                            last_feedback_msg = feedback
                            last_feedback_time = now
                        # 화면에 빨간 경고 표시
                        cv2.putText(frame, f"! {feedback}", (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # 정상 자세 - 초록 표시
                        cv2.putText(frame, "Good posture", (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        last_feedback_msg = ""  # 정상이면 메시지 초기화

                # --- 카운트 로직 ---
                if angle < 95 and ex_stage == "up":
                    ex_stage = "down"
                if angle > 160 and ex_stage == "down":
                    ex_stage = "up"
                    ex_counter += 1
                    speak(f"{ex_counter}회")

    cv2.imshow('Audio PT Master', frame)

pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
