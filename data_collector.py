import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np

# --- 설정 ---
CSV_FILE = 'pose_dataset.csv'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# CSV 파일 초기 세팅
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        landmarks = ['class']
        for val in range(1, 34):
            landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
        csv_writer.writerow(landmarks)

cap = cv2.VideoCapture(0)

# 상태 변수
recording_class = None
countdown_start_time = 0
frames_collected = 0
TARGET_FRAMES = 50  # 한 번에 수집할 데이터 개수

print("데이터 수집기 2.0 실행 완료.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # 기본 UI
        cv2.putText(frame, "1: Squat_Down | 2: Squat_Up | 3: Idle | ESC: Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 키 입력 감지 (수집 중이 아닐 때만)
        key = cv2.waitKey(1) & 0xFF
        if recording_class is None:
            if key == ord('1'):
                recording_class = "Squat_Down"
                countdown_start_time = time.time()
                frames_collected = 0
            elif key == ord('2'):
                recording_class = "Squat_Up"
                countdown_start_time = time.time()
                frames_collected = 0
            elif key == ord('3'):
                recording_class = "Idle"
                countdown_start_time = time.time()
                frames_collected = 0
            elif key == 27:
                break

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # --- 타이머 및 자동 수집 로직 ---
            if recording_class is not None:
                elapsed_time = time.time() - countdown_start_time
                
                # 1. 3초 카운트다운 대기 (자세 잡을 시간)
                if elapsed_time < 3.0:
                    remain_sec = int(3.0 - elapsed_time) + 1
                    cv2.putText(frame, f"Get Ready: {remain_sec}", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.putText(frame, f"Target: {recording_class}", (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 2. 카운트다운 종료 후 자동 수집 시작
                elif frames_collected < TARGET_FRAMES:
                    pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten())
                    pose_row.insert(0, recording_class)
                    
                    with open(CSV_FILE, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(pose_row)
                    
                    frames_collected += 1
                    
                    # 찰칵거리는 효과 및 진행도
                    if frames_collected % 5 == 0:
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
                    cv2.putText(frame, f"Recording... {frames_collected}/{TARGET_FRAMES}", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 3. 수집 완료 시 초기화
                else:
                    recording_class = None
                    cv2.putText(frame, "DONE!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv2.imshow('Smart Data Collector', frame)

cap.release()
cv2.destroyAllWindows()
