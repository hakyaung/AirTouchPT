import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
import mss
import csv

# --- 환경 설정 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================================
# ⚙️ 학습 설정 (만능 조종석)
# ==========================================================
EXERCISE_NAME = "test"  # 원하는 운동 이름으로 변경하세요

CSV_FILE = f'{EXERCISE_NAME.lower()}_data.csv'
MODEL_FILE = f'{EXERCISE_NAME.lower()}_model.pkl'
CLASS_DOWN = f'{EXERCISE_NAME}_Down'
CLASS_UP = f'{EXERCISE_NAME}_Up'
# ==========================================================

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def init_csv():
    if not os.path.exists(CSV_FILE):
        headers = ['class']
        for i in range(33):
            headers.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def save_data(class_name, world_landmarks):
    # 🚨 [핵심 수정] 2D 픽셀 좌표가 아닌 골반 중심의 3D 절대 좌표(World Landmarks)를 저장합니다!
    row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in world_landmarks]).flatten())
    row.insert(0, class_name)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def train_model():
    print(f"\n🚀 [{EXERCISE_NAME} 학습 시작] 데이터를 바탕으로 AI를 훈련합니다...")
    if not os.path.exists(CSV_FILE):
        print("❌ 데이터가 없습니다. 먼저 1번, 2번을 눌러 데이터를 수집해 주세요.")
        return

    df = pd.read_csv(CSV_FILE)
    if len(df) < 10:
        print("❌ 데이터가 너무 적습니다. 최소 10장 이상 수집한 후 다시 시도하세요.")
        return

    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("="*40)
    print(f"🎉 학습 완료! 모의고사 정확도: {acc * 100:.2f}%")
    print("="*40)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"✨ '{MODEL_FILE}' 파일이 성공적으로 생성되었습니다!\n")

def main():
    # 이전에 잘못 모았던 2D 데이터 파일이 있으면 꼬일 수 있으니, 실행 시 파일 존재하면 경고 띄우기
    if os.path.exists(CSV_FILE):
        print(f"⚠️ [경고] 기존에 수집된 '{CSV_FILE}' 파일이 있습니다.")
        print("절대 좌표로 다시 모아야 하므로, 기존 파일을 수동으로 삭제하거나 다른 이름으로 백업 후 다시 실행해주세요!")
        return

    init_csv()

    print("\n" + "="*50)
    print(f" 🎥 {EXERCISE_NAME} 3D 자동 스캔 데이터 수집 시작!")
    print("  [안내] 유튜브 영상을 전체화면 혹은 크게 틀어두세요.")
    print(f"  - [1] 꾹 누르기: '{CLASS_DOWN}' 저장")
    print(f"  - [2] 꾹 누르기: '{CLASS_UP}' 저장")
    print("  - [t] 누르기: AI 훈련 및 pkl 저장")
    print("  - [ESC] 또는 [q]: 종료")
    print("="*50 + "\n")

    down_count = 0
    up_count = 0

    with mss.mss() as sct:
        # 모니터 1번(주 모니터) 전체 영역
        monitor = sct.monitors[1] 
        
        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb_frame)

            view_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            key = cv2.waitKey(1) & 0xFF

            # 🚨 [핵심 수정] pose_landmarks와 pose_world_landmarks가 모두 인식되었을 때만 작동
            if results.pose_landmarks and results.pose_world_landmarks:
                mp_draw.draw_landmarks(view_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                if key == ord('1'):
                    # 저장할 때는 반드시 world_landmarks를 넘김
                    save_data(CLASS_DOWN, results.pose_world_landmarks.landmark)
                    down_count += 1
                    cv2.putText(view_frame, f"SAVING: {CLASS_DOWN}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                elif key == ord('2'):
                    save_data(CLASS_UP, results.pose_world_landmarks.landmark)
                    up_count += 1
                    cv2.putText(view_frame, f"SAVING: {CLASS_UP}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if key == ord('t'):
                train_model()
            elif key == 27 or key == ord('q'):
                break

            h, w = view_frame.shape[:2]
            cv2.putText(view_frame, f"Down: {down_count}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(view_frame, f"Up: {up_count}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("AI Vision - Auto Scanner", view_frame)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
