import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
import os

# 불필요한 경고 메시지 차단
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def main():
    print("📊 1. 푸시업 데이터 불러오는 중...")
    # CSV 파일 읽기
    try:
        df = pd.read_csv('pushup_dataset.csv')
    except FileNotFoundError:
        print("❌ 'pushup_dataset.csv' 파일을 찾을 수 없습니다. 수집기 코드를 먼저 실행했는지 확인하세요.")
        return

    # X(입력값: 좌표 데이터), y(정답: 동작 이름) 나누기
    X = df.drop('class', axis=1)
    y = df['class']

    print(f"✅ 총 {len(df)}개의 푸시업 자세 데이터를 성공적으로 읽었습니다!")

    print("\n✂️ 2. 학습용과 테스트용(모의고사) 데이터 분리 중...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n💪 3. 푸시업 인공지능 모델 학습 중... (수초 내외 소요)")
    # n_jobs=-1 (모든 코어 사용)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n📝 4. 모의고사 채점 중...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("="*40)
    print(f"🎉 학습 완료! 푸시업 모델 최종 정확도: {accuracy * 100:.2f}%")
    print("="*40)

    print("\n[동작별 세부 인식률 성적표]")
    print(classification_report(y_test, y_pred))

    print("\n💾 5. 학습된 푸시업 뇌(모델) 파일로 저장 중...")
    with open('pushup_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("✨ 'pushup_model.pkl' 파일이 성공적으로 생성되었습니다!")

# 윈도우 멀티프로세싱 충돌 방지를 위한 필수 방어막
if __name__ == '__main__':
    main()
