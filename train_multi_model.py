import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# 1. 파일 이름 설정
DATA_FILE = '3d_distances.csv'
LABEL_FILE = 'labels.csv'

def main():
    print("📊 1. 데이터 불러오기 및 합치기...")
    try:
        df_data = pd.read_csv(DATA_FILE)
        df_label = pd.read_csv(LABEL_FILE)
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return

    # pose_id를 기준으로 두 파일 합치기 (Merge)
    df = pd.merge(df_data, df_label, on='pose_id')
    
    # 정답 열 이름은 'pose'입니다.
    target_col = 'pose'
    
    print(f"✅ 합치기 완료! 총 {len(df)}개의 정답 데이터를 확보했습니다.")
    
    # X(16개 거리값), y(운동 이름) 분리
    # pose_id는 학습에 필요 없으므로 제거합니다.
    X = df.drop(['pose_id', target_col], axis=1)
    y = df[target_col]

    print(f"🏋️ 학습될 동작: {list(y.unique())}")

    print("\n✂️ 2. 데이터 분리 및 학습 시작...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n📝 3. 성능 검사 중...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("="*50)
    print(f"🎉 5종 운동 통합 모델 학습 완료! 정확도: {accuracy * 100:.2f}%")
    print("="*50)

    print("\n[동작별 상세 성적표]")
    print(classification_report(y_test, y_pred))

    # 4. 모델 저장
    with open('multi_fitness_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # 5. 메인 코드에서 사용할 컬럼 순서 저장 (매우 중요)
    with open('feature_columns.txt', 'w') as f:
        f.write(",".join(X.columns.tolist()))

    print("\n✨ 'multi_fitness_model.pkl'와 'feature_columns.txt'가 생성되었습니다!")

if __name__ == '__main__':
    main()
