import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def main():
    print("📊 1. 4개의 모든 데이터 영혼까지 끌어모으는 중...")
    try:
        df_dist = pd.read_csv('3d_distances.csv')
        df_angle = pd.read_csv('angles.csv')
        df_xyz = pd.read_csv('xyz_distances.csv')
        df_land = pd.read_csv('landmarks.csv')
        df_label = pd.read_csv('labels.csv')
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return

    # 전부 pose_id 기준으로 병합 (Merge)
    df = pd.merge(df_dist, df_angle, on='pose_id')
    df = pd.merge(df, df_xyz, on='pose_id')
    df = pd.merge(df, df_land, on='pose_id')
    df = pd.merge(df, df_label, on='pose_id')

    target_col = 'pose'
    print(f"✅ 합치기 완료! 총 {len(df)}줄, 특징(Feature) 개수: {len(df.columns)-2}개")

    X = df.drop(['pose_id', target_col], axis=1)
    y = df[target_col]

    # 비율을 완벽히 맞추어 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n💪 2. [초월 등급] 4대장 AI 전문가 앙상블 학습 중... (시간이 조금 걸립니다)")
    
    # 1. 밸런스 패치된 랜덤 포레스트 (나무 1000개)
    rf = RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42, n_jobs=-1)
    
    # 2. 강력한 부스팅
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=0.1, random_state=42)
    
    # 3. 딥러닝 망 확장 (은닉층 512-256-128)
    mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, early_stopping=True, random_state=42)
    
    # 4. [NEW] 공간 분할의 마법사 서포트 벡터 머신 (SVC)
    svc = SVC(C=10, kernel='rbf', probability=True, random_state=42)

    # 4개의 뇌를 다수결 투표로 합치되, SVC에 가중치를 부여
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('hgb', hgb), ('mlp', mlp), ('svc', svc)],
        voting='soft',
        weights=[1, 1, 1, 1.5]
    )

    # 스케일러 적용 파이프라인
    model = make_pipeline(
        StandardScaler(),
        ensemble
    )
    
    model.fit(X_train, y_train)

    print("\n📝 3. 성능 검사 중...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("="*50)
    print(f"🎉 초월 등급 앙상블 모델 학습 완료! 테스트 정확도: {accuracy * 100:.2f}%")
    print("="*50)
    
    print("\n[동작별 상세 성적표]")
    print(classification_report(y_test, y_pred))

    print("\n💾 4. 학습된 파이프라인(모델) 저장 중...")
    with open('ultimate_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('ultimate_features.txt', 'w') as f:
        f.write(",".join(X.columns.tolist()))

    print("✨ 'ultimate_model.pkl'가 성공적으로 업데이트되었습니다!")

if __name__ == '__main__':
    main()
