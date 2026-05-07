import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. 모든 데이터 소환
df_dist = pd.read_csv('3d_distances.csv')
df_angle = pd.read_csv('angles.csv')
df_label = pd.read_csv('labels.csv')

# 2. 문제지 합체 (거리 + 각도)
# pose_id를 기준으로 모든 특징을 옆으로 붙입니다.
df_features = pd.merge(df_dist, df_angle, on='pose_id')
df_final = pd.merge(df_features, df_label, on='pose_id')

# 3. 학습 준비
X = df_final.drop(['pose_id', 'pose'], axis=1)
y = df_final['pose']

# 4. 강력한 모델로 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# 정확도 확인
print(f"🔥 고정밀 융합 모델 정확도: {model.score(X_test, y_test)*100:.2f}%")

# 5. 저장 (컬럼 순서가 매우 중요함)
with open('super_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('super_features.txt', 'w') as f:
    f.write(",".join(X.columns.tolist()))
