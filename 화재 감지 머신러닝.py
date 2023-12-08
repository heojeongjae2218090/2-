# 필요한 라이브러리 import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# CSV 파일로부터 데이터셋 로드
file_path = r"C:\Users\user\OneDrive\문서\GitHub\smoke_detection_iot.csv"
data = pd.read_csv(file_path)

# 데이터 확인
print(data.head())

# 랜덤 포레스트 분류기 모델 생성
model = RandomForestClassifier(random_state=42)

# 모델 훈련
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
predictions = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, predictions)
print(f'모델 정확도: {accuracy}')
