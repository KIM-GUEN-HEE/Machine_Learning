import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# CSV 파일 읽기
data = pd.read_csv(r'C:\Users\user\Desktop\Dataset1.csv')

# 데이터 전처리
# 필요한 특성 선택
features = ['Packet_Rate', 'Packet_Drop_Rate', 'Packet_Duplication_Rate', 'Data_Throughput',
            'Signal_Strength', 'SNR', 'Battery_Level', 'Energy_Consumption_Rate',
            'Number_of_Neighbors', 'Route_Request_Frequency', 'Route_Reply_Frequency',
            'Data_Transmission_Frequency', 'Data_Reception_Frequency', 'Error_Rate',
            'CPU_Usage', 'Memory_Usage', 'Bandwidth']

X = data[features]
y = data['Is_Malicious']

# 훈련 세트와 테스트 세트로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# train_test_split : scikit-learn 라이브러리에서 훈련세트와 테스트 세트로 무작위로 분할하는 데 사용

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 머신러닝 모델 선택 (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 훈련
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 분류 보고서 출력
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
