import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch


df = pd.read_csv('synthetic_data_2018_2023.csv')


label_encoders = {}
for col in ['Day', 'Weather']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop(columns=['Number of patients', 'Date', 'PPDH', 'Number of doctors', 'Time'])
y = df['Number of patients']

                                                                                             
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)iu
X_test_scaled = scaler.transform(X_test)


X_train_np = X_train_scaled
y_train_np = y_train.to_numpy().reshape(-1, 1)
X_test_np = X_test_scaled
y_test_np = y_test.to_numpy().reshape(-1, 1)


model = TabNetRegressor()
model.fit(
    X_train_np, y_train_np,
    eval_set=[(X_train_np, y_train_np), (X_test_np, y_test_np)],
    eval_name=['train', 'valid'],
    eval_metric=['mae'],
    max_epochs=200,
    patience=20,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)


y_pred = model.predict(X_test_np)

mae = mean_absolute_error(y_test_np, y_pred)
mse = mean_squared_error(y_test_np, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_np, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")


example_input = {
    'Day': label_encoders['Day'].transform(['Monday'])[0],
    'Holiday': 0,
    'Special Event': 0,  


    'Weather': label_encoders['Weather'].transform(['Sunny'])[0]
}

example_df = pd.DataFrame([example_input])
example_scaled = scaler.transform(example_df)
predicted_patients = model.predict(example_scaled)

print(f"Predicted Number of Patients: {predicted_patients[0][0]}")
