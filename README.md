# Estimated-Patient-Count-Prediciton-System

## Overview

This project implements a machine learning model using the TabNet Regressor to predict the number of patients visiting a healthcare facility based on various factors such as day of the week, weather conditions, and holidays.

## Dataset

The dataset used for training spans from 2018 to 2023 and contains the following features:

- **Day**: Day of the week
- **Weather**: Weather conditions
- **Holiday**: Indicator for public holidays
- **Special Event**: Indicator for special events
- **Other relevant factors**

The target variable is:

- **Number of Patients**: The count of patients expected for a given day

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn pytorch-tabnet torch
```

## Preprocessing Steps

1. Load the dataset using Pandas.
2. Encode categorical variables (Day, Weather) using Label Encoding.
3. Drop unnecessary columns (e.g., Date, PPDH, Number of doctors, Time).
4. Split data into training and testing sets (80% train, 20% test).
5. Standardize the features using StandardScaler.

## Model Training

- **Model**: TabNet Regressor
- **Hyperparameters**:
  - `max_epochs=200`
  - `patience=20`
  - `batch_size=256`
  - `virtual_batch_size=128`
  - `drop_last=False`
- The model is trained with Mean Absolute Error (MAE) as the evaluation metric.

## Performance Metrics

After training, the model's performance is evaluated using:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## Example Prediction

To predict the number of patients for a given day:

```python
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
```

## Future Enhancements

- Improve feature engineering by adding more relevant factors.
- Optimize hyperparameters for better model performance.
- Experiment with additional machine learning models for comparison.



