# ECG Analysis

This project provides a simple ECG analysis tool using a Logistic Regression model trained on ECG images.

## Usage

1.  Upload an ECG image through the Streamlit app.
2.  The app will display the image and predict the condition of the patient.

## Model

The model is trained on a dataset of ECG images categorized into four classes:

*   Normal Person
*   Patient with History of MI
*   Patient with abnormal heartbeat
*   Myocardial Infarction Patient

## Files

*   `ecg_model/train.py`: Training script for the Logistic Regression model.
*   `ecg_model/test.py`: Testing script for evaluating the trained model.
*   `ecg_model/app.py`: Streamlit app for image upload and prediction.
*   `model.pkl`: Trained Logistic Regression model.
