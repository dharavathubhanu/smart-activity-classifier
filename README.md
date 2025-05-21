# PIRVision: FoG Presence Detection using Machine Learning

This project uses a supervised machine learning model to classify human presence states using PIR motion sensor data. It is based on the PIRVision FoG dataset and distinguishes between vacant areas, stationary human presence, and motion activity in office/classroom environments.

##  Project Files

- project_2_bonus_model.ipynb – Jupyter notebook with model training, evaluation, and prediction
- project_2_bonus_model.pkl – Trained ML model for classification
- project_2_bonus_model_scaler.pkl – Scaler object for feature normalization
- pirvision_fog_presence_detection.zip – Dataset with sensor readings and activity labels

##  Dataset Description

The dataset used is **PIRVision: FoG Presence Detection**, consisting of PIR (Passive Infrared) sensor data collected in controlled office and classroom settings. Sensor data includes timestamped events reflecting motion, used to predict presence status.

### Target Classes:
- `0` – Vacancy
- `1` – Stationary Human Presence
- `3` – Other Motion/Activity

##  Objective

The aim is to build a robust classifier to:
- Automatically detect human presence
- Enable smarter energy usage and space awareness in buildings
- Work with simple binary sensor data instead of video

## Technologies Used

- Python 3.7+
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Joblib / Pickle

## ⚙ Requirements
pip install scikit-learn pandas numpy matplotlib seaborn joblib

## Results
The model demonstrated strong classification performance on real-world sensor readings. Presence states were accurately classified even with low-resolution binary data.

## Future Enhancements
Integration with Raspberry Pi for live sensor feeds

Streaming-based real-time detection

Ensemble methods for improved accuracy

Deployable API for smart home systems

## Author
Bhanu Prasad Dharavathu

