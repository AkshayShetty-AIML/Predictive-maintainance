⚙️ Predictive Maintenance
🔍 Overview
This project aims to predict and detect machinery faults using a combination of unsupervised (Isolation Forest, Autoencoder) and supervised (XGBoost Classifier) machine learning models. The goal is to enable early fault detection and reduce unplanned equipment downtime by analyzing sensor data from the CWRU dataset.

🎯 Key Objectives
✅ Clean and preprocess sensor data
✅ Perform anomaly detection using Isolation Forest
✅ Predict fault categories with XGBoost Classifier
✅ Use an Autoencoder for unsupervised reconstruction error-based detection
✅ Tune models for best performance
✅ Evaluate inference time and prediction accuracy

📊 Dataset
Source: CWRU (Case Western Reserve University) Bearing Data

File: CWRU_DS.csv

Features: Various sensor measurements (e.g., vibration, temperature)

Target: fault — indicates fault type or normal operation

🛠️ Tech Stack
Programming Language: Python

Libraries:

Data Handling: numpy, pandas

Visualization: seaborn, matplotlib

ML Models: scikit-learn, xgboost

Deep Learning: tensorflow (Keras API)

Utilities: joblib, time, datetime

📂 Project Structure
plaintext
Copy
Edit
predictive_maintenance/
│── data/                  # Raw dataset (CWRU_DS.csv)
│── notebooks/             # Notebooks for EDA & experimentation (optional)
│── scripts/               # Python scripts for training & evaluation
│── models/                # Saved model files (joblib, h5, etc.)     
│── README.md              # Project documentation
│── requirements.txt       # Dependencies
│── .gitignore             # Files to ignore
🚀 How to Run the Project
1️⃣ Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/predictive_maintenance.git
cd predictive_maintenance
2️⃣ Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run Training & Inference
You can run the main Python script or Jupyter Notebook to:

Preprocess Data: Handle missing values, scale features, encode labels

Isolation Forest: Detect anomalies

XGBoost Classifier: Predict fault categories

Autoencoder: Detect anomalies via reconstruction error

Example:

bash
Copy
Edit
python scripts/train_models.py
📌 Highlights
✅ Isolation Forest
Hyperparameter tuning with GridSearchCV

Detects outliers/anomalies in the dataset

Computes anomaly scores and labels (Normal/Anomaly)

✅ XGBoost Classifier
Predicts multi-class fault labels

Uses GridSearchCV for tuning n_estimators and learning_rate

Evaluates model using MAE and accuracy score

✅ Autoencoder
Builds and trains a simple feed-forward autoencoder

Uses reconstruction error for anomaly detection

Computes dynamic threshold based on normal samples

✅ Performance Metrics
MAE (Mean Absolute Error)

R2 score for regression

F1 score (optional for classification)

Inference time per sample

🔮 Future Improvements
Test with larger real-world industrial datasets

Add more complex deep learning architectures (e.g., LSTM Autoencoders)

Integrate with edge devices for real-time monitoring

Deploy via an API using Flask/FastAPI

🤝 Contributing
Fork this repository

Create a new branch (feature/improvement)

Commit your changes

Push to your fork

Create a Pull Request

📜 License
This project is licensed under the MIT License.

✍️ Author
Akshay Shetty

⭐ If you found this project helpful:
Star ⭐ the repository and share your feedback!
