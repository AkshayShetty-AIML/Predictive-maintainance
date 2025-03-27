import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split, cross_val_score as cvs, KFold
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
import xgboost as xg 
from sklearn.metrics import mean_absolute_error,mean_squared_error,  f1_score
import datetime as dt 
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib

df = pd.read_csv("CWRU_DS.csv")
print(np.shape(df))

#Feature Engineering
print(pd.isnull(df).sum()) #missing values

X_data = df.drop('fault', axis = 1)
y_data = df['fault']
print(np.shape(X_data))

scaler = StandardScaler()
X = scaler.fit_transform(X_data)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_data)  

X_train, X_Val, y_train, y_Val = train_test_split(X, y, test_size = 0.4)
X_Val, X_test, y_Val, y_test = train_test_split(X, y, test_size = 0.15)

#IsoLation Forest
iso_forest = IsolationForest(n_estimators=100,
                            contamination=0.05,
                            max_samples=256,
                            random_state=42)
#n_estimators: no of trees, contamination - Anomaly % in the Dataset


param_grid_IF = {'n_estimators': [20, 50, 100, 200],  
              'max_samples': [128, 256, 512]} 

grid_search_iso = GridSearchCV(iso_forest, param_grid_IF, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search_iso.fit(X_train, y_train)

# Get best parameters
best_param_iso = grid_search_iso.best_params_
print("Best Parameters:", best_param_iso)

# Use the best model
best_mode_iso = grid_search_iso.best_estimator_

Result_iso = pd.DataFrame()
Result_iso['anomaly_score'] = best_mode_iso.decision_function(X)
Result_iso['Result'] = best_mode_iso.predict(X)

Result_iso['Result'].value_counts()

start_time = time.time()
best_mode_iso.predict(X)
end_time = time.time()

print(f"Inference Time: {(end_time - start_time)/len(X):.4f} seconds per sample")

Result_iso['Result'] = Result_iso['Result'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
print(Result_iso)

#XGB Classifier
XGB_model = xg.XGBClassifier(
    objective='multi:softprob',
    num_class=10,      
    learning_rate=0.001, 
    n_estimators=100
)

param_grid_xgb = {'n_estimators': [20, 50, 100, 200],  
              'learning_rate': [0.001, 0.0001]} 


grid_search_xgb = GridSearchCV(XGB_model, param_grid_xgb, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

# Get best parameters
best_param_xgb = grid_search_xgb.best_params_
print("Best Parameters:", best_param_xgb)

# Use the best model
best_mode_xgb = grid_search_xgb.best_estimator_


XGB_ypred = best_mode_xgb.predict(X_test)
XGB_ypred_df = pd.DataFrame(XGB_ypred, columns = ['Result'])
print("MAE:", mean_absolute_error(y_test, XGB_ypred))
print("XGB Regression Model - Score:",best_mode_xgb.score(X_test, y_test))


#AutoEncoder
# Define Autoencoder
input_dim = X_train.shape[1]

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)

# Decoder
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(X_train, X_train, 
                          epochs=20, 
                          batch_size=64, 
                          validation_data=(X_Val, X_Val),
                          verbose=1)



X_test_pred = autoencoder.predict(X_test)

reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=1)

threshold = np.percentile(reconstruction_error[y_test == 0], 95)

y_pred_xgb = (reconstruction_error > threshold).astype(int)

print(y_pred_xgb)
