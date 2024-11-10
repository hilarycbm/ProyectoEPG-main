import torch
import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

def load_model(model_path, model_type):
    if model_type == "Isolation Forest":
        with open(model_path, 'rb') as file:
            model = joblib.load(model_path)
    elif model_type == "Autoencoder":
        model = torch.load(model_path) 
    else:
        raise ValueError("Tipo de modelo no reconocido.")
    return model

def preprocess_data(data):
    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(data)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    return data_scaled

def predict_anomalies(model, data, model_type):
    if model_type == "Isolation Forest":
    
        predictions = model.predict(data)
        anomalies = predictions

    elif model_type == "Autoencoder":
    
        data_tensor = torch.tensor(data, dtype=torch.float32)
        reconstructed_data = model(data_tensor).numpy()
        reconstruction_error = ((data - reconstructed_data) ** 2).mean(axis=1)
        
        threshold = 0.05
        anomalies = (reconstruction_error > threshold).astype(int) 

    else:
        raise ValueError("Tipo de modelo no reconocido para predicción.")

    return anomalies

def load_data_from_pkl(file):
    data = pickle.load(file)
    return pd.DataFrame(data)