import torch
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_model(model_path):
    model = torch.load(model_path) 
    return model

def preprocess_data(data):

    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(data)
    

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    return data_scaled

def predict_anomalies(model, data):

    predictions = model.predict(data)

    anomalies = (predictions == -1).astype(int)
    return anomalies

def load_data_from_pkl(file):
    data = pickle.load(file)
    return pd.DataFrame(data)
