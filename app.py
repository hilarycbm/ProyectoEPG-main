import streamlit as st
import pandas as pd
from utils import load_model, preprocess_data, predict_anomalies, load_data_from_pkl

st.set_page_config(page_icon="🤡", page_title="Trabajo Final")
st.image("logob_m_EPG.png", width=200)
st.title("Detección de anomalías para clientes en el sector financiero")
col1, col2 = st.columns([5, 5])


st.subheader("Algoritmo Isolation Forest para Detección de Fraude")
st.info("""
El algoritmo **Isolation Forest** es un método de aprendizaje automático utilizado para la detección de anomalías.
Se basa en el principio de que los puntos de datos anómalos (en este caso, posibles casos de fraude) son más fáciles de 
"aislar" o separar que los puntos normales. En el contexto financiero, Isolation Forest ayuda a identificar patrones 
sospechosos en las cuentas de los clientes, lo que puede indicar actividades fraudulentas.

Este modelo divide los datos en árboles de aislamiento y mide la facilidad con la que un punto se aísla. Si un punto es 
aislado rápidamente, se clasifica como una posible anomalía, permitiendo así una detección temprana de posibles fraudes 
en las cuentas de los clientes.
""")


model_path = 'isolation_forest_model.pth'
model = load_model(model_path)
st.success("Modelo Isolation Forest cargado desde el archivo local.")

data_file = st.file_uploader("Sube un archivo .pkl con datos de entrada", type="pkl", key="data_upload")
if data_file is not None:
    try:
        
        data = load_data_from_pkl(data_file)
        
        
        if isinstance(data, pd.DataFrame):
            st.write("Datos cargados con éxito:", data)
            
            
            data_scaled = preprocess_data(data)
            anomalies = predict_anomalies(model, data_scaled)
            
            
            st.write("Resultados de la detección de fraude:")
            data['EsFraude'] = anomalies
            st.write(data[['EsFraude']])
            st.info(f"Total de transacciones anómalas detectadas: {sum(anomalies)}")
        else:
            st.error("El archivo .pkl no contiene un DataFrame válido.")
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el archivo .pkl: {e}")
else:
    st.info("Por favor, sube el archivo de datos en formato .pkl.")
