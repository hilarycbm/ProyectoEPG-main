import streamlit as st
import pandas as pd
from utils import load_model, preprocess_data, predict_anomalies, load_data_from_pkl

st.set_page_config(page_icon="🎇", page_title="Trabajo Final")
st.image("logob_m_EPG.png", width=200)
st.title("Detección de anomalías para clientes en el sector financiero")

tab1, tab2 = st.tabs(["Interfaz Empresa", "Interfaz Usuario"])

with tab1:
    st.subheader("Seleccione el modelo para la detección de anomalías")
    model_option = st.selectbox(
        "Modelos disponibles:",
        ("Isolation Forest", "Autoencoder")
    )

    if model_option == "Isolation Forest":
        st.subheader("Algoritmo Isolation Forest para Detección de Fraude")
        st.info("""
        El algoritmo *Isolation Forest* es un método de aprendizaje automático utilizado para la detección de anomalías.
        Se basa en el principio de que los puntos de datos anómalos (en este caso, posibles casos de fraude) son más fáciles de 
        "aislar" o separar que los puntos normales. En el contexto financiero, Isolation Forest ayuda a identificar patrones 
        sospechosos en las cuentas de los clientes, lo que puede indicar actividades fraudulentas.
        """)

        model_path = 'best_isolation_forest_model_2.pkl'

    elif model_option == "Autoencoder":
        st.subheader("Red Neuronal Autoencoder para Detección de Anomalías")
        st.info("""
        El modelo *Autoencoder* es una red neuronal que aprende a reconstruir los datos de entrada. Las anomalías se detectan 
        cuando los datos reconstruidos difieren significativamente de los datos originales, indicando un posible caso de fraude.
        """)

        model_path = 'encoder_modelo_prueba.pth'

    model = load_model(model_path, model_option)
    st.success(f"Modelo {model_option} cargado desde el archivo local.")

    data_file = st.file_uploader("Sube un archivo .pkl con datos de entrada", type="pkl", key="data_upload")
    if data_file is not None:
        try:
            data = load_data_from_pkl(data_file)
            
            if isinstance(data, pd.DataFrame):
                st.write("Datos cargados con éxito:", data)
                
            
                data_scaled = preprocess_data(data)
                anomalies = predict_anomalies(model, data_scaled, model_type=model_option)
                
            
                st.write("Resultados de la detección de fraude, Para el modelo Isolation Forest(0: Normal 1: Anomalia). Para el modelo de Autoencoder(0:Anomalia 1:Normal)")
                data['EsFraude'] = anomalies
                st.write(data[['EsFraude']])
                st.info(f"Total de transacciones anómalas detectadas: {sum(anomalies)}")
            else:
                st.error("El archivo .pkl no contiene un DataFrame válido.")
        except Exception as e:
            st.error(f"Ocurrió un error al cargar el archivo .pkl: {e}")
    else:
        st.info("Por favor, sube el archivo de datos en formato .pkl.")

with tab2:
    st.subheader("Estimado usuario por favor ingrese los datos respectivos para su evaluación.")
    
    number = st.number_input("CC NUM")
    number1 = st.number_input("AMT")
    number2 = st.number_input("LATITUD")
    number3 = st.number_input("LONGITUD")
    number4 = st.number_input("CIUDAD")
    number5 = st.number_input("LATITUD APROXIMADA")
    number6 = st.number_input("LONGITUD APROXIMADA")
    number7 = st.number_input("HORA")
    number8 = st.number_input("DIA")
    number9 = st.number_input("MES")
    number10 = st.number_input("AÑO")
    number11 = st.number_input("CATEGORIA")
    number12 = st.number_input("GENERO")
    number13 = st.number_input("CIUDAD PRINCIPAL")
    number14 = st.number_input("ESTADO")
    number15 = st.number_input("TRABAJO")
    number16 = st.number_input("EDAD")

    if st.button("Evaluar Anomalía"):

        user_data = pd.DataFrame([[
            number, number1, number2, number3, number4, number5,
            number6, number7, number8, number9, number10, number11,
            number12, number13, number14, number15, number16
        ]], columns=[
            "CC NUM", "AMT", "LATITUD", "LONGITUD", "CIUDAD", 
            "LATITUD APROXIMADA", "LONGITUD APROXIMADA", "HORA", 
            "DIA", "MES", "AÑO", "CATEGORIA", "GENERO", "CIUDAD", 
            "ESTADO", "TRABAJO", "EDAD"
        ])


        user_data_scaled = preprocess_data(user_data)
        anomalies = predict_anomalies(model, user_data_scaled, model_type=model_option)


        if anomalies[0] == 1:
            st.warning("Anomalía detectada en la transacción ingresada.")
        else:
            st.success("La transacción ingresada es normal.")