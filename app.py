import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo guardado
modelo = joblib.load('modelo_estudiante.pkl')

st.title("Sistema Inteligente de Aprendizaje Personalizado")
st.write("Ingresa los datos del estudiante para obtener una recomendación:")

# Entradas para el usuario
horas = st.number_input("Horas de estudio", 0, 50, 15)
asistencia = st.slider("Asistencia (%)", 0, 100, 80)
estilo = st.selectbox("Estilo de Aprendizaje", [1, 2, 3, 4], format_func=lambda x: {1:"Visual", 2:"Auditivo", 3:"Kinestésico", 4:"Lectura"}[x])
motivacion = st.slider("Nivel de Motivación", 0, 10, 5)

if st.button("Generar Plan Personalizado"):
    # Crear el formato de entrada para el modelo
    datos_entrada = pd.DataFrame([[horas, asistencia, estilo, motivacion]], 
                                 columns=['StudyHours', 'Attendance', 'LearningStyle', 'Motivation'])
    
    prediccion = modelo.predict(datos_entrada)
    
    st.subheader(f"Rendimiento estimado: {prediccion[0]:.2f}%")
    
    # Lógica de personalización (Invisible para el usuario)
    if estilo == 1:
        st.success("Sugerencia: Tu perfil es Visual. Se han desbloqueado mapas mentales y videos.")
    elif estilo == 3:
        st.success("Sugerencia: Tu perfil es Kinestésico. Se han asignado laboratorios prácticos.")
