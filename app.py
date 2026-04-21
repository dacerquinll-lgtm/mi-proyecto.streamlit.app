import streamlit as st
import joblib
import numpy as np

# 1. Configuración visual
st.set_page_config(page_title="Predicador de Notas", page_icon="🎓")

st.title("🎓 ¿Qué nota sacarás en tu examen?")
st.write("Mueve los controles para ver la predicción de la IA basada en tus hábitos.")

# 2. Cargar el cerebro del modelo
# El archivo modelo_estudiante.pkl debe estar en la misma carpeta de GitHub
try:
    modelo = joblib.load('modelo_estudiante.pkl')
    
    # 3. Crear controles deslizantes (Sliders)
    # Ajusta los rangos (mínimo, máximo, valor inicial)
    horas = st.slider("Horas de estudio semanal", 0, 50, 15)
    asistencia = st.slider("% de Asistencia a clase", 0, 100, 80)
    sueno = st.slider("Horas de sueño diarias", 0, 12, 7)

    # 4. Botón de acción
    if st.button("Calcular mi nota estimada"):
        # Creamos el array con los datos (mismo orden que en Kaggle)
        datos_usuario = np.array([[horas, asistencia, sueno]])
        prediccion = modelo.predict(datos_usuario)
        
        # Mostrar resultado con estilo
        st.balloons()
        nota_final = round(prediccion[0], 2)
        st.metric(label="Nota Estimada", value=f"{nota_final} / 100")
        
        if nota_final >= 70:
            st.success("¡Excelente! Según tus datos, vas por muy buen camino.")
        else:
            st.warning("El modelo sugiere que podrías mejorar aumentando tus horas de estudio.")

except FileNotFoundError:
    st.error("⚠️ Error: No se encuentra el archivo 'modelo_estudiante.pkl'. Por favor, súbelo a este repositorio.")
