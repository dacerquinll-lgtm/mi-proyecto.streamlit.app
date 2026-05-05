import streamlit as st
import joblib
import numpy as np
from supabase import create_client # Necesitas agregar 'supabase' a requirements.txt

# 1. Configuración de página
st.set_page_config(
    page_title="AI Student Predictor",
    page_icon="📊",
    layout="wide"
)

# --- CONEXIÓN A SUPABASE ---
# Usamos los secrets que configuraste en el panel de Streamlit
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(url, key)
except Exception as e:
    st.error("Error al conectar con los Secrets de Supabase. Verifica el panel de Streamlit.")

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Cargar el modelo
@st.cache_resource 
def load_model():
    return joblib.load('modelo_estudiante.pkl')

try:
    model = load_model()

    # --- BARRA LATERAL ---
    st.sidebar.header("⚙️ Configuración")
    horas = st.sidebar.slider("Horas de estudio semanal", 0, 50, 15)
    asistencia = st.sidebar.slider("% de Asistencia a clases", 0, 100, 85)
    sueno = st.sidebar.slider("Horas de sueño diarias", 0, 12, 7)
    st.sidebar.info("Modelo de Regresión Lineal entrenado.")

    # --- CUERPO PRINCIPAL ---
    st.title("📊 Predictor de Rendimiento Académico")
    st.markdown("### ¿Cuál es tu probabilidad de éxito?")
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Estudio", f"{horas} hrs/sem")
    col2.metric("Asistencia", f"{asistencia}%")
    col3.metric("Descanso", f"{sueno} hrs")

    st.divider()

    # Botón central y resultado
    if st.button("🚀 GENERAR PREDICCIÓN"):
        # Realizar la predicción
        datos_array = np.array([[horas, asistencia, sueno]])
        prediccion = model.predict(datos_array)[0]
        nota_final = round(float(prediccion), 1)

        # MOSTRAR RESULTADOS
        with st.container():
            c1, c2 = st.columns([1, 2])
            with c1:
                if nota_final >= 70:
                    st.success(f"## Nota: {nota_final}")
                else:
                    st.warning(f"## Nota: {nota_final}")
            
            with c2:
                if nota_final >= 85:
                    st.balloons()
                    st.write("### ✨ ¡Excelente desempeño!")
                elif nota_final >= 70:
                    st.write("### 👍 Vas por buen camino")
                else:
                    st.write("### ⚠️ Atención recomendada")

        # --- GUARDAR EN BASE DE DATOS ---
        # Preparamos el diccionario con los nombres de tus columnas en Supabase
        registro = {
            "inputs": f"Horas: {horas}, Asistencia: {asistencia}, Sueño: {sueno}",
            "prediccion": str(nota_final)
        }
        
        try:
            # Insertar en la tabla 'predicciones'
            supabase.table("predicciones").insert(registro).execute()
            st.toast("✅ Datos guardados en la nube", icon="💾")
        except Exception as db_error:
            st.error(f"Error al guardar en la base de datos: {db_error}")

except Exception as e:
    st.error(f"Error general en la aplicación: {e}")
