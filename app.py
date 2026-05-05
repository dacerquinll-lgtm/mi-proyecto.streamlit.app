import streamlit as st
import joblib
import numpy as np
from supabase import create_client  # NUEVO

# 1. Configuración de página
st.set_page_config(
    page_title="AI Student Predictor",
    page_icon="📊",
    layout="wide"
)

# --- CONEXIÓN A BASE DE DATOS (NUEVO) ---
try:
    # Estos nombres deben coincidir con tus "Secrets" en Streamlit Cloud
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase = create_client(url, key)
except Exception as e:
    st.error(f"Error de configuración de Base de Datos: {e}")

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
    st.sidebar.divider()
    st.sidebar.info("Este modelo utiliza un algoritmo de Regresión Lineal.")

    # --- CUERPO PRINCIPAL ---
    st.title("📊 Predictor de Rendimiento Académico")
    st.markdown("### ¿Cuál es tu probabilidad de éxito?")
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Estudio", f"{horas} hrs/sem")
    col2.metric("Asistencia", f"{asistencia}%")
    col3.metric("Descanso", f"{sueno} hrs")

    st.divider()

    # --- BOTÓN Y LÓGICA DE PREDICCIÓN ---
    if st.button("🚀 GENERAR PREDICCIÓN"):
        datos = np.array([[horas, asistencia, sueno]])
        prediccion = model.predict(datos)[0]
        nota_final = round(float(prediccion), 1) # Aseguramos que sea float para la BD

        # Contenedor de resultado visual
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

        # --- GUARDAR EN SUPABASE (NUEVO) ---
        registro = {
            "inputs": f"Horas: {horas}, Asis: {asistencia}%, Sueño: {sueno}",
            "prediccion": str(nota_final)
        }
        
        try:
            supabase.table("predicciones").insert(registro).execute()
            st.toast("✅ Predicción guardada en la base de datos", icon="💾")
        except Exception as e:
            st.error(f"No se pudo guardar en la BD: {e}")

except Exception as e:
    st.error(f"Error al cargar la aplicación: {e}")
