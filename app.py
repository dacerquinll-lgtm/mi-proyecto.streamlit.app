import streamlit as st
import joblib
import numpy as np

# 1. Configuración de página con tema oscuro/claro profesional
st.set_page_config(
    page_title="AI Student Predictor",
    page_icon="📊",
    layout="wide"
)

# Estilo CSS personalizado para mejorar la apariencia
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
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
@st.cache_resource # Esto hace que la app cargue más rápido
def load_model():
    return joblib.load('modelo_estudiante.pkl')

try:
    model = load_model()

    # --- BARRA LATERAL ---
    st.sidebar.header("⚙️ Configuración")
    st.sidebar.write("Ajusta los parámetros para recalcular el pronóstico.")
    
    horas = st.sidebar.slider("Horas de estudio semanal", 0, 50, 15)
    asistencia = st.sidebar.slider("% de Asistencia a clases", 0, 100, 85)
    sueno = st.sidebar.slider("Horas de sueño diarias", 0, 12, 7)
    
    st.sidebar.divider()
    st.sidebar.info("Este modelo utiliza un algoritmo de Regresión Lineal entrenado con datos de rendimiento estudiantil.")

    # --- CUERPO PRINCIPAL ---
    st.title("📊 Predicctor de Rendimiento Académico")
    st.markdown("### ¿Cuál es tu probabilidad de éxito?")
    st.write("Esta herramienta utiliza Inteligencia Artificial para estimar tu nota final basándose en tus hábitos actuales.")

    st.divider()

    # Layout de 3 columnas para mostrar los datos ingresados
    col1, col2, col3 = st.columns(3)
    col1.metric("Estudio", f"{horas} hrs/sem")
    col2.metric("Asistencia", f"{asistencia}%")
    col3.metric("Descanso", f"{sueno} hrs")

    st.divider()

    # Botón central y resultado
    if st.button("🚀 GENERAR PREDICCIÓN"):
        datos = np.array([[horas, asistencia, sueno]])
        prediccion = model.predict(datos)[0]
        nota_final = round(prediccion, 1)

        # Contenedor de resultado
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
                    st.write("Tu perfil coincide con los estudiantes de más alto rendimiento.")
                elif nota_final >= 70:
                    st.write("### 👍 Vas por buen camino")
                    st.write("Mantén este ritmo para asegurar tu aprobación.")
                else:
                    st.write("### ⚠️ Atención recomendada")
                    st.write("El modelo sugiere que aumentar las horas de estudio podría mejorar drásticamente tu resultado.")

except Exception as e:
    st.error(f"Error al cargar el modelo profesional: {e}")
