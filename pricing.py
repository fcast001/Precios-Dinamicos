import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

# Configurar la página para que sea más ancha
st.set_page_config(layout="wide")

# Ocultar el header, footer y el menú principal
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Título de la aplicación
st.title("Precios dinámicos con Python y Aprendizaje Automático")

# Cargar datos
data = pd.read_csv("Data/dynamic_pricing.csv")

st.subheader("Vista Previa de los datos")
# Sección 1: Diseño en una sola columna pero con imagen controlada en ancho
st.write(data.head())

st.subheader("Vista Descriptiva de los datos")
# Crear columnas para el texto y la tabla
col1, col2 = st.columns(2)
with col1:
    # Texto descriptivo
    st.write("Realizamos un análisis exploratorio de datos para tener una mejor visión de las estadísticas descriptivas de los datos.")

    import streamlit as st

# Título de la sección
st.title("Precios dinámicos basados en datos mediante Python y aprendizaje automático")

# Sección del Jupyter Notebook
st.subheader("Sección orientada a Jupyter Notebook")

st.markdown("""
Aquí puedes ver un fragmento de código tomado de nuestro análisis en Jupyter Notebook:
""")

# Mostrar código del notebook
notebook_code = """
import pandas as pd

# Cargamos los datos de precios
df = pd.read_csv('precios_cataluna.csv')

# Visualizamos los primeros datos
df.head()
"""

st.code(notebook_code, language='python')


with col2:
    # Mostrar información del DataFrame
    buffer = StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)  # Mostrar la información como texto

# Sección 2: Mostrar los datos y el gráfico de dispersión en dos columnas
col1, col2 = st.columns(2)

with col1:
    st.image("img/demanda.png", caption="Descripción de la imagen", width=500)  # Mostrar algunos datos

with col2:
    # Crear gráfico de dispersión
    fig = px.scatter(data, x='Expected_Ride_Duration', 
                     y='Historical_Cost_of_Ride',
                     title='Expected Ride Duration vs Historical Cost of Ride',
                     trendline='ols')
    st.plotly_chart(fig)
