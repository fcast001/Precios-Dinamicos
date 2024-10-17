import streamlit as st
import pandas as pd
import plotly.express as px

# Configurar la página para que sea más ancha
st.set_page_config(layout="wide")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Título de la aplicación
st.title("Expected Ride Duration vs Historical Cost of Ride")

# Cargar datos
data = pd.read_csv("Data/dynamic_pricing.csv")

# Sección 1: Diseño en una sola columna pero con imagen controlada en ancho
 # Ajustar el ancho de la imagen
st.write(data.head())
# Sección 2: Mostrar los datos y el gráfico de dispersión en dos columnas
col1, col2 = st.columns(2)

with col1:
    st.image("img/demanda.png", caption="Descripción de la imagen", width=500)   # Mostrar algunos datos

with col2:
    # Crear gráfico de dispersión
    fig = px.scatter(data, x='Expected_Ride_Duration', 
                     y='Historical_Cost_of_Ride',
                     title='Expected Ride Duration vs Historical Cost of Ride',
                     trendline='ols')
    st.plotly_chart(fig)
