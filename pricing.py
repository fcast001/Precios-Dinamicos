import streamlit as st
import pandas as pd
import plotly.express as px

# Título de la aplicación
st.title("Expected Ride Duration vs Historical Cost of Ride")

# Opción para seleccionar el número de columnas
layout_option = st.selectbox("Selecciona el diseño de columnas", ("Una columna", "Dos columnas"))

# Cargar datos
data = pd.read_csv("Data/dynamic_pricing.csv")

# Mostrar los datos (opcional)
st.write(data.head())

# Crear gráfico de dispersión
fig = px.scatter(data, x='Expected_Ride_Duration', 
                 y='Historical_Cost_of_Ride',
                 title='Expected Ride Duration vs Historical Cost of Ride',
                 trendline='ols')

# Condición para cambiar entre una y dos columnas
if layout_option == "Una columna":
    # Diseño de una columna
    st.image("img/demanda.png", caption="Descripción de la imagen", use_column_width=True)
    st.plotly_chart(fig)

elif layout_option == "Dos columnas":
    # Diseño de dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("img/demanda.png", caption="Descripción de la imagen", use_column_width=True)
    
    with col2:
        st.plotly_chart(fig)
