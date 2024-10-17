import streamlit as st
import pandas as pd
import plotly.express as px

# Título de la aplicación
st.title("Expected Ride Duration vs Historical Cost of Ride")

# Cargar datos
data = pd.read_csv("Data/dynamic_pricing.csv")

# Mostrar los datos en la aplicación (opcional)
st.write(data.head())

# Crear gráfico de dispersión
fig = px.scatter(data, x='Expected_Ride_Duration', 
                 y='Historical_Cost_of_Ride',
                 title='Expected Ride Duration vs Historical Cost of Ride',
                 trendline='ols')

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)
