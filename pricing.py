import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

st.markdown("""
    <style>
    .custom-subheader {
        background-color: #838483; /* Cambia el color aquí */
        color: white;
        padding: 10px;
        font-size: 24px;
    }
    </style>
    <div class="custom-subheader">Vista Descriptiva de los datos</div>
    """, unsafe_allow_html=True)
# Crear columnas para el texto y la tabla
col1, col2 = st.columns(2)
with col1:
    # Texto descriptivo
    st.write("Realizamos un análisis exploratorio de datos para tener una mejor visión de las estadísticas descriptivas de los datos.")

    # Mostrar código del notebook
    notebook_code = """
    import pandas as pd

    # Cargamos los datos de precios
    data = pd.read_csv("Data/dynamic_pricing.csv")

    # Visualizamos los primeros datos
    data.head()
    """

    st.code(notebook_code, language='python')

#####################################################################################################################################
st.markdown("""
    <style>
    .custom-subheader {
        background-color: #838483; /* Cambia el color aquí */
        color: white;
        padding: 10px;
        font-size: 24px;
    }
    </style>
    <div class="custom-subheader">Correlación de variables</div>
    """, unsafe_allow_html=True)
with col2:
    # Mostrar información del DataFrame
    buffer = StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)  # Mostrar la información como texto

# Sección 2: Mostrar los datos y el gráfico de dispersión en dos columnas
col1, col2 = st.columns(2)

with col1:
    #st.image("img/demanda.png", caption="Descripción de la imagen", width=500)  # Mostrar algunos datos


    st.markdown("<br>Ahora veamos la relación entre la duración esperada del viaje y el costo histórico del viaje:", unsafe_allow_html=True)

    # Mostrar código del notebook
    notebook_code = """
    #Gráfico de dispersión de la duración del viaje frente al coste del mismo
    fig = px.box(data, x='Vehicle_Type',
                y='Historical_Cost_of_Ride',
                title='Duración prevista del viaje vs coste histórico del viaje')
    fig.show()
    """

    st.code(notebook_code, language='python')

with col2:
    # Crear gráfico de dispersión
    fig = px.scatter(data, x='Expected_Ride_Duration', 
                     y='Historical_Cost_of_Ride',
                     title='Duración prevista del viaje vs coste histórico del viaje',
                     trendline='ols')
    st.plotly_chart(fig)
#####################################################################################################################################

st.markdown("""
    <style>
    .custom-subheader {
        background-color: #838483; /* Cambia el color aquí */
        color: white;
        padding: 10px;
        font-size: 24px;
    }
    </style>
    <div class="custom-subheader">Descripción de Costos por tipo de vehiculo</div>
    """, unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    #st.image("img/demanda.png", caption="Descripción de la imagen", width=500)  # Mostrar algunos datos


    st.markdown("<br>Ahora veamos la distribución del costo histórico de los viajes según el tipo de vehículo:", unsafe_allow_html=True)

    # Mostrar código del notebook
    notebook_code = """
        fig = px.box(data, x='Vehicle_Type', 
                    y='Historical_Cost_of_Ride',
                    title='Historical Cost of Ride Distribution by Vehicle Type')
        fig.show()
    """

    st.code(notebook_code, language='python')

with col2:
    # Crear gráfico de dispersión
    fig = px.box(data, x='Vehicle_Type', 
                    y='Historical_Cost_of_Ride',
                    title='Historical Cost of Ride Distribution by Vehicle Type')
    st.plotly_chart(fig)


st.markdown("""
    <style>
    .custom-subheader {
        background-color: #838483; /* Cambia el color aquí */
        color: white;
        padding: 10px;
        font-size: 24px;
    }
    </style>
    <div class="custom-subheader">Matríz de correlación</div>
    """, unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    #st.image("img/demanda.png", caption="Descripción de la imagen", width=500)  # Mostrar algunos datos


    st.markdown("<br>Ahora echemos un vistazo a la matriz de correlación:", unsafe_allow_html=True)

    # Mostrar código del notebook
    notebook_code = """
        corr_matrix = data.corr()

        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, 
                                        x=corr_matrix.columns, 
                                        y=corr_matrix.columns,
                                        colorscale='Viridis'))
        fig.update_layout(title='Correlation Matrix')
        fig.show()
    """

    st.code(notebook_code, language='python')

with col2:
    data_numeric = data.select_dtypes(include=[float, int])  # Solo columnas numéricas
    corr_matrix = data_numeric.corr()

    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, 
                                    x=corr_matrix.columns, 
                                    y=corr_matrix.columns,
                                    colorscale='Viridis'))
    fig.update_layout(title='Correlation Matrix')

    st.plotly_chart(fig)


#####################################################################################################################################

st.markdown("""
    <style>
    .custom-subheader {
        background-color: #838483; /* Cambia el color aquí */
        color: white;
        padding: 10px;
        font-size: 24px;
    }
    </style>
    <div class="custom-subheader">Implementación de una estrategia de precios dinámicos</div>
    """, unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    #st.image("img/demanda.png", caption="Descripción de la imagen", width=500)  # Mostrar algunos datos


    st.markdown("<br>Los datos proporcionados por la empresa indican que la empresa utiliza un modelo de precios que"+ 
                "solo toma la duración esperada del viaje como un factor para determinar el precio del mismo. Ahora, "+
                "implementaremos una estrategia de precios dinámicos con el objetivo de ajustar los costos de los viajes de" +
                "forma dinámica en función de los niveles de demanda y oferta observados en los datos. Captará períodos de alta" +
                "demanda y escenarios de baja oferta para aumentar los precios, mientras que los períodos de baja demanda y situaciones"+ 
                "de alta oferta conducirán a reducciones de precios. A continuación se explica cómo implementar esta estrategia de precios dinámicos utilizando Python:", unsafe_allow_html=True)


    # Mostrar código del notebook
    notebook_code = """
        # Calculate demand_multiplier based on percentile for high and low demand
        high_demand_percentile = 75
        low_demand_percentile = 25

        data['demand_multiplier'] = np.where(data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                            data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                            data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile))

        # Calculate supply_multiplier based on percentile for high and low supply
        high_supply_percentile = 75
        low_supply_percentile = 25

        data['supply_multiplier'] = np.where(data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'], low_supply_percentile),
                                            np.percentile(data['Number_of_Drivers'], high_supply_percentile) / data['Number_of_Drivers'],
                                            np.percentile(data['Number_of_Drivers'], low_supply_percentile) / data['Number_of_Drivers'])

        # Define price adjustment factors for high and low demand/supply
        demand_threshold_high = 1.2  # Higher demand threshold
        demand_threshold_low = 0.8  # Lower demand threshold
        supply_threshold_high = 0.8  # Higher supply threshold
        supply_threshold_low = 1.2  # Lower supply threshold

        # Calculate adjusted_ride_cost for dynamic pricing
        data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride'] * (
            np.maximum(data['demand_multiplier'], demand_threshold_low) *
            np.maximum(data['supply_multiplier'], supply_threshold_high)
        )
    """

    st.code(notebook_code, language='python')

with col2:
    st.markdown("<br>")
    high_demand_percentile = 75
    low_demand_percentile = 25

    data['demand_multiplier'] = np.where(data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                        data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                        data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile))

    # Calculate supply_multiplier based on percentile for high and low supply
    high_supply_percentile = 75
    low_supply_percentile = 25

    data['supply_multiplier'] = np.where(data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'], low_supply_percentile),
                                        np.percentile(data['Number_of_Drivers'], high_supply_percentile) / data['Number_of_Drivers'],
                                        np.percentile(data['Number_of_Drivers'], low_supply_percentile) / data['Number_of_Drivers'])

    # Define price adjustment factors for high and low demand/supply
    demand_threshold_high = 1.2  # Higher demand threshold
    demand_threshold_low = 0.8  # Lower demand threshold
    supply_threshold_high = 0.8  # Higher supply threshold
    supply_threshold_low = 1.2  # Lower supply threshold

    # Calculate adjusted_ride_cost for dynamic pricing
    data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride'] * (
        np.maximum(data['demand_multiplier'], demand_threshold_low) *
        np.maximum(data['supply_multiplier'], supply_threshold_high)
    )

    st.write(data.head(50))