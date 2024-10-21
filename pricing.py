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