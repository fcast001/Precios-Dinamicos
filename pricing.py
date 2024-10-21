import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    <div class="custom-subheader">Descripción de Costos por Estado del día</div>
    """, unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    #st.image("img/demanda.png", caption="Descripción de la imagen", width=500)  # Mostrar algunos datos


    st.markdown("<br>Ahora veamos la distribución del costo histórico de los viajes según el tipo de hora en la que esta el día:", unsafe_allow_html=True)

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
    fig = px.box(data, x='Time_of_Booking',
             y='Historical_Cost_of_Ride',
             title='Historical Cost of Ride Distribution by Time of Booking')
    st.plotly_chart(fig)

########################################################################################################################################

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

    st.write(data.head(10))


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


    st.markdown("<br>En el código anterior, calculamos primero el multiplicador de demanda comparando la cantidad de"+ 
                 "pasajeros con percentiles que representan niveles de demanda alta y baja. Si la cantidad de pasajeros"+ 
                 "supera el percentil de demanda alta, el multiplicador de demanda se establece como la cantidad de pasajeros"+ 
                 "dividida por el percentil de demanda alta. De lo contrario, si la cantidad de pasajeros cae por debajo del"+ 
                 "percentil de demanda baja, el multiplicador de demanda se establece como la cantidad de pasajeros dividida por"+ 
                 "el percentil de demanda baja. A continuación, calculamos el multiplicador de la oferta comparando el número"+ 
                 "de impulsores con los percentiles que representan los niveles de oferta altos y bajos. Si el número de"+ 
                 "impulsores supera el percentil de oferta baja, el multiplicador de la oferta se establece como el percentil"+ 
                 "de oferta alta dividido por el número de impulsores. Por otro lado, si el número de impulsores está por debajo"+ 
                 "del percentil de oferta baja, el multiplicador de la oferta se establece como el percentil de oferta baja dividido"+ 
                 "por el número de impulsores. Por último, calculamos el costo del viaje ajustado para el precio dinámico. Multiplica"+ 
                 "el costo histórico del viaje por el máximo del multiplicador de demanda y un umbral inferior (demand_threshold_low),"+ 
                 "y también por el máximo del multiplicador de oferta y un umbral superior (supply_threshold_high). Esta multiplicación"+
                 "garantiza que el costo del viaje ajustado capture el efecto combinado de los multiplicadores de demanda y oferta,"+ 
                 "y que los umbrales sirvan como topes o pisos para controlar los ajustes de precios.", unsafe_allow_html=True)


    # Mostrar código del notebook
    notebook_code = """
        # Calculate the profit percentage for each ride
        data['profit_percentage'] = ((data['adjusted_ride_cost'] - data['Historical_Cost_of_Ride']) / data['Historical_Cost_of_Ride']) * 100
        # Identify profitable rides where profit percentage is positive
        profitable_rides = data[data['profit_percentage'] > 0]

        # Identify loss rides where profit percentage is negative
        loss_rides = data[data['profit_percentage'] < 0]


        import plotly.graph_objects as go

        # Calculate the count of profitable and loss rides
        profitable_count = len(profitable_rides)
        loss_count = len(loss_rides)

        # Create a donut chart to show the distribution of profitable and loss rides
        labels = ['Profitable Rides', 'Loss Rides']
        values = [profitable_count, loss_count]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
        fig.update_layout(title='Profitability of Rides (Dynamic Pricing vs. Historical Pricing)')
        fig.show()
    """

    st.code(notebook_code, language='python')

with col2:
    # Calculate the profit percentage for each ride
    data['profit_percentage'] = ((data['adjusted_ride_cost'] - data['Historical_Cost_of_Ride']) / data['Historical_Cost_of_Ride']) * 100
    # Identify profitable rides where profit percentage is positive
    profitable_rides = data[data['profit_percentage'] > 0]

    # Identify loss rides where profit percentage is negative
    loss_rides = data[data['profit_percentage'] < 0]


    import plotly.graph_objects as go

    # Calculate the count of profitable and loss rides
    profitable_count = len(profitable_rides)
    loss_count = len(loss_rides)

    # Create a donut chart to show the distribution of profitable and loss rides
    labels = ['Profitable Rides', 'Loss Rides']
    values = [profitable_count, loss_count]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig.update_layout(title='Profitability of Rides (Dynamic Pricing vs. Historical Pricing)')

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
    <div class="custom-subheader">Real vs Estrategia</div>
    """, unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    #st.image("img/demanda.png", caption="Descripción de la imagen", width=500)  # Mostrar algunos datos


    st.markdown("<br>Ahora veamos la relación entre la duración esperada del viaje y el costo del viaje según la estrategia de precios dinámicos", unsafe_allow_html=True)


    # Mostrar código del notebook
    notebook_code = """
        fig = px.scatter(data, 
                 x='Expected_Ride_Duration', 
                 y='adjusted_ride_cost',
                 title='Expected Ride Duration vs. Cost of Ride', 
                 trendline='ols')
        fig.show()
    """

    st.code(notebook_code, language='python')

with col2:
    fig = px.scatter(data, 
                 x='Expected_Ride_Duration', 
                 y='adjusted_ride_cost',
                 title='Duración esperada del viaje vs costo del viaje', 
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
    <div class="custom-subheader">Implementando el modelo</div>
    """, unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    #st.image("img/demanda.png", caption="Descripción de la imagen", width=500)  # Mostrar algunos datos


    st.markdown("<br><p>Paso numero 1: </p>Ahora que hemos implementado una estrategia de precios dinámicos, vamos a entrenar un modelo de aprendizaje automático. Antes de entrenar el modelo, vamos a preprocesar los datos", unsafe_allow_html=True)


    # Mostrar código del notebook
    notebook_code = """
        def data_preprocessing_pipeline(data):
            #Identify numeric and categorical features
            numeric_features = data.select_dtypes(include=['float', 'int']).columns
            categorical_features = data.select_dtypes(include=['object']).columns

            #Handle missing values in numeric features
            data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

            #Detect and handle outliers in numeric features using IQR
            for feature in numeric_features:
                Q1 = data[feature].quantile(0.25)
                Q3 = data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (1.5 * IQR)
                upper_bound = Q3 + (1.5 * IQR)
                data[feature] = np.where((data[feature] < lower_bound) | (data[feature] > upper_bound),
                                        data[feature].mean(), data[feature])

            #Handle missing values in categorical features
            data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

            return data
    """

    st.code(notebook_code, language='python')

with col2:
    st.markdown("<br><p>Paso numero 2: </p>Como el tipo de vehículo es un factor valioso, vamos a convertirlo en una característica numérica antes de continuar:", unsafe_allow_html=True)
 # Mostrar código del notebook
    notebook_code = """
        data["Vehicle_Type"] = data["Vehicle_Type"].map({"Premium": 1, "Economy": 0})

        #Ahora dividamos los datos y entrenemos un modelo de aprendizaje automático para predecir el costo de un viaje:

        from sklearn.model_selection import train_test_split
        x = np.array(data[["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type", "Expected_Ride_Duration"]])
        y = np.array(data[["adjusted_ride_cost"]])

        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        # Reshape y to 1D array
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        # Training a random forest regression model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
    """
    st.code(notebook_code, language='python')