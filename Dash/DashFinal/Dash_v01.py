# Importamos las librerias mínimas necesarias
import importlib.machinery
import importlib.util
import logging
import os
import sys
from pathlib import Path
import dash

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots



# Get path to mymodule. Credits to = https://csatlas.com/python-import-file-module/
script_dir = Path( __file__ ).parent.parent
mymodule_path = str( script_dir.joinpath( '..', 'Utilities', 'leer', 'readFileFinal_y_Geos.py' ) )

# Import mymodule
loader = importlib.machinery.SourceFileLoader( 'readFileFinal_y_Geos.py', mymodule_path )
spec = importlib.util.spec_from_loader( 'readFileFinal_y_Geos.py', loader )
mymodule = importlib.util.module_from_spec( spec )
loader.exec_module( mymodule )




# Use mymodule
listings_filtered_df = mymodule.leerFicheroFinal()
(jsonGeoNeigh,bigJSONNeigh) = mymodule.leerFicherosGeo()
opcionesGlobales = mymodule.opcionesGlobales()

data = []

for x in listings_filtered_df["neighbourhood_group_cleansed"].unique():
    data.append(go.Histogram(
                    x = listings_filtered_df[listings_filtered_df["neighbourhood_group_cleansed"] == x]['price'],
                    marker_color=opcionesGlobales["Colores Barrios"][x],
                    xbins=dict(
                        start= 0,
                        end= 600,
                        size=5
                    ),
                    opacity=0.5,
                    name = x
                )
    )
    
layout = go.Layout(title = "Distribución de los precios por barrios", xaxis_title = "Precios por noche", yaxis_title = "Frecuencia",
                   barmode = "overlay", bargap = 0.1)

fig = go.Figure(data = data, layout = layout)



app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])

logging.getLogger('werkzeug').setLevel(logging.INFO)
dash.register_page(__name__, path='/')



app.layout = dbc.Container(
    [
        dcc.Store(id="store"), # para guardar informacion. Es una variable
        html.H1("Estudio sobre los AirBnBs en NYC"),
        html.Hr(), # Cambio de tercio
        dbc.Tabs(
            [
                dbc.Tab(label="Rentabilidad", tab_id="profitability"),
                dbc.Tab(label="Descriptivo", tab_id="descriptive"),
            ],
            id="tabs",
            active_tab="profitability",
        ),
        html.Div(
            children=[
                dcc.Graph(figure=fig)
            ],
            id="tab-content", 
            className="p-4"
        ) 
    ]
)

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "profitability":
            return dcc.Graph(figure=fig)
        elif active_tab == "descriptive":
            return dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig), width=6),
                    dbc.Col(dcc.Graph(figure=fig), width=6),
                ]
            )
    return fig

""" @app.callback(Output("store", "data"),[Input("button"),Input("n_clicks")])
def generate_graphs(n):
    '''
    This callback generates three simple graphs from random data.
    '''
    if not n:
        # generate empty graphs when app loads
        return fig

    # simulate expensive graph generation process
    #time.sleep(2)

    # generate 100 multivariate normal samples
    data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)

    scatter = go.Figure(
        data=[go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers")]
    )
    hist_1 = go.Figure(data=[go.Histogram(x=data[:, 0])])
    hist_2 = go.Figure(data=[go.Histogram(x=data[:, 1])]) 

    # save figures in a dictionary for sending to the dcc.Store
    return fig
 """

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)


