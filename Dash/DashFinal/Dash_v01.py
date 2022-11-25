
## *: Importante: Belen es espectacular
## TODO: Margenes
## TODO: Centrado pagina web
## TODO: Zoom y centrado plot



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

import math
import plotly.express as px
from PIL import Image



def returnImage(direccion):
    """Devuelve la imagen a representar con una direccion no total

    Args:
        direccion (str): direccion local

    Returns:
        Imagen: Devuelve la imagen a representar
    """
    return Image.open(str(os.getcwd())+str(direccion))



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



figuraRentabilidadDistritos = go.Figure()
figuraRentabilidadDistritos.add_trace(trace=go.Choroplethmapbox(
                                                geojson=jsonGeoNeigh,
                                                featureidkey='properties.neighbourhood',
                                                locations=listings_filtered_df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                                                z=listings_filtered_df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['profitability'],
                                                colorscale=px.colors.sequential.YlGnBu,
                                                #colorscale=px.colors.diverging.balance,
                                                #zmin=zmin,zmax=zmax,
                                                colorbar=dict(thickness=10, ticklen=1,title="%",tickformat='1%',tickcolor='grey',tickfont=dict(size=14, color='grey'),titlefont=dict(color='grey')),
                                                text=listings_filtered_df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                                                hovertemplate = "<b>%{text}</b><br>" +
                                                                "Avg. return: %{z:.0%}<br>" +
                                                                "<extra></extra>"
                                            )
                                      )
figuraRentabilidadDistritos.update_layout(mapbox1=dict(zoom=8.5,style='carto-positron',center={"lat": 40.7, "lon": -74}))
#figuraRentabilidadDistritos.data[0].colorbar.x=-0.1
figuraRentabilidadDistritos.update_layout(height=300,width=400,margin=dict(t=0,b=0,l=0,r=0),title="Rentabilidad (%) media",
                                          paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)')    

figuraPrecioDistritos = go.Figure()
figuraPrecioDistritos.add_trace(trace=go.Choroplethmapbox(
                                                geojson=jsonGeoNeigh,
                                                featureidkey='properties.neighbourhood',
                                                locations=listings_filtered_df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                                                z=listings_filtered_df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['price'],
                                                colorscale=px.colors.sequential.YlGnBu,
                                                #colorscale=px.colors.diverging.balance,
                                                #zmin=zmin,zmax=zmax,
                                                colorbar=dict(thickness=10, ticklen=1,title="$",tickformat='1$',tickcolor='grey',tickfont=dict(size=14, color='grey'),titlefont=dict(color='grey')),
                                                text=listings_filtered_df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                                                hovertemplate = "<b>%{text}</b><br>" +
                                                                "Avg. price: %{z:.0$}<br>" +
                                                                "<extra></extra>"
                                            )
                                      )
figuraPrecioDistritos.update_layout(mapbox1=dict(zoom=8.5,style='carto-positron',center={"lat": 40.7, "lon": -74}))
figuraPrecioDistritos.update_layout(height=300,width=400,margin=dict(t=0,b=0,l=0,r=0),title="Precio ($) media",
                                          paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)')    
 


app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])

logging.getLogger('werkzeug').setLevel(logging.INFO)
dash.register_page(__name__, path='/')


""" 
checklist = html.Div(
    [
        dbc.Label("Elige"),
        dbc.Checklist(
            options=[
                {"label": "Option 1", "value": 1},
                {"label": "Option 2", "value": 2},
                {"label": "Disabled Option", "value": 3, "disabled": True},
            ],
            value=[1],
            id="checklist-input",
        ),
    ]
) 
"""


itemsDropDownBarrios = [
        dbc.DropdownMenuItem("Todos"),
        dbc.DropdownMenuItem("Manhattan"),
        dbc.DropdownMenuItem("Brooklyn"),
        dbc.DropdownMenuItem("Queens"),
        dbc.DropdownMenuItem("Staten Island"),
        dbc.DropdownMenuItem("Bronx"),
    ]






app.layout = dbc.Container(
    [
        #dcc.Store(id="store"), # para guardar informacion. Es una variable
        
        dbc.Row([
                    html.Img(src=returnImage('\Images\AirBnB\logoRojo.png'),
                             style={
                                 "display":"inline-block",
                                 "width":"15%",
                                 "padding-right":"2%"
                             }
                            ),
                    html.H1("Estudio sobre los AirBnBs en NYC",
                            style={
                                    "display":"inline-block",
                                    "vertical-align": "bottom"
                                }),
                ],
                id = "Titulo",
                style ={
                    "align": "left",
                    'display':'inline-block',
                    "padding-top":"2%",
                    "width":"100%",
                    #"margin":"5%"
                }
        ),
        html.Hr(), # Cambio de tercio
        dbc.Row([
            dbc.Col([
                    dbc.Row([
                            html.H3("Rentabilidad (%) por distrito",
                                    style={
                                        "size":18,
                                        "color":"grey",
                                        "padding-bottom":"2%"
                                    },
                                    ),
                            dcc.Graph(figure=figuraRentabilidadDistritos)
                        ],
                            id="plt-profitability"
                    ),
                    dbc.Row([
                            html.H3("Precio ($) por distrito",
                                    style={
                                        "size":18,
                                        "color":"grey",
                                        "padding-bottom":"2%",
                                        "padding-top":"2%"
                                    },
                                    ),
                            dcc.Graph(figure=figuraPrecioDistritos)
                        ],
                            id="plt-price"
                    ),
                    dbc.Row([
                        html.H3("Filtros",
                                style={
                                        "display":"block",
                                        "vertical-align": "bottom",
                                        "padding-top":"1%"
                                    }),
                        html.Hr(), # Cambio de tercio
                        dbc.Label("Filtro rentabilidad (%)", html_for="range-slider"),
                        dcc.RangeSlider(id="range-slider-rentabilidad", min=0, max=20, value=[0, 3.5],
                                        
                                        #marks={i: '{}'.format((math.sqrt(i))) for i in [0.1,1,3,5,20]},
                                        marks={
                                            0: '0%',
                                            5: '5%',
                                            10: '10%',
                                            20: '20%',
                                            },
                                        
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        
                                        ),
                        html.Br(),
                        dbc.Label("Filtro Precio/noche ($)", html_for="range-slider",style={"padding-top":"1.5%"}),
                        dcc.RangeSlider(id="range-slider-precio", min=0, max=1300, value=[0, 400],
                                        
                                        #marks={i: '{}'.format((math.sqrt(i))) for i in [0.1,1,3,5,20]},
                                        marks={
                                            0: '0$',
                                            100: '100$',
                                            200: '200$',
                                            500: '500$',
                                            1300: '1300$',
                                            },
                                        
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        
                                        ),
                        
                        
                    ]),
                    dbc.Row([
                        html.Br(),
                        dbc.Col([dcc.Dropdown(
                                    id="barrios-seleccion",
                                    options=[
                                        "Todos",
                                        "Manhattan",
                                        "Brooklyn",
                                        "Queens",
                                        "Staten Island",
                                        "Bronx"
                                    ],
                                    value="Todos",
                                    style={
                                        "width":"80%",
                                        "padding-top":"3%"
                                    }
                                    ),
                                ],
                                id="seleccionBarrios",
                                style={
                                    'display':'inline-block',
                                },
                                width=8
                        ),
                        dbc.Col([html.Div(
                                        [
                                            dbc.Checklist(
                                                options=[
                                                    {"label": "Aplicar Filtros", "value": 1},
                                                ],
                                                value=[0],
                                                id="switches-input",
                                                switch=True,
                                                style={
                                                    "vertical-align":"middle",
                                                    "padding-top":"3%",
                                                    #"background":"grey",
                                                },
                                                input_checked_style={
                                                    "background-color":"#bada55"
                                                    
                                                },
                                            ),
                                        ]
                                    )
                        ],
                                width=4)
                    ]),
                    
            ]),
            dbc.Col([
                    dbc.Tabs([
                                dbc.Tab(label="Rentabilidad", tab_id="profitability"),
                                dbc.Tab(label="Descriptivo", tab_id="descriptive"),
                                dbc.Tab(label="Bonus", tab_id="bonus"),
                            ],
                            id="tabs",
                            active_tab="profitability",
                            )],
                    width=8,
                    )
            
        ])
    ],
    style={
        "width":"100%",
        #"margin":"3%",
        #"align":"center"
    }
)



def filtrarDF(rentabilidadMin,rentabilidadMax,barrio,precioMin,precioMax):
    """Devuelve DF filtrado con los posibles filtros existentes

    Args:
        rentabilidadMin (float): _description_
        rentabilidadMax (float): _description_
        barrio (str): Barrio o Todo
        precioMin (float): _description_
        precioMax (float): _description_

    Returns:
        DF: DF filtrado
    """    
    df = listings_filtered_df
    if barrio!="Todos":
        df = df[df["neighbourhood_group_cleansed"]==barrio]
    z = df[
           ((df["profitability"]>=rentabilidadMin)&(df["profitability"]<=rentabilidadMax))
           &
           ((df["price"]>=precioMin)&(df["profitability"]<=precioMax))
        ]
    return z
    
        





































""" 
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    '''
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    '''
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
 """
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


