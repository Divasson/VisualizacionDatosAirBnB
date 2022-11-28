
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
import pickle


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

# load ML model
rf_model = pickle.load(open(str(os.getcwd())+str("\\modelling\\random_forest_model.pickle"), 'rb'))

#################################################################################################################################################################################################
####################################################################################### FUNCIONES ###############################################################################################
#################################################################################################################################################################################################

def returnImage(direccion):
    """Devuelve la imagen a representar con una direccion no total

    Args:
        direccion (str): direccion local

    Returns:
        Imagen: Devuelve la imagen a representar
    """
    return Image.open(str(os.getcwd())+str(direccion))


def graph_rentabilidad_distritos(df):
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        figuraRentabilidadDistritos(figure): mapa rentabilidad distritos
    """
    figuraRentabilidadDistritos = go.Figure()
    figuraRentabilidadDistritos.add_trace(trace=go.Choroplethmapbox(
                                                geojson=jsonGeoNeigh,
                                                featureidkey='properties.neighbourhood',
                                                locations=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                                                z=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['profitability'],
                                                colorscale=px.colors.sequential.YlGnBu,
                                                #colorscale=px.colors.diverging.balance,
                                                #zmin=zmin,zmax=zmax,
                                                colorbar=dict(thickness=10, ticklen=1,title="%",tickformat='1%',tickcolor='grey',tickfont=dict(size=14, color='grey'),titlefont=dict(color='grey')),
                                                text=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
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
    
    return figuraRentabilidadDistritos

def graph_rentabilidad_barrios(df):
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        figuraRentabilidadDistritos(figure): mapa rentabilidad distritos
    """
    figuraRentabilidadBarrios = go.Figure()
    figuraRentabilidadBarrios.add_trace(trace=go.Choroplethmapbox(
                                                geojson=bigJSONNeigh,
                                                featureidkey='properties.neighbourhood_group',
                                                locations=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                                                z=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['profitability'],
                                                colorscale=px.colors.sequential.YlGnBu,
                                                #colorscale=px.colors.diverging.balance,
                                                #zmin=zmin,zmax=zmax,
                                                colorbar=dict(thickness=10, ticklen=1,title="%",tickformat='1%',tickcolor='grey',tickfont=dict(size=14, color='grey'),titlefont=dict(color='grey')),
                                                text=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                                                hovertemplate = "<b>%{text}</b><br>" +
                                                                "Avg. return: %{z:.0%}<br>" +
                                                                "<extra></extra>"
                                            )
                                      )
    figuraRentabilidadBarrios.update_layout(mapbox1=dict(zoom=8.5,style='carto-positron',center={"lat": 40.7, "lon": -74}))
    #figuraRentabilidadDistritos.data[0].colorbar.x=-0.1
    figuraRentabilidadBarrios.update_layout(height=300,width=400,margin=dict(t=0,b=0,l=0,r=0),title="Rentabilidad (%) media",
                                          paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)')    
    
    return figuraRentabilidadBarrios

def graph_precio_distritos(df):
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        figuraPrecioDistritos(figure): mapa precio distritos
    """

    figuraPrecioDistritos = go.Figure()
    figuraPrecioDistritos.add_trace(trace=go.Choroplethmapbox(
                                                geojson=jsonGeoNeigh,
                                                featureidkey='properties.neighbourhood',
                                                locations=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                                                z=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['price'],
                                                colorscale=px.colors.sequential.YlGnBu,
                                                #colorscale=px.colors.diverging.balance,
                                                #zmin=zmin,zmax=zmax,
                                                colorbar=dict(thickness=10, ticklen=1,title="$",tickformat='1$',tickcolor='grey',tickfont=dict(size=14, color='grey'),titlefont=dict(color='grey')),
                                                text=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                                                hovertemplate = "<b>%{text}</b><br>" +
                                                                "Avg. price: %{z:.0$}<br>" +
                                                                "<extra></extra>"
                                            )
                                      )
    figuraPrecioDistritos.update_layout(mapbox1=dict(zoom=8.5,style='carto-positron',center={"lat": 40.7, "lon": -74}))
    figuraPrecioDistritos.update_layout(height=300,width=400,margin=dict(t=0,b=0,l=0,r=0),title="Precio ($) media",
                                          paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)')    
    
    return figuraPrecioDistritos

def graph_precio_barrios(df):
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        figuraPrecioDistritos(figure): mapa precio distritos
    """

    figuraPrecioBarrios = go.Figure()
    figuraPrecioBarrios.add_trace(trace=go.Choroplethmapbox(
                                                geojson=bigJSONNeigh,
                                                featureidkey='properties.neighbourhood_group',
                                                locations=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                                                z=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['price'],
                                                colorscale=px.colors.sequential.YlGnBu,
                                                #colorscale=px.colors.diverging.balance,
                                                #zmin=zmin,zmax=zmax,
                                                colorbar=dict(thickness=10, ticklen=1,title="$",tickformat='1$',tickcolor='grey',tickfont=dict(size=14, color='grey'),titlefont=dict(color='grey')),
                                                text=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                                                hovertemplate = "<b>%{text}</b><br>" +
                                                                "Avg. price: %{z:.0$}<br>" +
                                                                "<extra></extra>"
                                            )
                                      )
    figuraPrecioBarrios.update_layout(mapbox1=dict(zoom=8.5,style='carto-positron',center={"lat": 40.7, "lon": -74}))
    figuraPrecioBarrios.update_layout(height=300,width=400,margin=dict(t=0,b=0,l=0,r=0),title="Precio ($) media",
                                          paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)')    
    
    return figuraPrecioBarrios
    
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

def graph_subplot_rentabilidad(df):
    fig = make_subplots(
        rows = 1,
        cols = 2,
        specs=[[{'type':'mapbox'}, {'type':'mapbox'}]], # Necesario para agregar un piechart
        subplot_titles = (
            "Rentabilidad media (%) por distrito",
            "Rentabilidad media (%) por barrio"
        )
    )

    #fig.show()
    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=jsonGeoNeigh,
                            featureidkey='properties.neighbourhood',
                            locations=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            z=100*(df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['profitability']),
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=0.46,title="%", tickformat='1$', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white'))
                        ),
                row=1,
                col=1    
    )
    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=bigJSONNeigh,
                            featureidkey='properties.neighbourhood_group', 
                            locations=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            z=100*(df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['profitability']),
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=1.02,title="%", tickformat='1$', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white'))
                        ),
                row=1,
                col=2           
    )
    fig.update_layout(mapbox1=dict(zoom=9.5,style='carto-positron',center={"lat": 40.7, "lon": -74}),
                    mapbox2=dict(zoom=9.5,style='carto-positron',center={"lat": 40.7, "lon": -74}))

    fig.update_layout(height=1000,width=2200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=12))    
    #fig.update_geos(fitbounds="locations",visible=True)
    fig.show()

    return fig

def predictPrice(barrio,lat,lon,acco,bed,bath,wifi,kitchen,dryer,heating,tv):
    
    latitude = float(lat),
    longitude = float(lon),
    accommodates = float(acco),
    beds = float(bed),
    baths = float(bath),
    has_wifi = wifi,
    has_kitchen = kitchen,
    has_dryer = dryer,
    has_heating = heating,
    has_tv = tv,

    if str(barrio).lower() == "manhattan":
        neighbourhood_group_cleansed_Manhattan = 1,
        neighbourhood_group_cleansed_Brooklyn = 0,
        neighbourhood_group_cleansed_Bronx = 0,
        neighbourhood_group_cleansed_Queens = 0,
        neighbourhood_group_cleansed_Staten_Island = 0
    elif str(barrio).lower() == "brooklyn":
        neighbourhood_group_cleansed_Manhattan = 0,
        neighbourhood_group_cleansed_Brooklyn = 1,
        neighbourhood_group_cleansed_Bronx = 0,
        neighbourhood_group_cleansed_Queens = 0,
        neighbourhood_group_cleansed_Staten_Island = 0
    elif str(barrio).lower() == "bronx":
        neighbourhood_group_cleansed_Manhattan = 0,
        neighbourhood_group_cleansed_Brooklyn = 0,
        neighbourhood_group_cleansed_Bronx = 1,
        neighbourhood_group_cleansed_Queens = 0,
        neighbourhood_group_cleansed_Staten_Island = 0
    elif str(barrio).lower() == "queens": 
        neighbourhood_group_cleansed_Manhattan = 0,
        neighbourhood_group_cleansed_Brooklyn = 0,
        neighbourhood_group_cleansed_Bronx = 0,
        neighbourhood_group_cleansed_Queens = 1,
        neighbourhood_group_cleansed_Staten_Island = 0
    elif str(barrio).lower() == "staten island": 
        neighbourhood_group_cleansed_Manhattan = 0,
        neighbourhood_group_cleansed_Brooklyn = 0,
        neighbourhood_group_cleansed_Bronx = 0,
        neighbourhood_group_cleansed_Queens = 0,
        neighbourhood_group_cleansed_Staten_Island = 1
    else:
        neighbourhood_group_cleansed_Manhattan = 0,
        neighbourhood_group_cleansed_Brooklyn = 0,
        neighbourhood_group_cleansed_Bronx = 0,
        neighbourhood_group_cleansed_Queens = 0,
        neighbourhood_group_cleansed_Staten_Island = 0

    data_model = pd.DataFrame(data = [latitude, longitude, accommodates, beds, baths,
                                        has_wifi, has_dryer, has_heating, has_kitchen, has_tv,
                                        neighbourhood_group_cleansed_Bronx,
                                        neighbourhood_group_cleansed_Brooklyn,
                                        neighbourhood_group_cleansed_Manhattan,
                                        neighbourhood_group_cleansed_Queens,
                                        neighbourhood_group_cleansed_Staten_Island], 
                                columns = ['latitude', 'longitude', 'accommodates', 'beds', 'baths',
                                        'has_wifi', 'has_dryer', 'has_heating', 'has_kitchen', 'has_tv',
                                        'neighbourhood_group_cleansed_Bronx',
                                        'neighbourhood_group_cleansed_Brooklyn',
                                        'neighbourhood_group_cleansed_Manhattan',
                                        'neighbourhood_group_cleansed_Queens',
                                        'neighbourhood_group_cleansed_Staten Island'])
    
    prediction = rf_model.predict(data_model)
    return prediction


#################################################################################################################################################################################################
####################################################################################### DASH APP ################################################################################################
#################################################################################################################################################################################################

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
  
                html.Img(src=returnImage('\Images\AirBnB\logoBlanco.png'),
                    style={
                        "display":"inline-block",
                        "width":"8%",
                        "vertical-align": "center",
                        "padding-right":"1%"
                    }
                ),

                html.H1("Estudio sobre los AirBnBs en NYC",
                    style={
                        "display":"inline-block",
                        "vertical-align": "bottom",
                        "horizontal-align": "right",
                        "textAlign": "right",
                        "color": "white",
                        "fontSize":"200%"
                    }
                ),
            ],

            id = "Titulo",
            style ={
                #"align": "right",
                'display':'inline-block',
                "padding-top":"1%",
                "width":"100%",
                "height":"10%",
                "margin":"1%"
            }
        ),
        
        html.Hr(), # Cambio de tercio
        dbc.Row([
            dbc.Col([
                    dbc.Row([
                            html.H3("Rentabilidad (%) por distrito",
                                    style={
                                        "fontSize":"130%",
                                        "color":"lightgrey",
                                        "padding-bottom":"2%"
                                    },
                                    ),
                            dcc.Graph(id = "fig-profitability-districts", figure=graph_rentabilidad_distritos(listings_filtered_df))
                        ],
                            id="plt-profitability"
                    ),
                    html.Br(),
                    dbc.Row([
                            html.H3("Precio ($) por distrito",
                                    style={
                                        "fontSize":"130%",
                                        "color":"lightgrey",
                                        "padding-bottom":"2%",
                                        "padding-top":"2%"
                                    },
                                    ),
                            dcc.Graph(id = "fig-price-districts", figure = graph_precio_distritos(listings_filtered_df))
                        ],
                            id="plt-price"
                    ),
                    html.Br(),
                    html.Br(),
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
                    html.Br(),
                    html.Br(),
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
                                                value=0,
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
                                dbc.Tab(label="Predicción de precios", tab_id="model_prediction"),
                                dbc.Tab(label="Bonus", tab_id="bonus")     
                            ],
                            id="tabs",
                            active_tab="model_prediction",

                            ),
                    
                    html.Div(id='tabs-content', 
                            children=[

                            ])
                    ],

                    width=10,
                )
            
            
        ])
    ],
    fluid=True,
    style={
        "width":"95%",
        "height":"100%",
        "margin-left":"2%",
        "margin-right":"3%",
        "align":"center"
    }
)

#################################################################################################################################################################################################
####################################################################################### TAB CONTENTS#############################################################################################
#################################################################################################################################################################################################

tab_profitability_content = dbc.Card(
    dbc.CardBody(
        dcc.Graph(id="subplot-profitability",style={'width': '100%', 'height': '100%'})
    ),
)

tab_descriptive_content = dbc.Card(
    dbc.CardBody(
        
    ),
)

tab_model_prediction_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row([
                html.H4("Rellene los siguientes datos para predecir el precio por noche del Airbnb que desee",id="instructions-form", 
                            style={
                                "text-align": "center",
                                "color":"lightgrey",
                                #"padding-bottom":"2%"
                            },
                        ),

                    html.Hr()
                ]
            ),

            html.Br(),

            dbc.Row(
                [

                    dbc.Col([
                        dbc.Label("Barrio", width=10, html_for="input-barrio", style={"fontSize":"150%", "text-align": "center", "color":"lightgrey"}),
                        dbc.Input(
                            id="input-barrio", placeholder="Introduzca el barrio del Airbnb"
                        )],
                        width=5,
                        
                    ),
                    
                    
                    dbc.Col([

                        dbc.Label("Latitud", width=10, html_for="input-latitude", style={"fontSize":"150%","text-align": "center","color":"lightgrey" }),
                        dbc.Input(
                            id="input-latitude", placeholder="Introduzca la latitud del Airbnb", type="number"
                        )],
                        width=2,
                    ),

                    
                    dbc.Col([
                        dbc.Label("Longitud", width=10, html_for="input-longitude", style={"fontSize":"150%", "text-align": "center", "color":"lightgrey"}),
                        dbc.Input(
                            id="input-longitude", placeholder="Introduzca la longitud del Airbnb", type="number"
                        )],
                        width=2,
                    ),
                ],
                justify="center",
                
            ),

            html.Br(),
            html.Br(),

            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label("Nº de huéspedes", width=10, html_for="input-accommodates", style={"fontSize":"150%", "text-align": "center", "color":"lightgrey"}),
                        dbc.Input(
                            id="input-accommodates", placeholder="Introduzca el barrio del Airbnb", type="number", min=0
                        )],
                        width=2,
                        align="end",
                    ),
                    
                    
                    dbc.Col([

                        dbc.Label("Nº de camas", width=10, html_for="input-beds", style={"fontSize":"150%","text-align": "center","color":"lightgrey" }),
                        dbc.Input(
                            id="input-beds", placeholder="Introduzca la latitud del Airbnb", type="number", min=0
                        )],
                        width=2,
                        align="end",
                    ),

                    
                    dbc.Col([
                        dbc.Label("Nº de baños", width=10, html_for="input-baths", style={"fontSize":"150%", "text-align": "center", "color":"lightgrey"}),
                        dbc.Input(
                            id="input-baths", placeholder="Introduzca la longitud del Airbnb", type="number", min=0
                        )],
                        width=2,
                        align="end",
                    ),
                ],

                justify="center",
            ),

            html.Br(),
            html.Br(),

            dbc.Row(
                [
                    dbc.Col([
                        dbc.Checklist(
                            options=[
                                {"label": "Tiene Wifi", "value": "wifi"},
                                {"label": "Tiene TV", "value": "tv"},
                                {"label": "Tiene cocina", "value": "kitchen"},
                                {"label": "Tiene calefacción", "value": "heating"},
                                {"label": "Tiene secador", "value": "dryer"},
                            ],
                            value=0,
                            id="amenities-input",
                            switch=True,
                            style={
                            "align":"center",
                            "fontSize":"150%"
                            },
                            input_checked_style={
                                "background-color":"#bada55"
                                
                            },
                            labelStyle = dict(display='block', align="center")
                        ),  

                        html.Br()

                    ],
                    width=2,
                    align="center",
                    )     
                ],
               justify="center",
            ),

            html.Br(),
            html.Br(),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("ESTIMAR PRECIO", color="white", style={"color":"gray", 'backgroundColor':"white", "fontSize":"120%"},n_clicks=0, id="button-predict"),
                        width=2,
                        align = "center",
                    )
                    
                ],

                justify="center",
            ),

            html.Br(),
            html.Hr(),
            html.Br(),

            dbc.Row(
                [
                    dbc.Col(
                        html.H2("El precio por noche estimado para alojarse en un Airbnb con dichas características es de ", 
                            style={
                                "text-align": "center",
                                "color":"lightgrey",
                                #"padding-bottom":"2%"
                            },
                        ),
                    ),      
                ],
               justify="center",
            ),

            dbc.Row(
                [
                    dbc.Label("$273", id="predicted-price",
                        style={
                            "fontSize":"250%",
                            "text-align": "center",
                            "color":"white",
                        },
                    ),
                ],
               justify="center",
            ),
        ]
    ),
)

tab_bonus_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 1!", className="card-text"),
            dbc.Button("Click here", color="success"),
        ]
    ),
    className="mt-3",
)


#################################################################################################################################################################################################
####################################################################################### CALLBACKS ###############################################################################################
#################################################################################################################################################################################################

# Callback para cambiar de tab
@app.callback(
    Output("tabs-content", "children"), 
    Input("tabs", "active_tab"))
def switch_tab(tab):
    if tab == "profitability":
        return tab_profitability_content
    elif tab == "descriptive":
        return tab_descriptive_content
    elif tab == "model_prediction":
        return tab_model_prediction_content
    elif tab == "bonus":
        return tab_bonus_content
    else:
        return html.P("This shouldn't ever be displayed...")

# callback para actualizar subplot rentabilidad
@app.callback(
    Output('subplot-profitability', 'figure'),
    #Output('fig-profitability-neighbourhoods-tab', 'figure'),
    Input('range-slider-rentabilidad', 'value'),
    Input('barrios-seleccion', 'value'),
    Input('range-slider-precio', 'value'),
    Input('switches-input', 'value'),
)
def update_graph_rentabilidad(rentabilidad,barrio,precio,checkFiltros):
    """
    Args:
        rentabilidad (array-float): _description_
        barrio (str): Barrio o Todo
        precio (float): _description_
        checkFiltros (int): _description_
    
    Return:
        grpah_updated (figure): gráfico actualizado

    """
    if checkFiltros:
        #filtramos el df
        df_filtered = filtrarDF(rentabilidad[0],rentabilidad[1],barrio,precio[0],precio[1])

        return graph_subplot_rentabilidad(df_filtered)
        #return [graph_rentabilidad_distritos(df_filtered),graph_rentabilidad_barrios(df_filtered)] #devolvemos el nuevo gráfico
    
    else:
        return graph_subplot_rentabilidad(listings_filtered_df)
       #return [graph_rentabilidad_distritos(listings_filtered_df),graph_rentabilidad_barrios(listings_filtered_df)]



@app.callback(
    Output('fig-price-districts-tab', 'figure'),
    Output('fig-price-disneighbourhoods-tab', 'figure'),
    Input('range-slider-rentabilidad', 'value'),
    Input('barrios-seleccion', 'value'),
    Input('range-slider-precio', 'value'),
    Input('switches-input', 'value')
    
)
def update_graph_precio(rentabilidad,barrio,precio,checkFiltros):
    """
    Args:
        rentabilidad (float): _description_
        barrio (str): Barrio o Todo
        precio (float): _description_
        checkFiltros (int): _description_
    
    Return:
        grpah_updated (figure): gráfico actualizado

    """
    if checkFiltros:
        #filtramos el df
        df_filtered = filtrarDF(rentabilidad[0],rentabilidad[1],barrio,precio[0],precio[1])

        return [graph_precio_distritos(df_filtered), graph_precio_barrios(df_filtered)] #devolvemos el nuevo gráfico
    
    else:
        return [graph_precio_distritos(listings_filtered_df), graph_precio_barrios(listings_filtered_df)]

# callback prediccion precio
@app.callback(
    Output('predicted-price', 'value'),
    Input('input-barrio', 'value'),
    Input('input-latitude', 'value'),
    Input('input-longitude', 'value'),
    Input('input-accommodates', 'value'),
    Input('input-beds', 'value'),
    Input('input-baths', 'value'),
    Input('amenities-input', 'value'),
    Input('button-predict', 'n_clicks'),
)
def update_predicted_price(barrio,latitude,longitude,accommodates,beds,baths,amenities,button):
    """
    Args:
        barrio (str): _description_
        latitude (float): _description_
        longitude (float): _description_
        accommodates (float): _description_
        beds (float): _description_
        baths (float): _description_
        amenities (arr str): _description_
        button (int): _description_
    
    Return:
        price_predicted (str): 

    """
    wifi = 0,
    kitchen = 0,
    heating = 0,
    tv = 0,
    dryer = 0,


    if button != 0:
        #hacemos la prediccion

        if "wifi" in amenities:
            wifi = 1
        else: 
            wifi = 0

        if "dryer" in amenities:
            dryer = 1
        else:
            dryer = 0
        
        if "heating" in amenities:
            heating = 1
        else:
            heating = 0
        
        if "tv" in amenities:
            tv = 1
        else:
            tv = 0
        
        if "kitchen" in amenities:
            kitchen = 1
        else:
            kitchen = 0

        precio_pred = predictPrice(barrio,latitude,longitude,accommodates,beds,baths,wifi,kitchen,dryer,heating,tv)
        str_precio = "$" + str(precio_pred)
        return  #devolvemos el nuevo gráfico
    
    else:
        return ""



























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


