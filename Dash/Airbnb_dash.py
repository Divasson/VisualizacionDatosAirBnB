# Importamos las librerias mínimas necesarinoatasdas
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
import geopy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

import math
import plotly.express as px
from PIL import Image
import pickle

#import geopy


# Get path to mymodule. Credits to = https://csatlas.com/python-import-file-module/
script_dir = Path( __file__ ).parent.parent
mymodule_path = str( script_dir.joinpath('Utilities', 'leer', 'readFileFinal_y_Geos.py' ) )

# Import mymodule
loader = importlib.machinery.SourceFileLoader( 'readFileFinal_y_Geos.py', mymodule_path )
spec = importlib.util.spec_from_loader( 'readFileFinal_y_Geos.py', loader )
mymodule = importlib.util.module_from_spec( spec )
loader.exec_module( mymodule )

# Use mymodule
listings_filtered_df = mymodule.leerFicheroFinal()
criminality_df = mymodule.leerFicherosCriminality()
hosts_df = mymodule.leerFicheroHosts()
(jsonGeoNeigh,bigJSONNeigh) = mymodule.leerFicherosGeo()
opcionesGlobales = mymodule.opcionesGlobales()

# load ML model
rf_model = pickle.load(open(str(os.getcwd())+str("\\modelling\\LightGBM_model_tunned.pickle"), 'rb'))

ListingDiffSemanaFinde = pd.read_parquet((str(os.getcwd())+"\\Data\\total data\\modified data\\calendario.parquet"))
ListingDiffSemanaFindePlotear = pd.read_parquet((str(os.getcwd())+"\\Data\\total data\\modified data\\paraPlotearCalendario.parquet"))

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
    """Pinta la rentabilidad por distritos

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

def graph_rentabilidad_barrios(df): # no se utiliza
    """Pinta la rentabilidad por barrios

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
    """Pinta el precio por distritos

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
                                                colorbar=dict(thickness=10, ticklen=1,title="$",tickformat='1.0$',tickcolor='grey',tickfont=dict(size=14, color='grey'),titlefont=dict(color='grey')),
                                                text=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                                                hovertemplate = "<b>%{text}</b><br>" +
                                                                "Avg. price: %{z:1$}<br>" +
                                                                "<extra></extra>"
                                            )
                                      )
    figuraPrecioDistritos.update_layout(mapbox1=dict(zoom=8.5,style='carto-positron',center={"lat": 40.7, "lon": -74}))
    figuraPrecioDistritos.update_layout(height=300,width=400,margin=dict(t=0,b=0,l=0,r=0),title="Precio ($) media",
                                          paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)')    
    
    return figuraPrecioDistritos

def graph_precio_barrios(df): # no se utiliza
    """Pinta el precio por barrios

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
                                                colorbar=dict(thickness=10, ticklen=1,title="$",tickformat='1.0$',tickcolor='grey',tickfont=dict(size=14, color='grey'),titlefont=dict(color='grey')),
                                                text=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                                                hovertemplate = "<b>%{text}</b><br>" +
                                                                "Avg. price: %{z:1}$<br>" +
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
        rentabilidadMin (float): filtro de rentabilidad min
        rentabilidadMax (float): filtro de rentabilidad max
        barrio (str): Barrio o Todo
        precioMin (float): filtro de precio min
        precioMax (float): filtro de precio max

    Returns:
        DF: DF filtrado
    """    
    #pasamos los porcentages a float
    rentabilidadMin=rentabilidadMin/100
    rentabilidadMax=rentabilidadMax/100
    

    df = listings_filtered_df
    if barrio!="Todos":
        df = df[df["neighbourhood_group_cleansed"]==barrio]
    z = df[
            ((df["profitability"]>=rentabilidadMin)&(df["profitability"]<=rentabilidadMax))
            &
            ((df["price"]>=precioMin)&(df["price"]<=precioMax))
        ]
    
    return z 

def graph_subplot_rentabilidad(df,barrio):
    """Pinta la rentabilidad en función de filtros

    Args:
        df (DataFrame): dataframe filtrado
        barrio (str): barrio seleccionado

    Returns:
        fig: figura a representar
    """
    fig = make_subplots(
        rows = 1,
        cols = 2,
        specs=[[{'type':'mapbox'}, {'type':'mapbox'}]], # Necesario para agregar un piechart
        subplot_titles = (
            "<b>Rentabilidad media (%) por distrito<b>",
            "<b>Rentabilidad media (%) por barrio<b>"
        )
    )

    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=jsonGeoNeigh,
                            featureidkey='properties.neighbourhood',
                            locations=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            z=(df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['profitability']),
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=0.46,title="%", tickformat='1.0%', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white')),
                            text=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            "Avg. return: %{z:.1%}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=1    
    )
    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=bigJSONNeigh,
                            featureidkey='properties.neighbourhood_group', 
                            locations=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            z=(df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['profitability']),
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=1.02,title="%", tickformat='1.0%', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white')),
                            text=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            "Avg. return: %{z:.1%}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=2           
    )
    fig.update_layout(mapbox1=dict(zoom=opcionesGlobales["Centros"][barrio][2],style='carto-positron',center={"lat": opcionesGlobales["Centros"][barrio][0], "lon": opcionesGlobales["Centros"][barrio][1]}),
                    mapbox2=dict(zoom=opcionesGlobales["Centros"][barrio][2],style='carto-positron',center={"lat": opcionesGlobales["Centros"][barrio][0], "lon": opcionesGlobales["Centros"][barrio][1]}))

    fig.update_layout(height=1000,width=2200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=20))    

    return fig

def graph_subplot_prices(df,barrio):
    fig = make_subplots(
        rows = 1,
        cols = 2,
        specs=[[{'type':'mapbox'}, {'type':'mapbox'}]], # Necesario para agregar un piechart
        subplot_titles = (
            "<b>Precio medio por noche por distrito<b>",
            "<b>Precio medio por noche por barrio<b>"
        )
    )

    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=jsonGeoNeigh,
                            featureidkey='properties.neighbourhood',
                            locations=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            z=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['price'],
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=0.46,title="$", tickformat='1$', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white')),
                            text=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            "Avg. price: $%{z:.1$}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=1    
    )
    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=bigJSONNeigh,
                            featureidkey='properties.neighbourhood_group', 
                            locations=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            z=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['price'],
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=1.02,title="$", tickformat='1$', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white')),
                            text=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            "Avg. price: $%{z:.1$}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=2           
    )
    fig.update_layout(mapbox1=dict(zoom=opcionesGlobales["Centros"][barrio][2],style='carto-positron',center={"lat": opcionesGlobales["Centros"][barrio][0], "lon": opcionesGlobales["Centros"][barrio][1]}),
                    mapbox2=dict(zoom=opcionesGlobales["Centros"][barrio][2],style='carto-positron',center={"lat": opcionesGlobales["Centros"][barrio][0], "lon": opcionesGlobales["Centros"][barrio][1]}))

    fig.update_layout(height=800,width=2200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=20))    

    return fig

def graph_histogram_prices(df):
    data = []
    colorsBarrios = {
        "Brooklyn": "#1f77b4",
        "Bronx": "#ff7f0e",
        "Staten Island":"#2ca02c",
        "Queens":"#9467bd",
        "Manhattan":"#d62728"
    }

    for x in df["neighbourhood_group_cleansed"].unique():
        data.append(go.Histogram(
                        x = df[df["neighbourhood_group_cleansed"] == x]['price'],
                        marker_color=colorsBarrios[x],
                        xbins=dict(
                            start= 0,
                            end= 600,
                            size=5
                        ),
                        opacity=0.5,
                        name = x
                    )
        )
        
    layout = go.Layout(title = "<b>Distribución de los precios por barrios<b>", xaxis_title = "Precios por noche", yaxis_title = "Frecuencia",
                    barmode = "overlay", bargap = 0.1, height=450, width=2200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=12))

    fig = go.Figure(data = data, layout = layout)

    return fig

def graph_subplot_occupancy_rates(df,barrio):
    """Pinta el grafico de ratio de ocupación filtrado

    Args:
        df (DataFrame): dataframe filtrado
        barrio (str): barrio seleccionado

    Returns:
        fig: figura a representar
    """
    fig = make_subplots(
        rows = 1,
        cols = 2,
        specs=[[{'type':'mapbox'}, {'type':'mapbox'}]], # Necesario para agregar un piechart
        subplot_titles = (
            "<b>Occupancy rate medio (%) por distrito<b>",
            "<b>Occupancy rate medio (%) por barrio<b>"
        )
    )

    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=jsonGeoNeigh,
                            featureidkey='properties.neighbourhood',
                            locations=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            z=(df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['occupancy_rate']),
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=0.46,title="%", tickformat='1%', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white')),
                            text=df.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            "Avg. rate: %{z:.1%}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=1    
    )
    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=bigJSONNeigh,
                            featureidkey='properties.neighbourhood_group', 
                            locations=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            z=(df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['occupancy_rate']),
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=1.02,title="%", tickformat='1%', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white')),
                            text=df.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            "Avg. rate: %{z:.1%}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=2           
    )
    fig.update_layout(mapbox1=dict(zoom=opcionesGlobales["Centros"][barrio][2],style='carto-positron',center={"lat": opcionesGlobales["Centros"][barrio][0], "lon": opcionesGlobales["Centros"][barrio][1]}),
                    mapbox2=dict(zoom=opcionesGlobales["Centros"][barrio][2],style='carto-positron',center={"lat": opcionesGlobales["Centros"][barrio][0], "lon": opcionesGlobales["Centros"][barrio][1]}))

    fig.update_layout(height=1000,width=2200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=20))    

    return fig

def graph_subplot_criminality(df):
    """Pinta el grafico inmutable de criminalidad

    Args:
        df (DataFrame): dataframe previamente tratado

    Returns:
        fig: figura a representar
    """
    GranTitulo = "Crimenes en NYC"
    title = "<i>No se aplican filtros a estas visualizaciones<i>"
    nombreBarra = "# crímenes"
    columna_a_mirar = "OFNS_DESC"
    OFNS_DESC= "MURDER & NON-NEGL. MANSLAUGHTER"

    #zmin= min(NYC_crime[NYC_crime["OFNS_DESC"]==OFNS_DESC].groupby("neighbourhood",as_index=False).count()[columna_a_mirar])
    #zmax= max(NYC_crime[NYC_crime["OFNS_DESC"]==OFNS_DESC].groupby("neighbourhood",as_index=False).count()[columna_a_mirar])

    fig = make_subplots(
        rows = 1,
        cols = 2,
        specs=[[{'type':'mapbox'}, {'type':'mapbox'}]], # Necesario para agregar un piechart
        subplot_titles = (
            "<b>Media anual de asesinatos por distrito<b>",
            "<b>Media anual de crímenes sexuales por distrito<b>"
        )
    )

    #fig.show()
    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=jsonGeoNeigh,
                            featureidkey='properties.neighbourhood',
                            locations=df[df["OFNS_DESC"]=="MURDER & NON-NEGL. MANSLAUGHTER"].groupby("neighbourhood",as_index=False).count()['neighbourhood'],
                            z=df[df["OFNS_DESC"]=="MURDER & NON-NEGL. MANSLAUGHTER"].groupby("neighbourhood",as_index=False).count()[columna_a_mirar],
                            colorscale=px.colors.sequential.Reds,
                            #colorscale=px.colors.diverging.balance,
                            #zmin=zmin,zmax=zmax,
                            colorbar=dict(thickness=20, x=0.46,title=nombreBarra,tickformat='1',ticklen=3),
                            text=df[df["OFNS_DESC"]=="MURDER & NON-NEGL. MANSLAUGHTER"].groupby("neighbourhood",as_index=False).count()['neighbourhood'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            title+": %{z:1}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=1    
    )

    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=jsonGeoNeigh,
                            featureidkey='properties.neighbourhood', 
                            locations=df[(df["OFNS_DESC"]=="RAPE")|(df["OFNS_DESC"]=="SEX CRIMES")].groupby("neighbourhood",as_index=False).count()['neighbourhood'],
                            z=df[(df["OFNS_DESC"]=="RAPE")|(df["OFNS_DESC"]=="SEX CRIMES")].groupby("neighbourhood",as_index=False).count()[columna_a_mirar],
                            colorscale=px.colors.sequential.Reds,
                            #zmin=zmin,zmax=zmax,
                            colorbar=dict(thickness=20, x=1.02,title=nombreBarra,tickformat='1',ticklen=3),
                            text=df[(df["OFNS_DESC"]=="RAPE")|(df["OFNS_DESC"]=="SEX CRIMES")].groupby("neighbourhood",as_index=False).count()['neighbourhood'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            title+": %{z:1}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=2           
    )


    fig.update_layout(mapbox1=dict(zoom=9.5,style='carto-positron',center={"lat": 40.7, "lon": -74})
                    ,mapbox2=dict(zoom=9.5,style='carto-positron',center={"lat": 40.7, "lon": -74})
                    )
    #x=0.5,y=0.95
    fig.update_layout(height=1000,width=2200,paper_bgcolor='rgba(0,0,0,0)', title = dict(text=title, x=1,y=0.98,font=dict(size=15,color='red')), plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=15))    
    return fig

def graph_bar_hosts_time_overall(df):
    """Pintar el tiempo de respuesta de los hosts

    Args:
        df (DataFrame): DataFrame filtrado

    Returns:
        fig: figura a representar
    """
    
    diccionarioColoresTiempo={
        #dusty teal
        "a few days or more":"#1BA597",
        "within a day":"#AEE78D",
        "within a few hours":"#F4F39B",
        "within an hour":"#F3CE88"
    }

    
    level_count = pd.DataFrame(df.groupby("host_response_time")["host_is_superhost"].value_counts()).rename(columns = {"host_is_superhost": "count"}).reset_index()
    group_count = pd.DataFrame(level_count.groupby(["host_response_time"])["count"].sum()).reset_index()
    level_count=level_count.merge(group_count, on='host_response_time', how='left').rename(columns = {"count_x": "count","count_y": "total"})
    level_count

    nivelesIntermedios = level_count["host_response_time"].unique().tolist()
    lista1 = []
    lista2 = []
    lista3 = []
    lista7 = []


    for index,tiempo in enumerate(nivelesIntermedios):
        lista1.append(index+1)
        lista2.append(len(nivelesIntermedios)+1)
        #print(level_count[(level_count["host_response_time"]==tiempo)&(level_count["host_is_superhost"]=='t')]["count"].values)
        print()
        
        try:
            lista3.append(level_count[(level_count["host_response_time"]==tiempo)&(level_count["host_is_superhost"]=='t')]["count"].values[0])
        except:
            lista3.append(0)
        
        lista7.append("gold")
        
        lista1.append(index+1)
        lista2.append(len(nivelesIntermedios)+2)
        #print(level_count[(level_count["host_response_time"]==tiempo)&(level_count["host_is_superhost"]=='f')]["count"].values)
        try:
            lista3.append(level_count[(level_count["host_response_time"]==tiempo)&(level_count["host_is_superhost"]=='f')]["count"].values[0])
        except:
            lista3.append(0)
        
        
        lista7.append("lightgray")
    lista4 = []
    lista5 = []
    lista6 = []

    for index,tiempo in enumerate(nivelesIntermedios):
        lista4.append(0)
        lista5.append(index+1)
        lista6.append(level_count[(level_count["host_response_time"]==tiempo)]["total"].values[0])
        lista7.append(diccionarioColoresTiempo[tiempo])
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = ["Todos los hosts"]+nivelesIntermedios+["SuperHost","Host"],
            color = "white"
        ),
        link = dict(
            source = lista1+lista4, 
            target = lista2+lista5,
            value = lista3+lista6,
            color= lista7
        ))])

    fig.update_layout(height=750,width=2200,paper_bgcolor='rgba(0,0,0,0)', title = dict(text="<b>Distribución tiempo de respuesta para host y superhost<b>", x=0.5,y=0.95,font=dict(size=17,color='white')), plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=17))    



    return fig

def graph_table_hosts(df):
    """Pinta la tabla de hosts

    Args:
        df (DataFrame): dataframe entero

    Returns:
        fig: figura a representar
    """
    filtered_data = df.nlargest(10, 'Total de Airbnbs')
    filtered_data = filtered_data.drop(["ID host","Nombre host"],axis=1)
    column_host = ["Host 1", "Host 2", "Host 3", "Host 4", "Host 5", "Host 6", "Host 7", "Host 8", "Host 9", "Host 10"]
    filtered_data.insert(0, "Host",column_host)

    fig = go.Figure()

    fig.add_trace(
        go.Table(
            header=dict(
                values=filtered_data.columns,
                line_color='white',
                fill_color='#4D5656',
                font=dict(size=17, color="#D7DBDD"),
                align=['center'],
            ),
            cells=dict(
                values= [filtered_data[k].tolist() for k in filtered_data.columns],
                align = ['center'],
                font=dict(size=16, color="#4D5656"),
                height=30
            )

        ),
    )

    fig.update_layout(height=500,width=2200,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=18), 
                    title = dict(text="<b>Información sobre los top 10 host de Nueva York<b>",x=0.5,y=0.90, font=dict(size=17)))

    return fig

def graph_pie_property_type(df):
    """Pinta la grafica de tarta con tipos de casas

    Args:
        df (DataFrame): DataFrame filtrado

    Returns:
        fig: figura a representar
    """
    tiposDeListing = ["Entire rental unit", "Private room in rental unit", "Private room in home", "Entire condo", "Entire home"]
    fig = px.pie(data_frame=df[df['property_type'].isin(tiposDeListing)].groupby("property_type",as_index=False).count(), 
             values="id",names="property_type", color = "property_type",color_discrete_sequence = px.colors.sequential.YlGnBu,
             )
    fig.update_layout(title=dict(text="<b>Tipos de viviendas en NYC<b>", x=0.20,y=0.97, font=dict(size=17)),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=15), height=500,width=734)

    return fig

def graph_spider_features(df,barrio):
    """Pinta el grafico de araña de los AirBnBs

    Args:
        df (DataFrame): dataframe filtrado
        barrio (str): barrio seleccionado

    Returns:
        fig: figura a representar
    """
    colorsBarrios = {
            "Brooklyn": "#1f77b4",
            "Bronx": "#ff7f0e",
            "Staten Island":"#2ca02c",
            "Queens":"#9467bd",
            "Manhattan":"#d62728"
        }

    lista = ["occupancy_rate",
        "beds",
        "price",
        "baths",
        "review_scores_rating",
        "profitability"]

    maximo = {}
    for y in lista:
        maximo[y]=(max(df.groupby("neighbourhood_group_cleansed").agg('mean')[y]))

    fig = go.Figure()
    lista.append(lista[0])
    
    if barrio=="Todos":

        for x in df["neighbourhood_group_cleansed"].unique():
            data = []
            for y in lista:
                data.append((df[df["neighbourhood_group_cleansed"]==x].groupby("neighbourhood_group_cleansed").agg('mean')[y][0])/maximo[y])
                #print((df[df["neighbourhood_group_cleansed"]==x].groupby("neighbourhood_group_cleansed").agg('mean')[y][0]))
            #data.append(data[0])

            
            fig.add_trace(go.Scatterpolar(
                        r=data,
                        theta=lista,
                        mode='lines',
                        line_color=colorsBarrios[x],
                        name=x,
                        )
            )
    else:
        data = []
        for y in lista:
            data.append((df[df["neighbourhood_group_cleansed"]==barrio].groupby("neighbourhood_group_cleansed").agg('mean')[y][0])/maximo[y])
            #print((df[df["neighbourhood_group_cleansed"]==x].groupby("neighbourhood_group_cleansed").agg('mean')[y][0]))
        #data.append(data[0])

        
        fig.add_trace(go.Scatterpolar(
                    r=data,
                    theta=lista,
                    mode='lines',
                    line_color=colorsBarrios[barrio],
                    name=barrio,
                    )
        )

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True
        ),
        
    ),
    showlegend=True,
    )

    fig.update_layout(title=dict(text="<b>Valores medios por cada geografía</b><br><i>No se filtra por Barrio para poder verlos todos</i>", x=0.50,y=0.97, font=dict(size=17)),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=15), height=500,width=734)

    return fig

def graph_spider_features_normalized(df):    
    """Pinta el grafico de spider normalizado

    Args:
        df (DataFrame): DataFrame inmutable

    Returns:
        fig: figura a representar
    """
    colorsBarrios = {
            "Brooklyn": "#1f77b4",
            "Bronx": "#ff7f0e",
            "Staten Island":"#2ca02c",
            "Queens":"#9467bd",
            "Manhattan":"#d62728"
        }

    lista = ["occupancy_rate",
        "beds",
        "price",
        "baths",
        "review_scores_rating",
        "profitability"]


    fig = go.Figure()

    maximo = {}
    for y in lista:
        maximo[y]=(np.mean(df.groupby("neighbourhood_group_cleansed").agg('mean')[y]),
                    (df.groupby("neighbourhood_group_cleansed").agg('mean')[y].std()),
                    max(df.groupby("neighbourhood_group_cleansed").agg('mean')[y]))

    lista.append(lista[0])

    for x in df["neighbourhood_group_cleansed"].unique():
        data = []
        for y in lista:
            #data.append((listings_filtered_df[listings_filtered_df["neighbourhood_group_cleansed"]==x].groupby("neighbourhood_group_cleansed").agg('mean')[y][0])/maximo[y])
            data.append((df[df["neighbourhood_group_cleansed"]==x].groupby("neighbourhood_group_cleansed").agg('mean')[y][0]-maximo[y][0])/maximo[y][1])
        data.append(data[0])
        
        fig.add_trace(go.Scatterpolar(
                    r=data,
                    theta=lista,
                    mode='lines',
                    line_color=colorsBarrios[x],
                    name=x,
                    
                    )
        )

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True
        ),
        
    ),
    showlegend=True,
    )

    fig.update_layout(title=dict(text="<b>Valores medios normalizados por cada geografía</b><br><i>No se aplican filtros al gráfico<i>", x=0.50,y=0.97, font=dict(size=17)),paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=15), height=500,width=734)


    return fig

def graph_bar_amenities(df):
    """Grafico de barras con el porcentaje de ammenities

    Args:
        df (DataFrame): dataframe a representar

    Returns:
        fig: figura a representar
    """

    total_neighbourhood = df.groupby("neighbourhood_group_cleansed").agg({"id":"count"}).reset_index()
    total_neighbourhood = total_neighbourhood.rename({"id":"total"},axis=1)

    grouped_amenities_df = df.groupby("neighbourhood_group_cleansed").agg({"has_wifi":"sum", "has_dryer":"sum", "has_tv":"sum", "has_heating":"sum", "has_kitchen":"sum"})
    grouped_amenities_df = grouped_amenities_df.reset_index()

    join_df = pd.merge(grouped_amenities_df,total_neighbourhood, how="left",on="neighbourhood_group_cleansed")

    join_df["percentage_has_wifi"] = (join_df["has_wifi"]/join_df["total"]).round(2)
    join_df["percentage_has_dryer"] = (join_df["has_dryer"]/join_df["total"]).round(2)
    join_df["percentage_has_tv"] = (join_df["has_tv"]/join_df["total"]).round(2)
    join_df["percentage_has_heating"] = (join_df["has_heating"]/join_df["total"]).round(2)
    join_df["percentage_has_kitchen"] = (join_df["has_kitchen"]/join_df["total"]).round(2)

    newnames = {'percentage_has_wifi':'Tiene wifi', 'percentage_has_dryer': 'Tiene secador', 'percentage_has_tv': 'Tiene TV', 'percentage_has_heating': 'Tiene calefacción', 'percentage_has_kitchen': 'Tiene cocina'}
    join_df = join_df.rename(newnames,axis=1)
    join_df

    fig = go.Figure()
    fig = px.bar(join_df, x="neighbourhood_group_cleansed", y=["Tiene wifi","Tiene secador", "Tiene TV", "Tiene calefacción", "Tiene cocina"], barmode='group', color_discrete_sequence=px.colors.diverging.Temps)
    fig.update_layout(title = dict(text="<b>Porcentaje de Aribnbs por barrios que presentan los servicios más demandados</b><br><i>No se aplican filtros a este gráfico<i>", font=dict(size=17), x=0.5,y=0.95),
                        xaxis_title ="",yaxis_title = "Porcentaje de Airbnbs (%)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=15), height=500,width=2150,

                        legend= dict(title=""))
    return fig 

def predictPrice(barrio,lat,lon,acco,bed,bath,wifi,kitchen,dryer,heating,tv):
    """Predice el precio en función de las cosas metidas por el usuario

    Args:
        barrio (str): barrio del supuesto AirBnB
        lat (float): latitud del supuesto AirBnB
        lon (float): longitud del supuesto AirBnB
        acco (int): Numero de acomodados en en supuesto AirBnB
        bed (int): Numero de camas en el supuesto AirBnB
        bath (int): Numero de baños en el supuesto AirBnB
        wifi (int): Tiene wifi el supuesto AirBnB?
        kitchen (int): Tiene cocina el supuesto AirBnB?
        dryer (int): Tiene secador el supuesto AirBnB?
        heating (int): Tiene calentador el supuesto AirBnB?
        tv (int): Tiene televisión el supuesto AirBnB?

    Returns:
        float: precio predicho
    """
    

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

    neighbourhood_group_cleansed_Manhattan = 0
    neighbourhood_group_cleansed_Brooklyn = 0
    neighbourhood_group_cleansed_Bronx = 0
    neighbourhood_group_cleansed_Queens = 0
    neighbourhood_group_cleansed_Staten_Island = 0

    if str(barrio).lower() == "manhattan":
        neighbourhood_group_cleansed_Manhattan = 1
        neighbourhood_group_cleansed_Brooklyn = 0
        neighbourhood_group_cleansed_Bronx = 0
        neighbourhood_group_cleansed_Queens = 0
        neighbourhood_group_cleansed_Staten_Island = 0
    elif str(barrio).lower() == "brooklyn":
        neighbourhood_group_cleansed_Manhattan = 0
        neighbourhood_group_cleansed_Brooklyn = 1
        neighbourhood_group_cleansed_Bronx = 0
        neighbourhood_group_cleansed_Queens = 0
        neighbourhood_group_cleansed_Staten_Island = 0
    elif str(barrio).lower() == "bronx":
        neighbourhood_group_cleansed_Manhattan = 0
        neighbourhood_group_cleansed_Brooklyn = 0
        neighbourhood_group_cleansed_Bronx = 1
        neighbourhood_group_cleansed_Queens = 0
        neighbourhood_group_cleansed_Staten_Island = 0
    elif str(barrio).lower() == "queens": 
        neighbourhood_group_cleansed_Manhattan = 0
        neighbourhood_group_cleansed_Brooklyn = 0
        neighbourhood_group_cleansed_Bronx = 0
        neighbourhood_group_cleansed_Queens = 1
        neighbourhood_group_cleansed_Staten_Island = 0
    elif str(barrio).lower() == "staten island": 
        neighbourhood_group_cleansed_Manhattan = 0
        neighbourhood_group_cleansed_Brooklyn = 0
        neighbourhood_group_cleansed_Bronx = 0
        neighbourhood_group_cleansed_Queens = 0
        neighbourhood_group_cleansed_Staten_Island = 1
    else:
        neighbourhood_group_cleansed_Manhattan = 1
        neighbourhood_group_cleansed_Brooklyn = 0
        neighbourhood_group_cleansed_Bronx = 0
        neighbourhood_group_cleansed_Queens = 0
        neighbourhood_group_cleansed_Staten_Island = 0

    
    data = [latitude[0], longitude[0], accommodates[0], beds[0], baths[0],
                                        has_wifi[0], has_dryer[0], has_heating[0], has_kitchen[0], has_tv[0],
                                        neighbourhood_group_cleansed_Bronx,
                                        neighbourhood_group_cleansed_Brooklyn,
                                        neighbourhood_group_cleansed_Manhattan,
                                        neighbourhood_group_cleansed_Queens,
                                        neighbourhood_group_cleansed_Staten_Island]
    columns = ['latitude', 'longitude', 'accommodates', 'beds', 'baths',
                                    'has_wifi', 'has_dryer', 'has_heating', 'has_kitchen', 'has_tv',
                                    'neighbourhood_group_cleansed_Bronx',
                                    'neighbourhood_group_cleansed_Brooklyn',
                                    'neighbourhood_group_cleansed_Manhattan',
                                    'neighbourhood_group_cleansed_Queens',
                                    'neighbourhood_group_cleansed_Staten Island']

    data_model = pd.DataFrame(data=[data], columns = columns)
    prediction = rf_model.predict(data_model)
    return prediction

def getLatLong(address,barrio):
    """Devuelve la latitud y longitud del supuesto AirBnB

    Args:
        address (str): dirección del supuesto AirBnB
        barrio (str): barrio del supuesto AirBnB

    Returns:
        (float,float): latitud y longitud
    """
    print("Buscando Lat Lon")
    locator =  geopy.geocoders.Nominatim(user_agent="my_geocoder")
    # #####location = locator.geocode("122 E 19th St,Manhattan,New York,USA")
    location=locator.geocode(str(address)+","+str(barrio)+",New York,USA")
    lat= location[1][0]
    lon= location[1][1]

    print(lat,lon)
    return [lat,lon]
        
def pintarDireccionMetida(lat,lon,direccion):
    """Pinta el minimapa de la dirección metida

    Args:
        lat (float): latitud
        lon (float): longitud
        direccion (str): dirección del supuesto airbnb

    Returns:
        fig: figura a representar
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        text=[direccion],
        marker=go.scattermapbox.Marker(
            size=20
        ),
        marker_color='tomato',
        mode='text+markers',
        showlegend=False
    ))
    fig.update_layout(mapbox=dict(style='carto-positron',center={"lat": lat, "lon": lon},zoom=14))
    fig.update_layout(height=500,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',margin=dict(t=0,b=0,l=0,r=0))
    return fig

def pintarPlotAlgunosListings():
    """Pinta la evolución del precio de algunos airbnbs ya seleccionados

    Returns:
        fig: figura a representar
    """
    
    listaCambiantes = [16595,16821,13808,14290,23135,29683,2595,39282,23686,53469,57297,53470,59121]
    
    
    fig = px.line(data_frame=ListingDiffSemanaFinde[(ListingDiffSemanaFinde["listing_id"].isin(listaCambiantes))], x='date', y='priceNum', color='listing_id')
    fig.update_layout(height=500,width=2200,paper_bgcolor='rgba(0,0,0,0)', legend= dict(title="ID Airbnb"), xaxis_title = "", yaxis_title="Precio por noche ($)", title = dict(text="<b>Demostración de patrones de evolución del precio de algunos Airbnbs<b>", x=0.5,y=0.95,font=dict(size=17,color='white')), plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=15))    

    return fig

def pintarIncrementoPrecio():
    """Pinta el incremento de precio medio

    Returns:
        fig: figura a representar
    """
    
    fig = make_subplots(
        rows = 1,
        cols = 2,
        specs=[[{'type':'mapbox'}, {'type':'mapbox'}]], # Necesario para agregar un piechart
        subplot_titles = (
            "<b>Incremento de precio el fin de semana (%) por distrito<b>",
            "<b>Incremento de precio el fin de semana (%) por barrio<b>"
        )
    )

    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=jsonGeoNeigh,
                            featureidkey='properties.neighbourhood',
                            locations=ListingDiffSemanaFindePlotear.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            z=(ListingDiffSemanaFindePlotear.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['Percentage Diff Weekend Increase']),
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=0.46,title="%", tickformat='1%', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white')),
                            text=ListingDiffSemanaFindePlotear.groupby("neighbourhood_cleansed",as_index=False).agg("mean")['neighbourhood_cleansed'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            "Avg. rate: %{z:.1%}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=1    
    )
    fig.add_trace(trace=go.Choroplethmapbox(
                            geojson=bigJSONNeigh,
                            featureidkey='properties.neighbourhood_group', 
                            locations=ListingDiffSemanaFindePlotear.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            z=(ListingDiffSemanaFindePlotear.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['Percentage Diff Weekend Increase']),
                            colorscale=px.colors.sequential.YlGnBu,
                            colorbar=dict(thickness=20, x=1.02,title="%", tickformat='1%', tickcolor='white', tickfont=dict(size=20, color='white'),titlefont=dict(color='white')),
                            text=ListingDiffSemanaFindePlotear.groupby("neighbourhood_group_cleansed",as_index=False).agg("mean")['neighbourhood_group_cleansed'],
                            hovertemplate = "<b>%{text}</b><br>" +
                                            "Avg. rate: %{z:.1%}<br>" +
                                            "<extra></extra>"
                        ),
                row=1,
                col=2           
    )
    fig.update_layout(mapbox1=dict(zoom=opcionesGlobales["Centros"]["Todos"][2],style='carto-positron',center={"lat": opcionesGlobales["Centros"]["Todos"][0], "lon": opcionesGlobales["Centros"]["Todos"][1]}),
                    mapbox2=dict(zoom=opcionesGlobales["Centros"]["Todos"][2],style='carto-positron',center={"lat": opcionesGlobales["Centros"]["Todos"][0], "lon": opcionesGlobales["Centros"]["Todos"][1]}))

    fig.update_layout(height=1000,width=2200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font = dict(color = 'white', size=20))    

    
    
    return fig



#################################################################################################################################################################################################
####################################################################################### TAB CONTENTS#############################################################################################
#################################################################################################################################################################################################

tab_profitability_content = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id="subplot-profitability",style={'width': '100%', 'height': '100%'})
    ]),
)

tab_prices_content = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id="histogram-prices",style={'width': '100%', 'height': '100%'}),
        dcc.Graph(id="subplot-prices",style={'width': '100%', 'height': '100%'})        
    ]),
)

tab_occupancy_rate_content = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id="subplot-occupancy-rate",style={'width': '100%', 'height': '100%'})
    ]),
)

tab_criminality_content = dbc.Card(
    dbc.CardBody(
        [
            dcc.Graph(figure=graph_subplot_criminality(criminality_df), id="subplot-criminality",style={'width': '100%', 'height': '100%'})
        ]
    ),
)

tab_descriptive_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row([
                dbc.Col(
                    [
                        dcc.Graph(id="pie-property-type",style={'width': '100%', 'height': '90%'}),
                    ],
                    width=4,
                    style={"height": "100%"},
                ),

                dbc.Col(
                [
                    dcc.Graph(id="spider-features",style={'width': '100%', 'height': '100%'})
                ],
                width=4,
                style={"height": "100%"},),

                dbc.Col(
                [
                    dcc.Graph(id="spider-features-normalized",figure =graph_spider_features_normalized(listings_filtered_df) ,style={'width': '100%', 'height': '100%'})
                    
                ],
                width=4,
                style={"height": "100%"},),

            ], justify="center",style={"height": "50%"}),  

            dbc.Row([
                dbc.Col(
                [
                    dcc.Graph(figure=graph_bar_amenities(listings_filtered_df), id="bar-amenities",style={'width': '100%', 'height': '100%'})
                ],
                width=11,
                style={"height": "100%"},),

            ], justify="center",style={"height": "50%"})        
        ]

    ),
    className="mt-3",
)

tab_hosts_content = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id="bar-hosts-time-overall",style={'width': '100%', 'height': '100%'}),
        dcc.Graph(id="table-hosts", figure=graph_table_hosts(hosts_df), style={'width': '100%', 'height': '100%'})        
    ]),
)

tab_model_prediction_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row([
                html.H4("Rellene los siguientes datos para predecir el precio por noche del Airbnb que desee",id="instructions-form", 
                            style={
                                "text-align": "center",
                                "color":"lightgrey",
                            },
                        ),

                    html.Hr()
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label("Dirección", width=10, html_for="input-address", style={"fontSize":"150%", "text-align": "center", "color":"lightgrey"}),
                        dbc.Input(
                            id="input-address", placeholder="Introduzca la dirección del Airbnb"
                        )],
                        #width=5,
                        style={
                            "display":"inline-block",
                            "height":"100%"
                        }
                    ),
              
                    dbc.Col([
                        dbc.Label("Barrio", width=10, html_for="input-barrio", style={"fontSize":"150%","text-align": "center","color":"lightgrey"}),
                        dcc.Dropdown(id="input-barrio",
                                    options=[
                                        "Manhattan",
                                        "Brooklyn",
                                        "Queens",
                                        "Staten Island",
                                        "Bronx"
                                    ],
                                    placeholder="Introduzca el barrio",
                                    style={
                                        "display":"inline-block",
                                        "width":"80%",
                                        "height":"50%",
                                        "align":"center"
                                    }
                        )],
                        width=4,
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

                        dbc.Label("Nº de camas", width=10, html_for="input-beds", style={"fontSize":"150%","text-align": "center","color":"lightgrey"}),
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
                children=
                [

                ],
               justify="center",
               id="row-price"
            ),
        ]
    ),
)

tab_bonus_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="Calendar-Algunos-AirBnBs",style={'width': '100%', 'height': '100%'},figure=pintarPlotAlgunosListings()),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="Calendar-weekendIncrease",style={'width': '100%', 'height': '100%'},figure=pintarIncrementoPrecio()),
                ])
            ])
        ]
    ),
    className="mt-3",
)

#################################################################################################################################################################################################
####################################################################################### DASH APP ################################################################################################
#################################################################################################################################################################################################

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE],suppress_callback_exceptions=True)

logging.getLogger('werkzeug').setLevel(logging.INFO)
dash.register_page(__name__, path='/')
app.title="AirBnB Dash"
#app._favicon = (str(os.getcwd())+'\Images\AirBnB\airnbnb.svg')


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
        dcc.Store(id="store-nclicks", storage_type='session'), # para guardar informacion. Es una variable global para el numero de clicks
        
        dbc.Row([
            dbc.Col([
                dcc.Store(id="n_clicks_button_pred"),
                html.Img(src=returnImage('\Images\AirBnB\logoBlanco.png'),
                    style={
                        #"display":"inline-block",
                        "width":"25%",
                        "vertical-align": "center",
                        "padding-right":"1%"
                    }
                )],
                width=3
            ),
            dbc.Col([
                html.H1("Estudio sobre los AirBnBs en NYC",
                    style={
                        #"display":"inline-block",
                        "vertical-align": "bottom",
                        "horizontal-align": "right",
                        "textAlign": "right",
                        "color": "white",
                        "fontSize":"60"
                    }
                ),
                ],
                width=12,
            ) 
            ],

            id = "Titulo",
            style ={
                "align": "right",
                #'display':'inline-block',
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
                                dbc.Tab(label="Precios", tab_id="prices"),
                                dbc.Tab(label="Occupancy rates", tab_id="occupancy-rate"),
                                dbc.Tab(label="Descriptivo", tab_id="descriptive"),
                                dbc.Tab(label="Hosts", tab_id="hosts"),
                                dbc.Tab(label="Criminalidad", tab_id="criminality"),
                                dbc.Tab(label="Predicción de precios", tab_id="model_prediction"),                               
                                dbc.Tab(label="Variación de precios", tab_id="bonus")     
                            ],
                            id="tabs",
                            active_tab="profitability",

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
####################################################################################### CALLBACKS ###############################################################################################
#################################################################################################################################################################################################

# Callback para cambiar de tab
@app.callback(
    Output("tabs-content", "children"), 
    Input("tabs", "active_tab"))
def switch_tab(tab):
    """Cambia de pestaña

    Args:
        tab (str): id de la pestaña

    Returns:
        children: contenido a representar
    """
    if tab == "profitability":
        return tab_profitability_content
    elif tab == "prices":
        return tab_prices_content
    elif tab == "occupancy-rate":
        return tab_occupancy_rate_content
    elif tab == "criminality":
        return tab_criminality_content
    elif tab == "descriptive":
        return tab_descriptive_content
    elif tab == "hosts":
        return tab_hosts_content
    elif tab == "model_prediction":
        return tab_model_prediction_content
    elif tab == "bonus":
        return tab_bonus_content
    else:
        return html.P("This shouldn't ever be displayed...")

# callback para actualizar subplot rentabilidad
@app.callback(
    Output('subplot-profitability', 'figure'),
    Input('range-slider-rentabilidad', 'value'),
    Input('barrios-seleccion', 'value'),
    Input('range-slider-precio', 'value'),
    Input('switches-input', 'value'),
)
def update_subplot_rentabilidad(rentabilidad,barrio,precio,checkFiltros):
    """
    Args:
        rentabilidad (array float): rentabilidad minima y maxima
        barrio (str): Barrio o Todo
        precio (array float): precio maximo y minimo
        checkFiltros (int): filtros activos?
    
    Return:
        grpah_updated (figure): gráfico actualizado

    """
    
    if checkFiltros:
        #filtramos el df
        df_filtered = filtrarDF(rentabilidad[0],rentabilidad[1],barrio,precio[0],precio[1])

        return graph_subplot_rentabilidad(df_filtered,barrio)
    
    else:
        return graph_subplot_rentabilidad(listings_filtered_df,"Todos")

# callback para actualizar subplot prices
@app.callback(
    Output('subplot-prices', 'figure'),
    Output('histogram-prices', 'figure'),
    Input('range-slider-rentabilidad', 'value'),
    Input('barrios-seleccion', 'value'),
    Input('range-slider-precio', 'value'),
    Input('switches-input', 'value'),
)
def update_plots_prices(rentabilidad,barrio,precio,checkFiltros):
    """
    Args:
        rentabilidad (array float): rentabilidad minima y maxima
        barrio (str): Barrio o Todo
        precio (array float): precio maximo y minimo
        checkFiltros (int): filtros activos?
    
    Return:
        grpah_updated (figure): gráfico actualizado

    """
    if checkFiltros:
        #filtramos el df
        df_filtered = filtrarDF(rentabilidad[0],rentabilidad[1],barrio,precio[0],precio[1])

        return [graph_subplot_prices(df_filtered,barrio),graph_histogram_prices(df_filtered)]
    
    else:
        return [graph_subplot_prices(listings_filtered_df,"Todos"),graph_histogram_prices(listings_filtered_df)]

# callback para actualizar subplot prices
@app.callback(
    Output('subplot-occupancy-rate', 'figure'),
    Input('range-slider-rentabilidad', 'value'),
    Input('barrios-seleccion', 'value'),
    Input('range-slider-precio', 'value'),
    Input('switches-input', 'value'),
)
def update_subplot_occupancy_rate(rentabilidad,barrio,precio,checkFiltros):
    """
    Args:
        rentabilidad (array float): rentabilidad minima y maxima
        barrio (str): Barrio o Todo
        precio (array float): precio maximo y minimo
        checkFiltros (int): filtros activos?
    
    Return:
        grpah_updated (figure): gráfico actualizado

    """
    if checkFiltros:
        #filtramos el df
        df_filtered = filtrarDF(rentabilidad[0],rentabilidad[1],barrio,precio[0],precio[1])

        return graph_subplot_occupancy_rates(df_filtered,barrio)
    
    else:
        return graph_subplot_occupancy_rates(listings_filtered_df,"Todos")

# callback para actualizar subplot hosts
@app.callback(
    Output('bar-hosts-time-overall', 'figure'),
    Input('range-slider-rentabilidad', 'value'),
    Input('barrios-seleccion', 'value'),
    Input('range-slider-precio', 'value'),
    Input('switches-input', 'value'),
)
def update_bar_hosts_time_overall(rentabilidad,barrio,precio,checkFiltros):
    """
    Args:
        rentabilidad (array float): rentabilidad minima y maxima
        barrio (str): Barrio o Todo
        precio (array float): precio maximo y minimo
        checkFiltros (int): filtros activos?
    
    Return:
        grpah_updated (figure): gráfico actualizado

    """
    if checkFiltros:
        #filtramos el df
        df_filtered = filtrarDF(rentabilidad[0],rentabilidad[1],barrio,precio[0],precio[1])

        return graph_bar_hosts_time_overall(df_filtered)
    
    else:
        return graph_bar_hosts_time_overall(listings_filtered_df)
    

@app.callback(
    Output('pie-property-type', 'figure'),
    Input('range-slider-rentabilidad', 'value'),
    Input('barrios-seleccion', 'value'),
    Input('range-slider-precio', 'value'),
    Input('switches-input', 'value'),
)
def update_pie_chart(rentabilidad,barrio,precio,checkFiltros):
    """
    Args:
        rentabilidad (array float): rentabilidad minima y maxima
        barrio (str): Barrio o Todo
        precio (array float): precio maximo y minimo
        checkFiltros (int): filtros activos?
    
    Return:
        grpah_updated (figure): gráfico actualizado

    """
    if checkFiltros:
        #filtramos el df
        df_filtered = filtrarDF(rentabilidad[0],rentabilidad[1],barrio,precio[0],precio[1])

        return graph_pie_property_type(df_filtered)
    
    else:
        return graph_pie_property_type(listings_filtered_df)

# callback para actualizar spider features
@app.callback(
    Output('spider-features', 'figure'),
    Input('range-slider-rentabilidad', 'value'),
    Input('barrios-seleccion', 'value'),
    Input('range-slider-precio', 'value'),
    Input('switches-input', 'value'),
)
def update_spider_feature(rentabilidad,barrio,precio,checkFiltros):
    """
    Args:
        rentabilidad (array float): rentabilidad minima y maxima
        barrio (str): Barrio o Todo
        precio (array float): precio maximo y minimo
        checkFiltros (int): filtros activos?
    
    Return:
        grpah_updated (figure): gráfico actualizado

    """
    if checkFiltros:
        #filtramos el df
        df_filtered = filtrarDF(rentabilidad[0],rentabilidad[1],"Todos",precio[0],precio[1])

        return graph_spider_features(df_filtered,barrio)
    
    else:
        return graph_spider_features(listings_filtered_df,"Todos")

# callback prediccion precio
@app.callback(
    Output('row-price', 'children'),
    Output('button-predict', 'n_clicks'),
    Input('input-address', 'value'),
    Input('input-barrio', 'value'),
    Input('input-accommodates', 'value'),
    Input('input-beds', 'value'),
    Input('input-baths', 'value'),
    Input('amenities-input', 'value'),
    Input("button-predict", "n_clicks"),
)
def update_predicted_price(direccion,barrio,accommodates,beds,baths,amenities,button):
    """
    Args:
        direccion (str): dirección del supuesto AirBnB
        barrio (str): barrio del supuesto AirBnB
        accommodates (float): numero de acomodados del supuesto AirBnB
        beds (float): numero de camas del supuesto AirBnB
        baths (float): numero de baños del supuesto AirBnb
        amenities (arr str): array de ammenities del supuesto AirBnB
        button (int): predecir?
    
    Return:
        children: precio + mapa

    """

    wifi = 0
    kitchen = 0
    heating = 0
    tv = 0
    dryer = 0

    #print("nclicks")
    print(button)
    
    #if (nclicks is not None) & ((button > int(nclicks)) | (nclicks ==1)):
    if (button > 0) & (barrio in opcionesGlobales["Barrios"]):

        if "wifi" in str(amenities):
            wifi = 1
        else: 
            wifi = 0

        if "dryer" in str(amenities):
            dryer = 1
        else:
            dryer = 0
        
        if "heating" in str(amenities):
            heating = 1
        else:
            heating = 0
        
        if "tv" in str(amenities):
            tv = 1
        else:
            tv = 0
        
        if "kitchen" in str(amenities):
            kitchen = 1
        else:
            kitchen = 0

        #obtenemos latitud y longitud
        latitude, longitude = getLatLong(direccion,barrio)
        
        precio_pred_arr = predictPrice(barrio,latitude,longitude,accommodates,beds,baths,wifi,kitchen,dryer,heating,tv)
        print("trasPrediccion")
        
        precio_pred = round(precio_pred_arr[0],2)
        str_precio = "$" + str(precio_pred)
        
        
        
        
        
        return  [[dbc.Label(str_precio, id="predicted-price",
                        style={
                            "fontSize":"250%",
                            "text-align": "center",
                            "color":"white",
                        },
                    ),
                 dcc.Graph(figure=pintarDireccionMetida(lat=latitude,lon= longitude,direccion=direccion), 
                           id="mapa-direccion",
                           style={'width': '100%', 'height': '100%'})]
                 ,0]
    
    else:
        return [dbc.Label(" ", id="predicted-price",
                        style={
                            "fontSize":"250%",
                            "text-align": "center",
                            "color":"white",
                        },
                    ),0]



if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
    app.config.suppress_callback_exceptions=True
    app.title = "AirBnB Dash"
    
    
    #variables globales
    n_clicks_button_pred = 0


