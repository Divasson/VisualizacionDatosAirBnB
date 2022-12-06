# Importamos las librerias mínimas necesarias
import pandas as pd
import numpy as np
import json
import sys

def leerFicheroFinal():
    """Devuelve el fichero bueno

    Returns:
        DF: DF final
    """  
    z = pd.read_parquet("Data/total data/modified data/listingsFinalConRentabilidad.parquet")
    return z

def leerFicherosGeo():
    """Devuelve una tupla de los ficheros geo

    Returns:
        (geodataframe,geodataframe): Los dos GeoDF
    """  
    jsonGeoNeigh = json.load(open("Data/neighbourhoods/neighbourhoods.geojson"))
    bigJSONNeigh = json.load(open("Data/neighbourhoods/bigneighbourhoods.geojson"))
    return (jsonGeoNeigh,bigJSONNeigh)

def leerFicherosCriminality():
    """Devuelve un fichero con info de criminalidad

    Returns:
        (geodataframe,geodataframe): Los dos GeoDF
    """  
    df_criminality = pd.read_parquet("Data/NYC/Data_with_neighbourhood.parquet")
    return df_criminality

def leerFicheroHosts():
    """Devuelve un fichero con info de hosts

    Returns:
        (geodataframe,geodataframe): Los dos GeoDF
    """  
    df_hosts = pd.read_parquet("Data/total data/modified data/hosts_df.parquet")
    return df_hosts

def opcionesGlobales():
    """Devuelve parámetros totales de la app
    
    Returns:
        dict: diccionario a través del cual se acceden a varios valores globales
    """
    z = {
            "Colores Barrios":{
                "Brooklyn": "#1f77b4",
                "Bronx": "#ff7f0e",
                "Staten Island":"#2ca02c",
                "Queens":"#9467bd",
                "Manhattan":"#d62728"
                },
            "HabitacionesTipo":["Entire rental unit", 
                                "Private room in rental unit", 
                                "Private room in home", 
                                "Entire condo", 
                                "Entire home"],
            "Opciones desplegable Vivienda":["property_type",
                                             "room_type",
                                             "beds",
                                             "baths",
                                             "review_scores_rating"],
            "Opciones desplegable Finanzas":["price",
                                             "occupancy_rate",
                                             "revenue",
                                             "price per unit",
                                             "profitability"],
            "Barrios":["Brooklyn",
                       "Bronx",
                       "Staten Island",
                       "Queens",
                       "Manhattan"],
            "Centros":{
                "Todos":[40.7,-74,9.5],
                "Brooklyn": [40.64822,-73.95232,10.5],
                "Bronx": [40.85038,-73.87173,11],
                "Staten Island":[40.58141,-74.15259,10.8],
                "Queens":[40.72,-73.82121,10],
                "Manhattan":[40.77819,-73.96796,10.5]
            }
        }
    return z

if __name__=="__main__":
    for p in sys.path:
        print(p)
    