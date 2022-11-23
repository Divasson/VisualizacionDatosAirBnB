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
                       "Manhattan"]
        }
    return z

if __name__=="__main__":
    for p in sys.path:
        print(p)
    