# Importamos las librerias mínimas necesarias
import pandas as pd
import numpy as np
import json

def leerFicheroFinal():
    """_summary_

    Returns:
        _type_: _description_
    """  
    z = pd.read_parquet("../../Data/total data/modified data/listingsFinalConRentabilidad.parquet")
    return z

def leerFicherosGeo():
    """_summary_

    Returns:
        _type_: _description_
    """  
    jsonGeoNeigh = json.load(open("../../Data/neighbourhoods/neighbourhoods.geojson"))
    bigJSONNeigh = json.load(open("../../Data/neighbourhoods/bigneighbourhoods.geojson"))
    return (jsonGeoNeigh,bigJSONNeigh)


def opcionesGlobales():
    z = {
            "Barrios":{
                "Brooklyn": "#1f77b4",
                "Bronx": "#ff7f0e",
                "Staten Island":"#2ca02c",
                "Queens":"#9467bd",
                "Manhattan":"#d62728"
            },
            "HabitacionesTipo":["Entire rental unit", "Private room in rental unit", "Private room in home", "Entire condo", "Entire home"]
        }
    return z

if __name__=="__main__":
    pass
    