import json
import uvicorn 
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

def get_test_data():
    
    neighbourhoods_json_str = '["IDOTRR","SawyerW","Timber","NAmes","NoRidge","Blmngtn","Somerst","BrkSide","OldTown","ClearCr","CollgCr","NridgHt","Mitchel","Crawfor","Sawyer","Edwards","Gilbert","StoneBr","NWAmes","SWISU","Blueste","Veenker","MeadowV","NPkVill","BrDale","GrnHill","Greens","Landmrk"]'
    
    json_str = '{"LotFrontage":57.0,"LotArea": 8923,"GrLivArea": 1382,"FullBath":2 ,"HalfBath": 1,"YrSold": 2009, "Neighbourhood": "NridgHt"}'

    jd = json.loads(json_str)

    jdf = pd.DataFrame([jd])

    jdf
