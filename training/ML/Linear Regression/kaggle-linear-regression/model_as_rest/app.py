import json
import uvicorn 
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
from pydantic import BaseModel

#https://medium.com/@alidu143/containerizing-fastapi-app-with-docker-a-comprehensive-guide-416521b2457c

app = FastAPI()
dir_path = os.path.dirname(os.path.realpath(__file__))
pickle_in = open(os.path.join(dir_path, "linear-regression-model.pkl"),"rb")
model=pickle.load(pickle_in)


class HouseFeatures(BaseModel):
    LotFrontage: float
    LotArea: int
    GrLivArea: int
    FullBath: int
    HalfBath: int
    YrSold: int
    Neighbourhood: str

def prep_model_data(data: dict):
    
    try:
        neighbourhoods_json_str = '["IDOTRR","SawyerW","Timber","NAmes","NoRidge","Blmngtn","Somerst","BrkSide","OldTown","ClearCr","CollgCr","NridgHt","Mitchel","Crawfor","Sawyer","Edwards","Gilbert","StoneBr","NWAmes","SWISU","Blueste","Veenker","MeadowV","NPkVill","BrDale","GrnHill","Greens","Landmrk"]'
        neighbour_narr = np.array(json.loads(neighbourhoods_json_str)).reshape(-1,1)

        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.set_output(transform='pandas')
        ohe.fit(neighbour_narr)

        json_str = '{"LotFrontage":57.0,"LotArea": 8923,"GrLivArea": 1382,"FullBath":2 ,"HalfBath": 1,"YrSold": 2009, "Neighbourhood": "NoRidge"}'
        jdf = pd.DataFrame([data])

        neigh_ohe = ohe.transform(np.array(jdf['Neighbourhood']).reshape(-1,1))

        jdf, neigh_ohe_tranformed_df = jdf.drop(['Neighbourhood'],axis=1), pd.DataFrame(neigh_ohe)

        train_df = pd.concat([jdf, neigh_ohe_tranformed_df])

        train_df = train_df.fillna(0)

        return train_df
    except Exception as e:
        print(f'error at prep_model_data {e}')

    


@app.post('/predict')
def predict_housing_price(data: HouseFeatures):
    
    try:
        input = prep_model_data(data.model_dump())

        predicted_price = model.predict(input)

        return predicted_price[0]
    except Exception as e:
        print(f'error at predict_housing_price {e}')

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)




