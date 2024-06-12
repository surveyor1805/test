import random

import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("lr.pkl", 'rb') as file:
    model = joblib.load(file)
class ModelRequestData(BaseModel):
    total_square: float
    rooms: float
    floor: float

class Result(BaseModel):
    result: float

@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

@app.get("/predict_get", response_model=Result)
def preprocess_data():
    rand_total_square = random.randint(1, 2070)
    rand_rooms = random.randint(1, 15)
    rand_floor = random.randint(1, 66)
    data = ModelRequestData(total_square=rand_total_square, rooms=rand_rooms, floor=rand_floor)
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.post("/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)