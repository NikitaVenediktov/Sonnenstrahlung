from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field

import regressor as regr
from predict import humanize_time


class Input(BaseModel):
    temperature: float
    time: datetime = Field(default_factory=datetime.now)


class Response(BaseModel):
    prediction: float


app = FastAPI(
    title="Solar ML API", description="API for solar prediction ml model", version="1.0"
)


@app.on_event("startup")
async def load_model():
    filename = r"data/model.sav"
    regr.model = load(open(filename, "rb"))


@app.get("/")
async def root():
    return {"just": "do it"}


@app.get("/info")
async def info():
    return {"info": "You can make a prediction!"}


@app.post(
    "/predict",
    tags=["Predictions"],
    response_model=Response,
    description="Get prediction from model",
)
async def get_prediction(new_data: Input):
    data = dict(new_data)
    today = humanize_time(data["time"])
    temperature = data["temperature"]
    prediction = regr.model.predict(
        pd.DataFrame(
            [[temperature, today["DayOfYear"], today["TimeOfDay(s)"]]],
            columns=["Temperature", "DayOfYear", "TimeOfDay(s)"],
        )
    )

    return {"prediction": prediction}
