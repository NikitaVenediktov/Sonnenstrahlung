"""Zaglushka plya docstring"""

from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field

import regressor as regr


def humanize_time(time: datetime) -> dict:
    """Add time stamps to datset."""
    res = {}
    res["MonthOfYear"] = int(time.strftime("%m"))
    res["DayOfYear"] = int(time.strftime("%j"))
    res["WeekOfYear"] = int(time.strftime("%U"))
    res["TimeOfDay(h)"] = time.hour
    res["TimeOfDay(m)"] = time.hour * 60 + time.minute
    res["TimeOfDay(s)"] = time.hour * 60 * 60 + time.minute * 60 + time.second
    return res


class Input(BaseModel):
    """

    Zaglushka plya docstring

    """

    temperature: float
    time: datetime = Field(default_factory=datetime.now)


class Response(BaseModel):
    """Zaglushka plya docstring"""

    prediction: float


app = FastAPI(
    title="Solar ML API", description="API for solar prediction ml model", version="1.0"
)


@app.on_event("startup")
async def load_model():
    """Zaglushka plya docstring"""

    filename = r"data/model.sav"
    regr.model = load(open(filename, "rb"))


@app.get("/")
async def root():
    """Zaglushka plya docstring"""

    return {"just": "do it"}


@app.get("/info")
async def info():
    """Zaglushka plya docstring"""

    return {"info": "You can make a prediction!"}


@app.post(
    "/predict",
    tags=["Predictions"],
    response_model=Response,
    description="Get prediction from model",
)
async def get_prediction(new_data: Input):
    """Zaglushka plya docstring"""

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
