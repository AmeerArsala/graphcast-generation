from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
import pandas as pd
import numpy as np
from pydantic import BaseModel
import signal
import os


class OutRow(BaseModel):
    index: int
    date: str
    time: str


class InRow(BaseModel):
    index: int
    main_temp: float
    main_feels_like: float
    main_pressure: float
    main_humidity: float
    main_dew_point: float
    wind_speed: float
    wind_deg: float
    wind_gust: float
    clouds_all: float
    rain_3h: float
    snow_3h: float
    visibility: float


INFILE_NAME = "/home/charles/Dev/ultimate-weather-forecast/src/test_infile.csv"
OUTFILE_NAME = "/home/charles/Dev/ultimate-weather-forecast/src/test_outfile.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, df_iter, num_received
    num_received = 0

    print("Reading input CSV...")
    df = pd.read_csv(INFILE_NAME)
    print("Done.")

    df_iter = df.iterrows()

    for param in InRow.model_fields.keys():
        if param != "index":
            df[param] = np.nan

    yield  # everything before yield run on start up, after run on shutdown

    print("Writing output CSV...")
    with open(OUTFILE_NAME, "w") as _:
        df.to_csv(OUTFILE_NAME, na_rep="NA", index=False)
    print("Done.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def get_row():
    global df_iter
    idx, row = next(df_iter, (None, None))

    if idx is None:
        return

    return OutRow(
        index=idx,
        date=row.at["date"],
        time=row.at["time"],
    )


@app.post("/")
async def post_row(new_row: InRow):
    global df, num_received

    df.loc[new_row.index, "main_temp"] = new_row.main_temp
    df.loc[new_row.index, "main_feels_like"] = new_row.main_feels_like
    df.loc[new_row.index, "main_pressure"] = new_row.main_pressure
    df.loc[new_row.index, "main_humidity"] = new_row.main_humidity
    df.loc[new_row.index, "main_dew_point"] = new_row.main_dew_point
    df.loc[new_row.index, "wind_speed"] = new_row.wind_speed
    df.loc[new_row.index, "wind_deg"] = new_row.wind_deg
    df.loc[new_row.index, "wind_gust"] = new_row.wind_gust
    df.loc[new_row.index, "clouds_all"] = new_row.clouds_all
    df.loc[new_row.index, "rain_3h"] = new_row.rain_3h
    df.loc[new_row.index, "snow_3h"] = new_row.snow_3h
    df.loc[new_row.index, "visibility"] = new_row.visibility

    num_received += 1
    if num_received == len(df):
        os.kill(os.getpid(), signal.SIGTERM)
    else:
        return Response(status_code=200)
