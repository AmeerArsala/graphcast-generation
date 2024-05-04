import multiprocessing as mp
import pandas as pd
from herbie import Herbie
from datetime import datetime
import xarray
import os

NUM_PROCESSES = 1

INFILE_NAME = "WeatherV4.csv"
OUTFILE_NAME = "WeatherV5.csv"
DELIMITER = "@"

HRRR_cutoff_date = datetime.datetime(2014) # TODO: fill this in

WEATHER_ATTRIBUTES = [
    "temp"
    "feels_like"
    "pressure"
    "humidity"
    "dew_point"
    "wind_speed"
    "wind_deg"
    "wind_gust"
    "rain_3h"
    "snow_3h"
    "clouds_all"
    "visibility"
]


def parse_herbie(H: Herbie):
    # TODO: fill this in with the parsing logic
    """
    ==================================== ==========================================================
    ``searchString=``                    GRIB messages that will be downloaded
    ==================================== ==========================================================
    ":TMP:2 m"                           Temperature at 2 m.
    ":TMP:"                              Temperature fields at all levels.
    ":UGRD:\d+ mb"                       U Wind at all pressure levels.
    ":500 mb:"                           All variables on the 500 mb level.
    ":APCP:"                             All accumulated precipitation fields.
    ":APCP:surface:0-[1-9]*"             Accumulated precip since initialization time
    ":APCP:.*:(?:0-1|[1-9]\d*-\d+) hour" Accumulated precip over last hour
    ":UGRD:10 m"                         U wind component at 10 meters.
    ":[U|V]GRD:[1,8]0 m"                 U and V wind component at 10 and 80 m.
    ":[U|V]GRD:"                         U and V wind component at all levels.
    ":.GRD:"                             (Same as above)
    ":[U|V]GRD:\d+ hybrid"               U and V wind components at all hybrid levels
    ":[U|V]GRD:\d+ mb"                   U and V wind components at all pressure levels
    ":(?:TMP|DPT):"                      Temperature and Dew Point for all levels .
    ":(?:TMP|DPT|RH):"                   TMP, DPT, and Relative Humidity for all levels.
    ":REFC:"                             Composite Reflectivity
    ":surface:"                          All variables at the surface.
    "^TMP:2 m.*fcst$"                    Beginning of string (^), end of string ($) wildcard (.*)
    ==================================== ==========================================================
    "start_date"
    "start_time"
    "duration"
    "temp"
    "feels_like"
    "pressure"
    "humidity"
    "dew_point"
    "wind_speed"
    "wind_deg"
    "wind_gust"
    "rain_3h"
    "snow_3h"
    "clouds_all"
    "visibility"
    """
    vgrd = H.xarray(":VGRD:\d+ mb")
    temp_dpt_rh = H.xarray(":(?:TMP|DPT|RH):")
    gust = H.xarray(":GUST:")
    humidity = temp_dpt_rh[1].r
    temp = temp_dpt_rh[2].t
    dew_point = temp_dpt_rh[2].dpt
    start_date, start_time = str(dpt[2].time.data).split("T")
    time = time.split(".")[0]
    wind_gust = gust.gust

    longitude = vgrd.longitude.size
    latitude = vgrd.longitude.size
    # TODO: return extracted weather values
    # see this https://herbie.readthedocs.io/en/stable/user_guide/tutorial/search.html
    # and this https://herbie.readthedocs.io/en/stable/user_guide/background/zarr.html
    return {}


def worker(idx_counter: mp.Value, resource_lock: mp.Lock, df: pd.DataFrame, id: int):
    while True:
        with resource_lock:
            idx = idx_counter.value
            idx_counter.value += 1

        if idx >= len(df):
            break

        row = df.iloc[idx]
        weather_date = datetime.strptime(row["datetime"], "%y-%m-%d %H:%M:%S")

        HRRR_cutoff_date = datetime()  # TODO: fill this in

        H = Herbie(
            date=weather_date,  # TODO: need to change source depending on dataset cutoff date
        )
        result = parse_herbie(H)
        # TODO: write the weather values to the df

    print(f"Terminating worker {id}.")


def main():
    print(f"Reading {INFILE_NAME}...")
    df = pd.read_csv(INFILE_NAME, sep="@")
    print("Done.")
    df["datetime"] = df["SampledDate"] + " " + df["SampledTime"]

    # TODO: add the weather attributes as columns to the df

    idx_counter = mp.Value("i", 0)
    lock = mp.Lock()

    print("Starting processes...")
    jobs = []
    for i in range(NUM_PROCESSES):
        jobs.append(mp.Process(target=worker, args=(idx_counter, lock, df, i)))
        jobs[i].start()
    print("Done.")

    for job in jobs:
        job.join()  # block main process until all workers are done
    print("All workers terminated.")

    print(f"Writing to {OUTFILE_NAME}...")
    df.to_csv(OUTFILE_NAME, sep=DELIMITER)
    print("Done.")


if __name__ == "__main__":
    main()
