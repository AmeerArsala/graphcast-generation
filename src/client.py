import requests
from datetime import datetime
from herbie import Herbie
from time import sleep
import random

SERVER_URL = "http://52.34.147.76:8000"

DATE_FORMAT = r"%Y-%m-%d"
TIME_FORMAT = r"%H:%M:%S"
WAIT_TIME = 600

# earlier than this => use ECMWF
MS_HRRR_CUTOFF = datetime.strptime("2021-03-21", DATE_FORMAT)
AWSGOOGLE_HRRR_CUTOFF = datetime.strptime("2014-07-30", DATE_FORMAT)

HRRR_SRC = "hrrr"
ECMWF_SRC = "ifs"


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
    start_date, start_time = str(dpt[2].time.data).split('T')
    time = time.split('.')[0]
    wind_gust = gust.gust


    longitude = vgrd.longitude.size
    latitude = vgrd.longitude.size
    return {
        
    }


def client():
    while True:
        try:
            row_json = requests.get(SERVER_URL).json()
        except:
            print(f"Server refused connection. Terminating...")
            return

        date = datetime.strptime(row_json["date"], DATE_FORMAT)
        time = datetime.strptime(row_json["time"], TIME_FORMAT)
        combined = datetime.combine(date.date(), time.time())

        aws_flag = False

        def set_provider():
            nonlocal aws_flag
            aws_flag = not aws_flag
            return "aws" if aws_flag else "google"

        while True:
            if date > MS_HRRR_CUTOFF:
                model = HRRR_SRC
                priority = "azure"
            elif date > AWSGOOGLE_HRRR_CUTOFF:
                model = HRRR_SRC
                priority = set_provider()
            else:
                model = ECMWF_SRC
                priority = set_provider()

            H = Herbie(
                str(combined),
                model=model,
                priority=priority,
                product="sfc",
                fxx=0,
            )
            if True:  # on success
                break

        while True:
            send_resp = requests.post(
                SERVER_URL,
                json=parse_herbie(H),
            )
            if send_resp.status_code == 200:
                break
            else:
                print("Server Error:", send_resp.content.decode())
                print("Sleeping...")
                sleep(random.randrange(10, 100))


if __name__ == "__main__":
    client()
