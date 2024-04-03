import requests
import multiprocessing as mp
from datetime import datetime
from herbie import Herbie
from time import sleep
import random

SERVER_URL = "http://52.34.147.76:8000"

DATE_FORMAT = r"%Y-%m-%d"
TIME_FORMAT = r"%H:%M:%S"
WAIT_TIME = 600

# earlier than this => use ECMWF
HRRR_CUTOFF_DATE = datetime.strptime("2014-07-30", DATE_FORMAT)

HRRR_SRC = "hrrr"
ECMWF_SRC = "ifs"


def worker(supplier, index):
    while True:
        try:
            row_json = requests.get(SERVER_URL).json()
        except:
            print(f"Server refused connection. Terminating worker {index}...")
            return

        idx = row_json["index"]

        # needs to be before source
        date = datetime.strptime(row_json["date"], DATE_FORMAT)

        source = HRRR_SRC if date >= HRRR_CUTOFF_DATE else ECMWF_SRC

        time = datetime.strptime(row_json["time"], TIME_FORMAT)

        combined = datetime.combine(date.date(), time.time())

        print("supplier:", supplier, "datetime:", combined, "source:", source)
        # H = Herbie(
        #     str(combined),
        #     model=source,
        #     product="sfc",
        #     fxx=[0],
        # )

        # parse herbie response
        # convert to weather API params

        send_resp = requests.post(
            SERVER_URL,
            json={
                "index": idx,
                "main_temp": 2134,
                "main_feels_like": 123,
                "main_pressure": 23,
                "main_humidity": 234,
                "main_dew_point": 25,
                "wind_speed": 903,
                "wind_deg": 3490,
                "wind_gust": 324,
                "clouds_all": 0,
                "rain_3h": 235,
                "snow_3h": 345,
                "visibility": 0,
            },
        )
        if send_resp.status_code != 200:
            print("Rate limit reached. Sleeping...")
            sleep(WAIT_TIME)
            print("Starting again.")


def main():
    suppliers = ["aws"]
    for i, supplier in enumerate(suppliers):
        mp.Process(target=worker, args=(supplier, i)).start()


if __name__ == "__main__":
    main()
