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
MS_HRRR_CUTOFF = datetime.strptime("2021-03-21", DATE_FORMAT)
AWSGOOGLE_HRRR_CUTOFF = datetime.strptime("2014-07-30", DATE_FORMAT)

HRRR_SRC = "hrrr"
ECMWF_SRC = "ifs"


def parse_herbie(H: Herbie):
    # TODO: fill this in with the parsing logic
    return {}


def client():
    while True:
        try:
            row_json = requests.get(SERVER_URL).json()
        except:
            print(f"Server refused connection. Terminating...")
            return

        idx = row_json["index"]

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
