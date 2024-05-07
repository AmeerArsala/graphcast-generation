import pandas as pd
from herbie import Herbie, FastHerbie


date_time: str = '2021-01-01 12:00'
model = 'hrrr'
product = 'sfc'  # surface fields

# Each hour for the last 6 hours
DATES = pd.date_range(
    start=date_time,
    periods=6,
    freq="1h"
)

print(DATES)
# Uses multithreading
FH = FastHerbie(DATES, model=model, product=product, fxx=range(0, 6))