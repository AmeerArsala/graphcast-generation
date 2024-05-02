import pandas as pd
from herbie import Herbie, FastHerbie

NUM_THREADS = 1

start_date = "2014-01-01"
end_date = "2022-12-31"

model = "hrrr"
product = "sfc"  # surface fields

time_df = pd.read_csv("test_infile.csv")
time_df = time_df.apply(lambda row: f"{row['date']} {row['time']}", axis=1).to_list()

FH = FastHerbie(time_df, model=model, product=product, fxx=[0], max_threads=NUM_THREADS)

FH.download(searchString="", max_threads=NUM_THREADS)
