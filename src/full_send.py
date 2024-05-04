import pandas as pd
from herbie import Herbie, FastHerbie

NUM_THREADS = 1

start_date = "2014-01-01"
end_date = "2022-12-31"

model = "hrrr"
product = "sfc"  # surface fields

time_df = pd.read_csv("src/test_infile.csv")
time_df = time_df.apply(lambda row: f"{row['date']} {row['time']}", axis=1).to_list()

FH = FastHerbie(time_df, model=model, product=product, fxx=[0], max_threads=NUM_THREADS)

# FH.download(searchString="", max_threads=NUM_THREADS)



# 234254,"2021-12-09","21:34:49"
# 234556,"1989-02-27","09:12:34"
# 234556,"2002-08-02","18:30:00"
