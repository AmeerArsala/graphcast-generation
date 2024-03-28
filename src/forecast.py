import jax
import jax.numpy as jnp
import pandas as pd
import xarray

# Weather
from herbie import Herbie, FastHerbie
from graphcast.data_utils import add_tisr_var, add_derived_vars


# TODO: this is the final thing that is retrieved
# try ECMWF
# if fails, try GraphCast
# date_time: GMT (Greenwich Mean Time)
def get_weather_data(date_time: str) -> xarray.Dataset:
    pass


# TODO: this is also part of the final thing that is retrieved
# try HRRR
# if fails, try to just get regular weather data (ECMWF, GraphCast)
# date_time: GMT (Greenwich Mean Time)
def get_usa_weather_data(date_time: str) -> xarray.Dataset:
    pass
