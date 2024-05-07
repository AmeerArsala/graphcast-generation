import cdsapi
cds = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/v2", key = "308352:704f911a-0b2a-427f-ac5b-c9afd4084f09", debug=True)
cds.retrieve('reanalysis-era5-pressure-levels', {
           "variable": "temperature",
           "pressure_level": "1000",
           "product_type": "reanalysis",
           "date": "2017-12-01/2017-12-31",
           "time": "12:00",
           "format": "grib"
       }, 'download.grib')