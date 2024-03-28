# GraphCast

**Weather State**

(batch, time, lat, lon, level)

- lat, lon

  - 721 x 1440
  - 0.25 degree precision

- time

  - timedelta64[ns] where ns = nanoseconds
  - 2 for input, 1 for output

- level

  - atmospheric pressure in hPA
  - For the purposes of SAR, we can just do `array([ 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=int32)`

![](file://C:\Users\night\AppData\Roaming\marktext\images\2024-03-25-13-12-09-image.png?msec=1711397529554)

- Need (per batch):

  - @time: timedelta64[ns]

    - @lat, lon (0.25 degree precision)

      - @`2m_temperature` --> `":TMP:2 m"`
      - @`10m_v_component_of_wind` --> `":VGRD:10 m"`
      - @`10m_u_component_of_wind` --> `":UGRD:10 m"`
      - !@ `total_precipitation_6hr` --> `":APCP:"`
      - @level (hPA)

        - @`temperature` --> `":TMP:"`
        - @`geopotential` --> `calculate_geopotential_height(level, lat)`
        - @`u_component_of_wind` --> `":UGRD:\d+ mb"`
        - @`v_component_of_wind` --> `":VGRD:\d+ mb"`
        - \*`vertical_velocity`
        - @`specific_humidity` --> `":RH:"` |> `relative_to_specific_humidity(level, temp_degC, RH)`

      - @EXTRA_IN: `toa_incident_solar_radiation` --> `graphcast.add_tisr_var(xarray.Dataset)`

    - @EXTRA_IN: elapsed year progress = `year_progress_sin`, `year_progress_cos`
    - @lon (for timezone)

      - @EXTRA_IN: local time of day = `day_progress_sin`, `day_progress_cos`

  - @lat, lon

    - @EXTRA_IN: `geopotential_at_surface` --> `calculate_geopotential_height(0, lat)`
    - \*EXTRA_IN: `land_sea_mask`

Links:

https://openweathermap.org/history-bulk#parameter

https://raphaelnussbaumer.com/GeoPressureAPI/

https://github.com/cjabradshaw/Anemometer?tab=readme-ov-file
