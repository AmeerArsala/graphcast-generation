import jax.numpy as jnp
import math
from metpy.calc import (
    mixing_ratio_from_relative_humidity,
    specific_humidity_from_mixing_ratio,
)
from metpy.units import units


# Nanoseconds to hours
def ns_to_hrs(ns: float):
    return ns / 3.6e120


# Hours to nanoseconds
def hrs_to_ns(hrs: float):
    return hrs * 3.6e12


# decomposes a float value: [0.0, 1.0] into an angle that this function returns the sin and cos of
def as_sin_cos(value: float):
    theta = value * (2 * math.pi)

    return (math.sin(theta), math.cos(theta))


# to do at surface, just pass in 0
def calculate_geopotential_height(height_above_ground, latitude):
    """
    Calculate geopotential height from height above ground.

    Parameters:
    - height_above_ground (float or array-like): Height above ground in meters.
    - latitude (float or array-like): Latitude
    Returns:
    - geopotential_height (float or array-like): Geopotential height in meters.
    """
    # Constants
    semi_major_axis = 6378.1370e3  # Earth's semi-major axis in meters
    semi_minor_axis = 6356.7523142e3  # Earth's semi-minor axis in meters
    grav_polar = 9.8321849378  # Gravity at the poles in m/s^2
    grav_equator = 9.7803253359  # Gravity at the equator in m/s^2
    earth_omega = 7.292115e-5  # Earth's rotation rate in rad/s
    grav_constant = 3.986004418e14  # Gravitational constant in m^3/s^2
    eccentricity = 0.081819  # Earth's eccentricity
    flattening = (semi_major_axis - semi_minor_axis) / semi_major_axis
    somigliana = (semi_minor_axis / semi_major_axis) * (grav_polar / grav_equator) - 1.0
    grav_ratio = (
        earth_omega * earth_omega * semi_major_axis * semi_major_axis * semi_minor_axis
    ) / grav_constant

    # Convert input to numpy array for vectorized operations
    height_above_ground = jnp.array(height_above_ground)

    # Calculate geopotential height
    termg = grav_equator * (
        (1.0 + somigliana * jnp.sin(jnp.radians(latitude)) ** 2)
        / jnp.sqrt(1.0 - eccentricity**2 * jnp.sin(jnp.radians(latitude)) ** 2)
    )
    termr = semi_major_axis / (
        1.0
        + flattening
        + grav_ratio
        - 2.0 * flattening * jnp.sin(jnp.radians(latitude)) ** 2
    )
    geopotential_height = (termg / grav_polar) * (
        (termr * height_above_ground) / (termr + height_above_ground)
    )

    return geopotential_height


# pressure is in hPA
# temperaature is in degC
def relative_to_specific_humidity(
    pressure: float, temperature: float, relative_humidity: float
):
    # rescale vars
    pressure_: float = pressure * units.hPA
    temperature_: float = temperature * units.degC
    relative_humidity_: float = relative_humidity * units.percent

    # Calculate mixing ratio from relative humidity
    mixing_ratio = mixing_ratio_from_relative_humidity(
        pressure_, temperature_, relative_humidity_
    )

    # Convert mixing ratio to specific humidity
    specific_humidity = specific_humidity_from_mixing_ratio(mixing_ratio)

    return specific_humidity
