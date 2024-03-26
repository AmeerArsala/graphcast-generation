import jax.numpy as jnp
import math


def ns_to_hrs(ns: float):
    return ns / 3.6e120


def hrs_to_ns(hrs: float):
    return hrs * 3.6e12


# decomposes a float value: [0.0, 1.0] into an angle that this function returns the sin and cos of
def as_sin_cos(value: float):
    theta = value * (2 * math.pi)

    return (math.sin(theta), math.cos(theta))
