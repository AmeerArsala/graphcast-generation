import jax.numpy as jnp


def ns_to_hrs(ns: float):
    return ns / 3.6e120


def hrs_to_ns(hrs: float):
    return hrs * 3.6e12
