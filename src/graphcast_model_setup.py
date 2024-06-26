import functools
from graphcast import autoregressive
from graphcast import casting
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax


# @title Build jitted functions, and possibly initialize random weights
diffs_stddev_by_level = None
mean_by_level = None
stddev_by_level = None


def set_normalization_data(diffs_stddev_by_level_, mean_by_level_, stddev_by_level_):
    global diffs_stddev_by_level, mean_by_level, stddev_by_level

    diffs_stddev_by_level = diffs_stddev_by_level_
    mean_by_level = mean_by_level_
    stddev_by_level = stddev_by_level_


def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
):
    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics),
    )


def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f
        )
        return loss, (diagnostics, next_state)

    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(
        params, state, inputs, targets, forcings
    )
    return loss, diagnostics, next_state, grads


# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn, model_config, task_config):
    return functools.partial(fn, model_config=model_config, task_config=task_config)


# Always pass params and state, so the usage below are simpler
def with_params(fn, params, state):
    return functools.partial(fn, params=params, state=state)


# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]
