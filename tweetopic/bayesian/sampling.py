"""Sampling utilities using jax and blackjax."""
from functools import partial
from typing import Any, Callable

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.hmc import HMCState
from blackjax.types import PyTree
from blackjax.vi.meanfield_vi import MFVIState
from optax import adam
from tqdm import trange

from tweetopic.func import spread

Sampler = Callable[..., tuple[list[PyTree], Any]]


def sample_nuts(
    initial_position: PyTree,
    logdensity_fn: Callable,
    seed: int = 0,
    n_warmup: int = 1000,
    n_samples: int = 1000,
) -> tuple[list[PyTree], list[HMCState]]:
    """NUTS sampling loop for any logdensity function that can be JIT compiled
    with JAX.

    Parameters
    ----------
    initial_position: PyTree
        Python object containing the initial position in parameter space.
        (mean of the prior for example)
    logdensity_fn: function
        JAX jittable function that calculates logdensity given the parameters.
    seed: int, default = 0
        Random seed for sampling.
    n_warmup: int, default = 1000
        Number of HMC window adaptation steps.
    n_samples: int, default = 1000
        Number of samples you intend to obtain from the posterior.

    Returns
    -------
    samples: list of PyTree
        Positions at each step of sampling.
    states: list of HMCState
        State of the Hamiltonian Monte Carlo at each step.
        Mostly useful for debugging.
    """
    rng_key = jax.random.PRNGKey(seed)
    warmup_key, sampling_key = jax.random.split(rng_key)
    print("Warmup, window adaptation")
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    (state, parameters), _ = warmup.run(
        warmup_key, initial_position, num_steps=n_warmup
    )
    kernel = jax.jit(blackjax.nuts(logdensity_fn, **parameters).step)
    states = []
    for i in trange(n_samples, desc="Sampling"):
        _, sampling_key = jax.random.split(sampling_key)
        state, _ = kernel(sampling_key, state)
        states.append(state)
    samples = [state.position for state in states]
    return samples, states


def sample_sgld(
    initial_position: dict,
    logdensity_fn: Callable,
    data: dict,
    seed: int = 0,
    n_samples: int = 1000,
    initial_step_size: float = 1000,
    decay: float = 2.5,
    data_axis: int = 0,
) -> tuple[list[PyTree], None]:
    """Stochastic Gradient Langevin Dynamics sampling loop with decaying step size
    for any logdensity function that is differentiable with JAX.

    Since there is no adaptation step, you have to manually discard
    the samples before the convergence of the Markov chain.

    Parameters
    ----------
    initial_position: PyTree
        Python object containing the initial position in parameter space.
        (mean of the prior for example)
    logdensity_fn: function
        JAX jittable function that calculates logdensity given the parameters.
    seed: int, default 0
        Random seed for sampling.
    n_samples: int, default 1000
        Number of samples you would like to get from the posterior.
    initial_step_size: float, default 1000
        Starting step size for inference.
    decay: float, default 2.5
        Determines how fast the step size decays, 0 means no decay.

    Returns
    -------
    samples: list of PyTree
        Positions at each step of sampling.
    states: None
        Ignored only returned for compatibilty.
    """
    logdensity_fn = spread(partial(logdensity_fn, **data))
    rng_key = jax.random.PRNGKey(seed)
    num_training_steps = n_samples
    schedule_fn = lambda k: initial_step_size * (k ** (-decay))
    schedule = [schedule_fn(i) for i in range(1, num_training_steps + 1)]
    grad_fn = lambda x, _: jax.grad(logdensity_fn)(x)
    sgld = jax.jit(blackjax.sgld(grad_fn))
    position = initial_position
    samples = []
    for i in trange(n_samples, desc="Sampling"):
        _, rng_key = jax.random.split(rng_key)
        position = sgld(rng_key, position, 0, schedule[i])
        samples.append(position)
    return samples, None


def sample_pathfinder(
    initial_position: dict,
    logdensity_fn: Callable,
    data: dict,
    seed: int = 0,
    n_samples: int = 1000,
    data_axis: int = 0,
) -> list[PyTree]:
    logdensity_fn = spread(partial(logdensity_fn, **data))
    rng_key = jax.random.PRNGKey(seed)
    optim_key, sampling_key = jax.random.split(rng_key)
    pathfinder = blackjax.pathfinder(logdensity_fn)
    print("Optimizing normal approximations.")
    state, _ = jax.jit(pathfinder.approximate)(
        rng_key=optim_key, position=initial_position, ftol=1e-4
    )
    print("Sampling approximate normals.")
    samples = pathfinder.sample(sampling_key, state, n_samples)
    return samples


def sample_meanfield_vi(
    initial_position: dict,
    logdensity_fn: Callable,
    data: dict,
    seed: int = 0,
    n_iter: int = 20_000,
    n_samples: int = 1000,
    n_optim_samples: int = 20,
    learning_rate: float = 0.08,
    data_axis: int = 0,
) -> tuple[list[PyTree], list[MFVIState]]:
    logdensity_fn = spread(partial(logdensity_fn, **data))
    rng_key = jax.random.PRNGKey(seed)
    optim_key, sampling_key = jax.random.split(rng_key)
    optimizer = adam(learning_rate)
    mfvi = blackjax.meanfield_vi(
        logdensity_fn, optimizer=optimizer, num_samples=n_optim_samples
    )
    states = []
    state = mfvi.init(initial_position)
    kernel = jax.jit(mfvi.step)
    for i in trange(n_iter, desc="Optimization"):
        _, optim_key = jax.random.split(optim_key)
        state, _ = kernel(optim_key, state)
        states.append(state)
    samples = mfvi.sample(sampling_key, state, num_samples=n_samples)
    return samples, states


def batch_data(rng_key, batch_size: int, data_size: int):
    while True:
        _, rng_key = jax.random.split(rng_key)
        idx = jax.random.choice(
            key=rng_key, a=jnp.arange(data_size), shape=(batch_size,)
        )
        yield idx


def get_batch(idx, data: dict, data_axis: int):
    return {
        key: jnp.take(value, idx, axis=data_axis)
        for key, value in data.items()
    }


# TODO
def sample_minibatch_hmc(
    initial_position: PyTree,
    logdensity_fn: Callable,
    data: dict,
    seed: int = 0,
    batch_size: int = 512,
    step_size: float = 0.001,
    n_warmup: int = 100,
    n_samples: int = 1000,
    data_axis=0,
) -> tuple[list[PyTree], list[HMCState]]:
    rng_key = jax.random.PRNGKey(seed)
    batch_key, warmup_key, sampling_key = jax.random.split(rng_key, 3)
    batches = batch_data(
        batch_key, batch_size, data_size=len(data[list(data.keys())[0]])
    )
    print("Warmup, window adaptation")
    warmup_batch = get_batch(next(batches), data, data_axis=data_axis)
    warmup = blackjax.window_adaptation(
        blackjax.hmc, partial(logdensity_fn, **warmup_batch)
    )
    (state, parameters), _ = warmup.run(
        warmup_key, initial_position, num_steps=n_warmup
    )
    sghmc = blackjax.sghmc()
    kernel = jax.jit(blackjax.nuts(logdensity_fn, **parameters).step)
    states = []
    for i in trange(n_samples, desc="Sampling"):
        _, sampling_key = jax.random.split(sampling_key)
        state, _ = kernel(sampling_key, state)
        states.append(state)
    samples = [state.position for state in states]
    return samples, states
