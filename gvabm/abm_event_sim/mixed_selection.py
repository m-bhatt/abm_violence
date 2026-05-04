import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.scipy.signal import convolve2d


def sample_event_mixed_selection(walk_radius, population_density, arm_density, atrisk_gathering_rate, mix_rate, rng_key):
    rng_key, subkey = jrand.split(rng_key)
    population_grid = jnp.full((30, 30), population_density)
    access_count = jrand.poisson(subkey, lam=arm_density * population_grid.size)
    flat_pop_grid = population_grid.flatten()
    probabilities = flat_pop_grid / jnp.sum(flat_pop_grid)
    rng_key, subkey = jrand.split(rng_key)

    max_access = 250
    access_points = jrand.choice(subkey, population_grid.size, (max_access,), p=probabilities)
    point_mask = jnp.arange(max_access) < access_count
    access_grid = jnp.zeros_like(population_grid)
    access_grid = access_grid.at[jnp.unravel_index(access_points, population_grid.shape)].set(point_mask)

    rng_key, subkey = jrand.split(rng_key)
    random_walk = jrand.randint(subkey, (100, 2), -1, 2)
    random_walk = jnp.cumsum(random_walk, axis=0) % population_grid.shape[0]
    walk_mask = jnp.arange(100) < walk_radius
    
    weapon_access = jnp.any(access_grid[random_walk[:, 0], random_walk[:, 1]]*walk_mask)

    walk_grid = jnp.zeros_like(population_grid)
    walk_grid = walk_grid.at[random_walk[:, 0], random_walk[:, 1]].set(walk_mask)
    population_encounters = jnp.sum(walk_grid * population_grid)
    rng_key, subkey = jrand.split(rng_key)

    max_location = 250
    location_count = jrand.poisson(subkey, lam=population_encounters * atrisk_gathering_rate)
    location_mask = jnp.arange(max_location) < location_count
    rng_key, subkey = jrand.split(rng_key)
    loc_gathering_size = jrand.weibull_min(subkey, 2.0, 0.4, (max_location,))
    max_gathering_size = jnp.max(loc_gathering_size*location_mask, initial=0)
    sample_gathering_size = jrand.weibull_min(subkey, 2.0, 0.4)
    sample_mask = jrand.uniform(subkey, ()) < mix_rate
    gathering_size = jnp.where(sample_mask, sample_gathering_size, max_gathering_size)
    rng_key, subkey = jrand.split(rng_key)
    return ((access_count > 0) & weapon_access & (location_count > 0)) * jrand.uniform(subkey, (), minval=0, maxval=gathering_size)
