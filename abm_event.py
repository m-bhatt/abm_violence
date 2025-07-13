import jax
import jax.numpy as jnp
import jax.random as jrand

def sample_event(walk_radius, population_density, arm_density, atrisk_gathering_rate, rng_key):
    rng_key, subkey = jrand.split(rng_key)
    population_grid = jnp.full((30, 30), population_density)
    access_count = jrand.poisson(subkey, lam=arm_density * population_grid.size)

    flat_pop_grid = population_grid.flatten()
    probabilities = flat_pop_grid / jnp.sum(flat_pop_grid)
    
    rng_key, subkey = jrand.split(rng_key)
    max_access = 50
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
    
    # atrisk_gathering_rate = 1e-6
    population_encounters = jnp.sum(walk_grid * population_grid)
    rng_key, subkey = jrand.split(rng_key)
    
    max_location = 20
    location_count = jrand.poisson(subkey, lam=population_encounters * atrisk_gathering_rate)
    location_mask = jnp.arange(max_location) < location_count
    rng_key, subkey = jrand.split(rng_key)
    loc_gathering_size = jrand.weibull_min(subkey, 10, 0.4, (max_location,))
    gathering_size = jnp.max(loc_gathering_size*location_mask, initial=0)
    rng_key, subkey = jrand.split(rng_key)
    return ((access_count > 0) & weapon_access & (location_count > 0)) * jrand.uniform(subkey, (), minval=0, maxval=gathering_size)

batch_sample = jax.jit(jax.vmap(sample_event, 0), device=jax.devices()[1])

from stat_utils import CityParams, GeneralParams
def conditional_sample(city_params: CityParams, general_params: GeneralParams, rng_key, num_samples=10000):
    population, population_density, arm_count, arm_density = city_params
    propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate = general_params
    # s = int(1e5)
    rng_batch = jrand.split(rng_key, num_samples)
    event_samples = batch_sample(jnp.full(num_samples, walk_radius), jnp.full(num_samples, population_density), 
                                 jnp.full(num_samples, arm_density*arm_weight+arm_bias), 
                                 jnp.full(num_samples, atrisk_gathering_rate), rng_batch)
    return event_samples