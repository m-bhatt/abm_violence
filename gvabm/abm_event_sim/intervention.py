import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.scipy.signal import convolve2d


def sample_event_intervention(walk_radius, population_density, arm_density, atrisk_gathering_rate, mix_perc, rng_key):
    rng_key, subkey = jrand.split(rng_key)

    pgrid = (jrand.uniform(rng_key, (30, 30)) > 0.95).astype(jnp.float32)
    #Use jax to convolve population_grid with a 7x7 gaussian kernel to create pgrid
    kernel = jnp.arange(-4, 5)
    kernel = jnp.exp(-0.5 * (kernel / 2.5) ** 2)
    kernel = kernel / jnp.sum(kernel)
    kernel = jnp.outer(kernel, kernel)
    pgrid = convolve2d(pgrid, kernel, mode='same')
    pgrid = pgrid / jnp.max(pgrid)
    population_grid = pgrid * population_density

    flat_pop_grid = population_grid.flatten()
    probabilities = flat_pop_grid / jnp.sum(flat_pop_grid)

    rng_key, subkey = jrand.split(rng_key)

    max_access = 1
    access_points = jrand.choice(subkey, population_grid.size, (max_access+1,), p=probabilities)

    total_lambda = arm_density * population_grid.size
    access_grid = pgrid / pgrid.sum() * total_lambda
    access_grid = jrand.poisson(subkey, lam=access_grid)

    rng_key, subkey = jrand.split(rng_key)
    start_pos_r, start_pos_c = jnp.unravel_index(access_points[-1], population_grid.shape)
    random_walk = jrand.randint(subkey, (100, 2), -1, 2)
    random_walk = (jnp.cumsum(random_walk, axis=0) + jnp.array([start_pos_r, start_pos_c])) % population_grid.shape[0]
    walk_mask = jnp.arange(100) < walk_radius

    weapon_rolls = jnp.sum(access_grid[random_walk[:, 0], random_walk[:, 1]]*walk_mask)
    weapon_access = (weapon_rolls > 0)
    rng_key, subkey = jrand.split(rng_key)
    high_weapon_access = jrand.bernoulli(subkey, p=1 - jnp.exp(-0.1 * weapon_rolls)) #Assume each weapon has a 10% chance of being high powered

    population_encounters = jnp.sum(population_grid[random_walk[:, 0], random_walk[:, 1]]*walk_mask)

    rng_key, s1, s2 = jrand.split(rng_key, 3)
    location_count = jrand.poisson(s1, lam=population_encounters * atrisk_gathering_rate) + 1
    base_location_count = jrand.poisson(s2, lam=population_grid[start_pos_r, start_pos_c] * atrisk_gathering_rate) + 1

    rng_key, subkey = jrand.split(rng_key)
    max_location = 250
    location_mask = jnp.arange(max_location+5) < location_count
    loc_gathering_size = jrand.weibull_min(subkey, 2, 0.4, (max_location+5,))
    max_gathering_size = jnp.max(loc_gathering_size*location_mask, initial=0)

    base_gathering_size = loc_gathering_size[max_location]
    base_weapon_access = access_grid[start_pos_r, start_pos_c] > 0

    rng_key, subkey = jrand.split(rng_key)
    base_high_weapon_access = jrand.bernoulli(subkey, p=1 - jnp.exp(-0.1)) & base_weapon_access

    rng_key, subkey = jrand.split(rng_key)
    opt_state = jrand.bernoulli(subkey, p=mix_perc)

    gathering_size = jnp.where(opt_state, max_gathering_size, base_gathering_size)
    weapon_access = jnp.where(opt_state, weapon_access, base_weapon_access)
    high_weapon_access = jnp.where(opt_state, high_weapon_access, base_high_weapon_access)
    location_count = jnp.where(opt_state, location_count, base_location_count)

    # binomial_lambda = 12 + 80*high_weapon_access
    # rng_key, subkey = jrand.split(rng_key)
    # fatalities = jrand.binomial(subkey, n=location_count.astype(jnp.int32) + (location_count==0), p=binomial_lambda / (binomial_lambda + location_count), shape=())

    expon_lambda = 6 + 120 * high_weapon_access
    rng_key, subkey = jrand.split(rng_key)
    fatalities = jnp.remainder(jrand.exponential(subkey, shape=()) * expon_lambda, gathering_size)
    return (weapon_access & (location_count > 0)) * fatalities#, opt_state, location_count, gathering_size, weapon_access, high_weapon_access, base_high_weapon_access
