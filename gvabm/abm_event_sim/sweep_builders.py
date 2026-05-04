import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.scipy.signal import convolve2d


def make_noopt_event(high_weapon_prob: float = 0.10, grid_size: int = 30):
    """
    Factory: return a nooptstate event sampler with configurable sensitivity params.

    Parameters
    ----------
    high_weapon_prob : float
        Per-encounter probability that a weapon roll yields a high-lethality
        weapon: ``p = 1 - exp(-high_weapon_prob * weapon_rolls)``.
        Baseline is 0.10 (matches sample_event_optstate_noopt).
    grid_size : int
        Side length N of the N×N spatial grid.  Baseline is 30.
        walk_radius is in grid steps, so the same step count covers a larger
        physical fraction of a smaller grid.  total_lambda = arm_density * N²
        (consistent with the baseline formula).

    Returns
    -------
    Callable with the same signature as sample_event_optstate_noopt.
    """
    def _sample(walk_radius, population_density, arm_density, atrisk_gathering_rate, rng_key):
        rng_key, subkey = jrand.split(rng_key)
        pgrid = (jrand.uniform(rng_key, (grid_size, grid_size)) > 0.95).astype(jnp.float32)
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
        access_points = jrand.choice(subkey, population_grid.size, (max_access + 1,), p=probabilities)

        total_lambda = arm_density * population_grid.size
        access_grid = pgrid / pgrid.sum() * total_lambda
        access_grid = jrand.poisson(subkey, lam=access_grid)

        rng_key, subkey = jrand.split(rng_key)
        start_pos_r, start_pos_c = jnp.unravel_index(access_points[-1], population_grid.shape)
        random_walk = jrand.randint(subkey, (100, 2), -1, 2)
        random_walk = (jnp.cumsum(random_walk, axis=0) + jnp.array([start_pos_r, start_pos_c])) % population_grid.shape[0]
        walk_mask = jnp.arange(100) < walk_radius

        weapon_rolls = jnp.sum(access_grid[random_walk[:, 0], random_walk[:, 1]] * walk_mask)
        weapon_access = weapon_rolls > 0
        rng_key, subkey = jrand.split(rng_key)
        high_weapon_access = jrand.bernoulli(subkey, p=1 - jnp.exp(-high_weapon_prob * weapon_rolls))

        population_encounters = jnp.sum(population_grid[random_walk[:, 0], random_walk[:, 1]] * walk_mask)
        rng_key, s1, s2 = jrand.split(rng_key, 3)
        location_count = jrand.poisson(s1, lam=population_encounters * atrisk_gathering_rate + 1)

        rng_key, subkey = jrand.split(rng_key)
        max_location = 250
        location_mask = jnp.arange(max_location + 5) < location_count
        loc_gathering_size = jrand.weibull_min(subkey, 2, 0.4, (max_location + 5,))
        gathering_size = loc_gathering_size[max_location]

        rng_key, subkey = jrand.split(rng_key)
        expon_lambda = 6 + 120 * high_weapon_access
        rng_key, subkey = jrand.split(rng_key)
        fatalities = jnp.remainder(jrand.exponential(subkey, shape=()) * expon_lambda, gathering_size)
        return (weapon_access & (location_count > 0)) * fatalities

    return _sample