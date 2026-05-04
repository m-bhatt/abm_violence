import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.scipy.signal import convolve2d


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity sweep factory (Tasks B and C)
# ──────────────────────────────────────────────────────────────────────────────

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


from gvabm.param_distr import CityParams, ParamsType, GeneralParams, MixedGeneralParams, BigGeneralParams
def tile_param_tup(param_list, batch_size):
    return tuple(jnp.tile(param, (batch_size,)) for param in param_list)

def build_conditional_sampler(sample_func):
    batch_sample = jax.jit(jax.vmap(sample_func, 0))
    def conditional_sample(city_params: CityParams, general_params: ParamsType, rng_key, num_samples=10000):
        year, population, population_density, arm_count, arm_density = city_params
        propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate = general_params[:5]
        rng_batch = jrand.split(rng_key, num_samples)
        event_samples = batch_sample(jnp.full(num_samples, walk_radius), 
                                        jnp.full(num_samples, population_density), 
                                        jnp.full(num_samples, arm_density*arm_weight+arm_bias), 
                                        jnp.full(num_samples, atrisk_gathering_rate), 
                                        *[jnp.full(num_samples, p) for p in general_params[5:]], 
                                        rng_batch)
        pseudocount_samples = jnp.concatenate([event_samples])
        return pseudocount_samples
    return conditional_sample

def build_conditional_sampler_meta(sample_func):
    batch_sample = jax.jit(jax.vmap(sample_func, 0))
    def conditional_sample(city_params: CityParams, general_params: GeneralParams | MixedGeneralParams, rng_key, num_samples=10000):
        year, population, population_density, arm_count, arm_density = city_params
        propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate = general_params[:5]
        rng_batch = jrand.split(rng_key, num_samples)
        event_samples = batch_sample(jnp.full(num_samples, walk_radius), jnp.full(num_samples, population_density), 
                                 jnp.full(num_samples, arm_density*arm_weight+arm_bias), 
                                 jnp.full(num_samples, atrisk_gathering_rate), 
                                 *[jnp.full(num_samples, p) for p in general_params[5:]], 
                                 rng_batch)
        return event_samples
    return conditional_sample

batch_sample = jax.jit(jax.vmap(sample_event, 0))
def conditional_sample(city_params: CityParams, general_params: GeneralParams, rng_key, num_samples=10000):
    population, population_density, arm_count, arm_density = city_params
    propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate = general_params
    rng_batch = jrand.split(rng_key, num_samples)
    event_samples = batch_sample(jnp.full(num_samples, walk_radius), jnp.full(num_samples, population_density), 
                                 jnp.full(num_samples, arm_density*arm_weight+arm_bias), 
                                 jnp.full(num_samples, atrisk_gathering_rate), rng_batch)
    pseudocount_samples = jnp.concatenate([event_samples, jnp.array([10, 100, 1000])])
    return pseudocount_samples

batch_sample_meta = jax.jit(jax.vmap(sample_event, 0))
def conditional_sample_meta(city_params: CityParams, general_params: GeneralParams, rng_key, num_samples=10000):
    population, population_density, arm_count, arm_density = city_params
    propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate = general_params
    rng_batch = jrand.split(rng_key, num_samples)
    event_samples = batch_sample(jnp.full(num_samples, walk_radius), jnp.full(num_samples, population_density), 
                                 jnp.full(num_samples, arm_density*arm_weight+arm_bias), 
                                 jnp.full(num_samples, atrisk_gathering_rate), rng_batch)
    return event_samples

import numpy as np
def conditional_sample_batch(city_param_l, general_params, rng_key, num_sample_l, batch_size=int(1e6)):
    batch_param_mat = np.zeros((3*(batch_size // 2), 4)) #hack to deal with overflows
    batch_param_mat_size = 0
    batch_num_samples = []

    prior_batch_sample = None
    
    city_ind = 0
    while city_ind < len(city_param_l):
        while batch_param_mat_size < batch_size:
            num_samples = num_sample_l[city_ind]
            population, population_density, arm_count, arm_density = city_param_l[city_ind]
            propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate = general_params
            param_mat = np.tile(np.array([walk_radius, population_density, 
                                        arm_density*arm_weight+arm_bias, 
                                        atrisk_gathering_rate]).reshape((1, -1)), 
                                        (num_samples, 1))
            batch_param_mat[batch_param_mat_size:batch_param_mat_size+num_samples, :] = param_mat
            #test code, do remove
            
            batch_param_mat_size += num_samples
            batch_num_samples.append(num_samples)
            city_ind += 1
            if city_ind >= len(city_param_l):
                break

        #run batch
        rng_key, subkey = jrand.split(rng_key, 2)
        rng_batch = jrand.split(subkey, batch_size)
        batch_event_samples = batch_sample(*[jnp.array(batch_param_mat[:batch_size, i]) for i in range(4)], rng_batch)

        #output batch
        chunked_event_samples = []
        next_prior_batch_sample = None
        si = 0
        for ns in batch_num_samples:
            if si + ns > batch_size:
                next_prior_batch_sample = batch_event_samples[si:]
                batch_num_samples = [(si + ns - batch_size)]
                break
            chunked_event_samples.append(batch_event_samples[si:si+ns])
            si += ns

        if not prior_batch_sample is None:
            chunked_event_samples[0] = jnp.concat([prior_batch_sample, chunked_event_samples[0]])
        if next_prior_batch_sample is None:
            batch_num_samples = []
        prior_batch_sample = next_prior_batch_sample

        for chunk in chunked_event_samples:
            yield jnp.concat([chunk, jnp.array([1, 10, 100])])
        if city_ind >= len(city_param_l):
            return
        
        batch_param_mat[:(batch_param_mat_size - batch_size), :] = batch_param_mat[batch_size:batch_param_mat_size, :]
        batch_param_mat_size = (batch_param_mat_size - batch_size)
        # print(f'Batch size: {batch_size}, batch_param_mat_size: {batch_param_mat_size}')
        if not prior_batch_sample is None:
            assert batch_num_samples[0] == batch_param_mat_size
            assert ns == (batch_num_samples[0] + prior_batch_sample.shape[0])

batch_sample_no_selection = jax.jit(jax.vmap(sample_event_no_selection, 0))
def conditional_sample_noselection(city_params: CityParams, general_params: GeneralParams, rng_key, num_samples=10000):
    population, population_density, arm_count, arm_density = city_params
    propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate = general_params
    rng_batch = jrand.split(rng_key, num_samples)
    event_samples = batch_sample_no_selection(jnp.full(num_samples, walk_radius), jnp.full(num_samples, population_density), 
                                 jnp.full(num_samples, arm_density*arm_weight+arm_bias), 
                                 jnp.full(num_samples, atrisk_gathering_rate), rng_batch)
    pseudocount_samples = jnp.concatenate([event_samples, jnp.array([10, 100, 1000])])
    return pseudocount_samples

batch_sample_mixed = jax.jit(jax.vmap(sample_event_mixed_selection, 0))
def conditional_sample_mixed(city_params: CityParams, general_params: MixedGeneralParams, rng_key, num_samples=10000):
    population, population_density, arm_count, arm_density = city_params
    propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate, mix_perc = general_params
    rng_batch = jrand.split(rng_key, num_samples)
    event_samples = batch_sample_mixed(jnp.full(num_samples, walk_radius), jnp.full(num_samples, population_density), 
                                 jnp.full(num_samples, arm_density*arm_weight+arm_bias), 
                                 jnp.full(num_samples, atrisk_gathering_rate), jnp.full(num_samples, mix_perc), rng_batch)
    pseudocount_samples = jnp.concatenate([event_samples, jnp.array([10, 100, 1000])])
    return pseudocount_samples