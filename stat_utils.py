from scipy.stats import ecdf
import numpy as np

from collections import namedtuple
GeneralParams = namedtuple('GeneralParams', ['propensity', 'walk_radius', 'arm_bias', 'arm_weight', 'atrisk_gathering_rate'])
CityParams = namedtuple('CityParams', ['population', 'population_density', 'arm_count', 'arm_density'])
EventOutcome = namedtuple('EventOutcome', ['date', 'fatalities', 'injured', 'total_victims'])

def logrange(start, end, num):
    return np.exp(np.linspace(np.log(start), np.log(end), num))

def sample_event(walk_radius, population_density, arm_density, population_grid, access_grid, walk_grid, rng): 
    population_grid.fill(population_density)
    access_count = rng.poisson(arm_density*population_grid.size)
    if(access_count <= 0):
        return 0
    access_points = rng.choice(population_grid.size, access_count, p=population_grid.flatten() / population_grid.sum())
    access_grid.fill(0)
    access_grid.reshape((-1,))[access_points] = 1

    random_walk = rng.integers(-1, 2, (walk_radius, 2))
    random_walk = np.cumsum(random_walk, axis=0) % population_grid.shape[0]
    weapon_access = np.any(access_grid[random_walk[:, 0], random_walk[:, 1]])
    if(not weapon_access):
        return 0
    walk_grid.fill(0)
    walk_grid[random_walk[:, 0], random_walk[:, 1]] = 1
    atrisk_gathering_rate = 1e-6
    population_encounters = np.sum(walk_grid*population_grid)
    location_count = rng.poisson(population_encounters*atrisk_gathering_rate)
    if(location_count <= 0):
        return 0
    loc_gathering_size = 10*rng.weibull(0.4, (location_count,))
    gathering_size = np.max(loc_gathering_size)
    return np.random.uniform(0, gathering_size)

def grenander_pdf_eval(x, knot_x, knot_slope):
    xp = np.searchsorted(knot_x, x)-1
    return knot_slope[xp]

def grenander_cdf_eval(x, knot_x, knot_slope):
    knot_del = knot_x[1:] - knot_x[:-1]
    knot_cdf = np.concatenate([np.zeros((1,)), np.cumsum(knot_del*knot_slope[:-1])])
    xp = np.searchsorted(knot_x, x) - 1
    x_del = x - knot_x[xp]
    x_cdf = x_del*knot_slope[xp] + knot_cdf[xp]
    return x_cdf

class GrenanderDistr:
    def __init__(self, knot_x, knot_slope):
        self.knot_x = knot_x
        self.knot_slope = knot_slope
    def pdf(self, x):
        return grenander_pdf_eval(x, self.knot_x, self.knot_slope)
    def cdf(self, x):
        return grenander_cdf_eval(x, self.knot_x, self.knot_slope)

def grenander_pdf(data):
    res = ecdf(data)
    quantiles = res.cdf.quantiles; quantiles = np.concatenate([[0], quantiles])
    prob = res.cdf.probabilities; prob = np.concatenate([[0], prob])

    knot_stack = []
    si, ni = 0, 1
    while ni < len(quantiles):
        slope = (prob[ni] - prob[si]) / (quantiles[ni] - quantiles[si])
        if knot_stack:
            knot_i, prev_slope = knot_stack[-1]
            if(prev_slope <= slope):
                knot_stack.pop()
                si = knot_i
                continue
        knot_stack.append((si, slope))
        si = ni
        ni += 1
    knot_ind = np.array([ki for ki, s in knot_stack])
    knot_slope = np.array([s for ki, s in knot_stack] + [0])
    knot_x = np.concatenate([quantiles[knot_ind], [quantiles[-1]]])
    return GrenanderDistr(knot_x, knot_slope)
