import jax.numpy as jnp
import jax.random as jrand

param_config = {
    "propensity_range": {"range": (-9, -5), "logscale": True},
    "walk_radius_range": {"range": (1, 95), "logscale": False},
    "arm_bias_range": {"range": (-5, 3), "logscale": True},
    "arm_weight_range": {"range": (-1.2, -0.9), "logscale": True},
    "atrisk_gather_rate_range": {"range": (-8, -4), "logscale": True}
}

class ParamDistr:
    def __init__(self, param_config):
        self.param_config = param_config
        self.range_lower = jnp.array([config["range"][0] for config in param_config.values()])
        self.range_upper = jnp.array([config["range"][1] for config in param_config.values()])
        self.range_width = self.range_upper - self.range_lower
        self.logscale = jnp.array([config["logscale"] for config in param_config.values()])
        self.param_names = list(param_config.keys())
        self.param_count = len(param_config)

    def sample_range(self, key):
        return jrand.uniform(key, shape=(self.param_count,)) * (self.range_upper - self.range_lower) + self.range_lower
    
    def transform_sample(self, sample):
        return jnp.where(self.logscale, jnp.exp(sample*jnp.log(10)), sample)
    def inverse_transform_sample(self, sample):
        return jnp.where(self.logscale, jnp.log(sample) / jnp.log(10), sample)
    def get_gaussian_proposal(self):
        return lambda key, params : params + jrand.normal(key, shape=(self.param_count,))*self.range_width*0.05
    
    def get_log_gaussian_proposal_pdf(self):
        def log_gaussian_proposal_pdf(p1, p2):
            return -jnp.sum((p1 - p2)**2 / (2 * (0.05 * self.range_width)**2))
        return log_gaussian_proposal_pdf
    
    #Prior on untransformed parameters
    def get_uniform_prior(self):
        space_volume = 1 #jnp.prod(self.range_width)
        def uniform_prior(p):
            return jnp.all((p >= self.range_lower) & (p <= self.range_upper)) / space_volume + 1e-6
        return uniform_prior
    
    def get_log_uniform_prior(self):
        space_volume = 1 #jnp.prod(self.range_width)
        def log_uniform_prior(p):
            return jnp.log(jnp.all((p >= self.range_lower) & (p <= self.range_upper)) / space_volume + 1e-6)
        return log_uniform_prior