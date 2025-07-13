import jax.numpy as jnp
import jax.random as jrand

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
        return jnp.where(self.logscale, jnp.exp(sample), sample)
    def inverse_transform_sample(self, sample):
        return jnp.where(self.logscale, jnp.log(sample), sample)
    def get_gaussian_proposal(self):
        return lambda key, params : params + jrand.normal(key, shape=(self.param_count,))*self.range_width*0.05
    
    def get_log_gaussian_proposal_pdf(self):
        def log_gaussian_proposal_pdf(p1, p2):
            return -jnp.sum((p1 - p2)**2 / (2 * (0.05 * self.range_width)**2))
        return log_gaussian_proposal_pdf
    
    def get_uniform_prior(self):
        def uniform_prior(p):
            return jnp.all((p >= self.range_lower) & (p <= self.range_upper))+1e-3
        return uniform_prior