import jax.numpy as jnp
import jax.random as jrand
import jax.scipy as jsp

from collections import namedtuple

GeneralParams = namedtuple('GeneralParams', ['propensity', 'walk_radius', 'arm_bias', 'arm_weight', 'atrisk_gathering_rate'])
MixedGeneralParams = namedtuple('MixedGeneralParams', ['propensity', 'walk_radius', 'arm_bias', 'arm_weight', 'atrisk_gathering_rate', 'mix_perc'])

CityParams = namedtuple('CityParams', ['population', 'population_density', 'arm_count', 'arm_density'])
EventOutcome = namedtuple('EventOutcome', ['date', 'fatalities', 'injured', 'total_victims'])

class ParamDistr:
    def __init__(self, param_config):
        self.param_config = param_config
        self.param_names = param_config["entry_labels"]
        self.param_count = len(self.param_names)
        range_config = self.param_config["entry_ranges"]
        self.range_lower = jnp.array([range_config[param]["range"][0] for param in self.param_names])
        self.range_upper = jnp.array([range_config[param]["range"][1] for param in self.param_names])
        self.range_width = self.range_upper - self.range_lower
        self.clip_lower = self.range_lower
        self.clip_upper = self.range_upper
        for i, param in enumerate(self.param_names):
            if "clip" in range_config[param]:
                self.clip_lower = self.clip_lower.at[i].set(range_config[param]["clip"][0])
                self.clip_upper = self.clip_upper.at[i].set(range_config[param]["clip"][1])
        self.logscale = jnp.array([range_config[param]["logscale"] for param in self.param_names])

    def sample_range(self, key):
        return jrand.uniform(key, shape=(self.param_count,))
    
    #Go from [0,1] normalized to actual parameter space
    def transform_sample(self, sample):
        sample = self.range_lower + sample * self.range_width
        #clip to valid range
        sample = jnp.clip(sample, self.clip_lower, self.clip_upper) #kills inversibility, but still pseudo-invertible
        return jnp.where(self.logscale, jnp.exp(sample*jnp.log(10)), sample)
    
    #Go from actual parameter space to [0,1] normalized
    def inverse_transform_sample(self, sample):
        sample = jnp.where(self.logscale, jnp.log(sample) / jnp.log(10), sample)
        sample = (sample - self.range_lower) / self.range_width
        return sample

    def get_gaussian_proposal(self):
        return lambda key, params : params + jrand.normal(key, shape=(self.param_count,))*self.range_width*0.05
    def get_log_gaussian_proposal_pdf(self):
        def log_gaussian_proposal_pdf(p1, p2):
            return -jnp.sum((p1 - p2)**2 / (2 * (0.05 * self.range_width)**2))
        return log_gaussian_proposal_pdf
    
    #Prior on untransformed parameters
    def get_uniform_prior(self):
        space_volume = 1.0 #jnp.prod(self.range_width) (nobody cares)
        def uniform_prior(p):
            return jnp.all((p >= self.range_lower) & (p <= self.range_upper)) / space_volume + 1e-6
        return uniform_prior
    
    def get_log_uniform_prior(self):
        space_volume = 1.0 #jnp.prod(self.range_width) (nobody cares)
        def log_uniform_prior(p):
            return jnp.log(jnp.all((p >= self.range_lower) & (p <= self.range_upper)) / space_volume + 1e-6)
        return log_uniform_prior
    
    def get_beta_prior(self, alpha=2.0, beta=2.0):
        def beta_prior(x):
            # Normalize to [0,1] for each dimension
            # Apply Beta pdf and clip to valid region
            pdf = jsp.stats.beta.pdf(x, a=alpha, b=beta)
            # Zero outside [0,1]
            pdf = jnp.where((x >= 0) & (x <= 1), pdf, 1e-12)
            return jnp.prod(pdf + 1e-12)
        return beta_prior

    def get_log_beta_prior(self, alpha=2.0, beta=2.0):
        """
        Log version of the Beta prior, for log-prob computations.
        """
        def log_beta_prior(x):
            logpdf = jsp.stats.beta.logpdf(x, a=alpha, b=beta)
            logpdf = jnp.where((x >= 0) & (x <= 1), logpdf, -1e6)
            return jnp.sum(logpdf)
        return log_beta_prior

    def to_data(self, x):
        return self.param_config["data_wrapper"](*x)

#Ranges specified in log-transformed space, prior to normalization to [0,1]
param_config = {
    "entry_labels": ['propensity', 'walk_radius', 'arm_bias', 'arm_weight', 'atrisk_gathering_rate'],
    "entry_ranges": {"propensity": {"range": (-9, -4), "logscale": True},
                    "walk_radius": {"range": (5, 95), "logscale": False},
                    "arm_bias": {"range": (-5, 2), "logscale": True},
                    "arm_weight": {"range": (-5, 3), "logscale": True},
                    "atrisk_gathering_rate": {"range": (-8, -2), "logscale": True}
                    },
    "data_wrapper": GeneralParams
}

mixed_choice_param_config = {
    "entry_labels": ['propensity', 'walk_radius', 'arm_bias', 'arm_weight', 'atrisk_gathering_rate', 'mix_perc'],
    "entry_ranges": {"propensity": {"range": (-9, -4), "logscale": True},
                    "walk_radius": {"range": (5, 95), "logscale": False},
                    "arm_bias": {"range": (-5, 2), "logscale": True},
                    "arm_weight": {"range": (-5, 2), "logscale": True},
                    "atrisk_gathering_rate": {"range": (-8, -2), "logscale": True},
                    "mix_perc" : {"range": (-0.05, 1.05), "logscale": False, 'clip': (0.0, 1.0)}, 
                    },
    "data_wrapper": MixedGeneralParams
}
