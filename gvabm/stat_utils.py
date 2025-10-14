from scipy.stats import ecdf
import numpy as np
from numba import jit, njit

def logrange(start, end, num):
    return np.exp(np.linspace(np.log(start), np.log(end), num))

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

class EmpiricalCDF:
    def __init__(self, data):
        self.data = data

    def pdf(self, x):
        # PDF is not well-defined for empirical CDF, return 0
        #throw error
        raise NotImplementedError("PDF is not defined for empirical CDF")

    def cdf(self, x):
        return np.mean(self.data <= x)  

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


# Numba-accelerated versions of Grenander functions

@njit
def grenander_pdf_eval_numba(x, knot_x, knot_slope):
    """Numba-accelerated PDF evaluation for scalar x."""
    # Binary search for position
    xp = np.searchsorted(knot_x, x) - 1
    if xp < 0:
        xp = 0
    elif xp >= len(knot_slope):
        xp = len(knot_slope) - 1
    return knot_slope[xp]

@njit
def grenander_pdf_eval_numba_vec(x, knot_x, knot_slope):
    """Numba-accelerated PDF evaluation for array x."""
    result = np.empty(len(x))
    for i in range(len(x)):
        xp = np.searchsorted(knot_x, x[i]) - 1
        if xp < 0:
            xp = 0
        elif xp >= len(knot_slope):
            xp = len(knot_slope) - 1
        result[i] = knot_slope[xp]
    return result

@njit
def grenander_cdf_eval_numba(x, knot_x, knot_slope):
    """Numba-accelerated CDF evaluation for scalar x."""
    # Compute knot CDFs
    n_knots = len(knot_x)
    knot_cdf = np.zeros(n_knots)
    for i in range(1, n_knots):
        knot_del = knot_x[i] - knot_x[i-1]
        knot_cdf[i] = knot_cdf[i-1] + knot_del * knot_slope[i-1]
    
    # Find position and compute CDF
    xp = np.searchsorted(knot_x, x) - 1
    if xp < 0:
        return 0.0
    elif xp >= len(knot_slope) - 1:
        return knot_cdf[-1]
    
    x_del = x - knot_x[xp]
    return x_del * knot_slope[xp] + knot_cdf[xp]

@njit
def grenander_cdf_eval_numba_vec(x, knot_x, knot_slope):
    """Numba-accelerated CDF evaluation for array x."""
    # Compute knot CDFs once
    n_knots = len(knot_x)
    knot_cdf = np.zeros(n_knots)
    for i in range(1, n_knots):
        knot_del = knot_x[i] - knot_x[i-1]
        knot_cdf[i] = knot_cdf[i-1] + knot_del * knot_slope[i-1]
    
    # Evaluate CDF for each x
    result = np.empty(len(x))
    for i in range(len(x)):
        xp = np.searchsorted(knot_x, x[i]) - 1
        if xp < 0:
            result[i] = 0.0
        elif xp >= len(knot_slope) - 1:
            result[i] = knot_cdf[-1]
        else:
            x_del = x[i] - knot_x[xp]
            result[i] = x_del * knot_slope[xp] + knot_cdf[xp]
    return result

@njit
def compute_ecdf_numba(data):
    """Compute empirical CDF using numba - returns unique quantiles and probabilities."""
    # Sort data
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Find unique values and their counts
    unique_vals = []
    counts = []
    current_val = sorted_data[0]
    current_count = 1
    
    for i in range(1, n):
        if sorted_data[i] == current_val:
            current_count += 1
        else:
            unique_vals.append(current_val)
            counts.append(current_count)
            current_val = sorted_data[i]
            current_count = 1
    unique_vals.append(current_val)
    counts.append(current_count)
    
    # Convert to arrays
    quantiles = np.array(unique_vals)
    probs = np.cumsum(np.array(counts)) / n
    
    return quantiles, probs

@njit
def grenander_algorithm_numba(quantiles, prob):
    """
    Numba-accelerated Grenander isotonic regression algorithm.
    Returns indices of knots and their slopes.
    """
    n = len(quantiles)
    
    # Use arrays instead of list for numba compatibility
    knot_indices = np.empty(n, dtype=np.int64)
    knot_slopes = np.empty(n, dtype=np.float64)
    stack_size = 0
    
    si = 0
    ni = 1
    
    while ni < n:
        slope = (prob[ni] - prob[si]) / (quantiles[ni] - quantiles[si])
        
        # Check if we need to merge with previous segment
        merged = False
        while stack_size > 0:
            prev_idx = knot_indices[stack_size - 1]
            prev_slope = knot_slopes[stack_size - 1]
            
            if prev_slope <= slope:
                # Pop the stack
                stack_size -= 1
                si = prev_idx
                slope = (prob[ni] - prob[si]) / (quantiles[ni] - quantiles[si])
            else:
                break
        
        # Add new segment to stack
        knot_indices[stack_size] = si
        knot_slopes[stack_size] = slope
        stack_size += 1
        
        si = ni
        ni += 1
    
    # Return only the used portion of arrays
    return knot_indices[:stack_size], knot_slopes[:stack_size]

def grenander_pdf_numba(data):
    """
    Create a Grenander distribution using numba-accelerated computation.
    
    Parameters:
    - data: array of data points
    
    Returns:
    - GrenanderDistrNumba instance
    """
    data = np.asarray(data, dtype=np.float64)
    
    # Compute ECDF using numba
    quantiles, prob = compute_ecdf_numba(data)
    
    # Prepend zero
    quantiles_ext = np.concatenate([np.array([0.0]), quantiles])
    prob_ext = np.concatenate([np.array([0.0]), prob])
    
    # Run Grenander algorithm
    knot_ind, knot_slopes = grenander_algorithm_numba(quantiles_ext, prob_ext)
    
    # Append final slope of 0
    knot_slopes_final = np.concatenate([knot_slopes, np.array([0.0])])
    
    # Build knot_x array
    knot_x = np.concatenate([quantiles_ext[knot_ind], np.array([quantiles_ext[-1]])])
    
    return GrenanderDistrNumba(knot_x, knot_slopes_final)

class GrenanderDistrNumba:
    """Grenander distribution with numba-accelerated evaluation."""
    
    def __init__(self, knot_x, knot_slope):
        self.knot_x = np.asarray(knot_x, dtype=np.float64)
        self.knot_slope = np.asarray(knot_slope, dtype=np.float64)
    
    def pdf(self, x):
        """Evaluate PDF at x (scalar or array)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            return grenander_pdf_eval_numba(x, self.knot_x, self.knot_slope)
        else:
            return grenander_pdf_eval_numba_vec(x, self.knot_x, self.knot_slope)
    
    def cdf(self, x):
        """Evaluate CDF at x (scalar or array)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            return grenander_cdf_eval_numba(x, self.knot_x, self.knot_slope)
        else:
            return grenander_cdf_eval_numba_vec(x, self.knot_x, self.knot_slope)


class HybridDistribution:
    def __init__(self, base_distr, tail_distr, threshold):
        # self.zero_prob = zero_prob
        self.base_distr = base_distr #P(x)
        self.tail_distr = tail_distr #P(x | x >= threshold)
        self.threshold = threshold
        # self.base_pdf_thres = self.base_distr.pdf(threshold)
        self.base_cdf_thres = self.base_distr.cdf(threshold)
    
    def pdf(self, x):
        base_pdf = self.base_distr.pdf(x)
        tail_pdf = (1-self.base_cdf_thres)*self.tail_distr.pdf(x)
        return np.where(x < self.threshold, base_pdf, tail_pdf)
        
    def cdf(self, x):
        base_cdf = self.base_distr.cdf(x)
        tail_cdf = self.base_cdf_thres + (1 - self.base_cdf_thres)*self.tail_distr.cdf(x)
        return np.where(x < self.threshold, base_cdf, tail_cdf)

from scipy.stats import genpareto
def hybrid_pdf(data):
    # zero_prob = np.mean(data == 0)
    # data = data[data > 0]
    base_distr = grenander_pdf_numba(data)
    tail_data = data[data > 1]
    partition_index = max(len(tail_data) - 500, 0)
    threshold = np.sort(tail_data)[partition_index]
    tail_data = tail_data[tail_data >= threshold]
    print(threshold, len(tail_data))
    if len(tail_data) < 5:
        return base_distr
    fitted_shape, fitted_loc, fitted_scale = genpareto.fit(tail_data)
    tail_distr = genpareto(c=fitted_shape, loc=fitted_loc, scale=fitted_scale)
    return HybridDistribution(base_distr, tail_distr, threshold)