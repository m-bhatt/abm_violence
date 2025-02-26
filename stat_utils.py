from scipy.stats import ecdf
import numpy as np

def grenander_pdf(data):
    res = ecdf(data)
    quantiles = res.cdf.quantiles; quantiles = np.concat([[0], quantiles])
    prob = res.cdf.probabilities; prob = np.concat([[0], prob])

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
    knot_ind = np.array([ki for ki, s in knot_stack] + [len(quantiles)-1])
    knot_slope = np.array([s for ki, s in knot_stack] + [0])
    return quantiles[knot_ind], knot_slope

def grenander_pdf_eval(x, knot_x, knot_slope):
    xp = np.searchsorted(knot_x, x)
    return knot_slope[xp]

def grenander_cdf_eval(x, knot_x, knot_slope):
    knot_del = knot_x[1:] - knot_x[:-1]
    knot_cdf = np.concat([np.zeros((1,)), np.cumsum(knot_del*knot_slope[:-1])])
    xp = np.searchsorted(knot_x, x)
    x_del = x - knot_x[xp]
    x_cdf = x_del*knot_slope[xp] + knot_cdf[xp]
    return x_cdf
