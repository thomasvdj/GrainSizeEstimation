import numpy as np
from math import exp
import pickle
import matplotlib as mpl
from scipy.stats import lognorm, expon, gamma
from statsmodels.distributions import ECDF, StepFunction
import matplotlib.pyplot as plt
mpl.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Sans serif']})
mpl.rc('text', usetex=True)
rng = np.random.default_rng(0)

m = 100
shapes = ["sphere", "dode", "kelvin",  "octa", "cube", "tetra"]
N = 50000
max_area = 5.3*10**(-5)
data = pickle.load(open("lognormal_0.4_50k.pkl", "rb"))


def bias_estimate(sample, estimate):
    y = np.cumsum(sample*np.append(estimate[0], np.diff(estimate)))
    return y / y[-1]


def discrete_cdf(values, cdf_at_values, x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        temp = cdf_at_values[values <= x[i]]
        y[i] = temp[-1]
    return y


for shape in shapes:
    size_ix = 0
    l1_errors = np.zeros(m)
    l1_errors_hb = np.zeros(m)

    estimates = pickle.load(open("lognormal_" + shape + "_estimates.pkl", "rb"))
    estimates_hb = pickle.load(open("lognormal_" + shape + "_estimateshb.pkl", "rb"))

    for j in range(m):
        lag, _ = data[j]
        target_volumes = np.sort(lag.get_volumes())
        h_true = ECDF(target_volumes)
        biased_volumes = bias_estimate(target_volumes ** (1. / 3), np.linspace(0, 1, len(target_volumes)))
        hb_true = StepFunction(target_volumes, biased_volumes, sorted=True, side='right')
        n = len(estimates[j][0])

        jump_locations = np.sort(np.append(estimates[j][0] ** 3, target_volumes))
        diffs = np.append(jump_locations[0], np.diff(jump_locations))
        l1_errors[j] = np.dot(
            np.abs(h_true(jump_locations) - discrete_cdf(estimates[j][0] ** 3, estimates[j][1], jump_locations)), diffs)

        jump_locations = np.sort(np.append(estimates_hb[j][0] ** 3, target_volumes))
        diffs = np.append(jump_locations[0], np.diff(jump_locations))
        l1_errors_hb[j] = np.dot(
            np.abs(hb_true(jump_locations) - discrete_cdf(estimates_hb[j][0]**3, estimates_hb[j][1], jump_locations)), diffs)

    print(shape)
    print(np.mean(l1_errors))
    print(np.mean(l1_errors_hb))

    f = open(shape + "_h_l1_errors.pkl", "wb")
    pickle.dump(l1_errors, f)
    f.close()

    f = open(shape + "_hb_l1_errors.pkl", "wb")
    pickle.dump(l1_errors_hb, f)
    f.close()

