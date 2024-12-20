import numpy as np
from math import exp
import pickle
import matplotlib as mpl
from statsmodels.distributions import ECDF, StepFunction
import matplotlib.pyplot as plt
mpl.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Sans serif']})
mpl.rc('text', usetex=True)
rng = np.random.default_rng(0)

m = 100
sigma = 0.4
mu = -1*sigma*sigma/2
shapes = ["sphere", "dode", "kelvin",  "octa", "cube", "tetra"]
N = 50000
# target_volumes = rng.lognormal(mean=mu, sigma=sigma, size=N)
# target_volumes = target_volumes / np.sum(target_volumes)
# h_true = ECDF(target_volumes)
max_area = 5.3*10**(-5)
data = pickle.load(open("lognormal_0.4_50k.pkl", "rb"))


def bias_estimate(sample, estimate):
    y = np.cumsum(sample*np.append(estimate[0], np.diff(estimate)))
    return y / y[-1]


for shape in shapes:
    size_ix = 0
    sup_errors = np.zeros(m)
    sup_errors_hb = np.zeros(m)

    estimates = pickle.load(open("lognormal_" + shape + "_estimates.pkl", "rb"))
    estimates_hb = pickle.load(open("lognormal_" + shape + "_estimateshb.pkl", "rb"))

    for j in range(m):
        lag, _ = data[j]
        target_volumes = np.sort(lag.get_volumes())
        h_true = ECDF(target_volumes)
        biased_volumes = bias_estimate(target_volumes ** (1. / 3), np.linspace(0, 1, len(target_volumes)))
        hb_true = StepFunction(target_volumes, biased_volumes, sorted=True, side='right')

        n = len(estimates[j][0])

        dist1 = np.max(np.abs(estimates[j][1] - h_true(estimates[j][0] ** 3)))
        dist2 = np.max(np.abs(np.append(0, estimates[j][1])[:n] - h_true(estimates[j][0] ** 3)))
        sup_errors[j] = max(dist1, dist2)

        dist1 = np.max(np.abs(estimates_hb[j][1] - hb_true(estimates_hb[j][0]**3)))
        dist2 = np.max(np.abs(np.append(0, estimates_hb[j][1])[:n] - hb_true(estimates_hb[j][0]**3)))
        sup_errors_hb[j] = max(dist1, dist2)

    print(shape)
    print(np.mean(sup_errors))
    print(np.mean(sup_errors_hb))

    f = open(shape + "_h_errors.pkl", "wb")
    pickle.dump(sup_errors, f)
    f.close()

    f = open(shape + "_hb_errors.pkl", "wb")
    pickle.dump(sup_errors_hb, f)
    f.close()

