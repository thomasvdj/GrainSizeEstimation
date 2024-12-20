import numpy as np
from math import pi
import vorostereology as vs
import matplotlib.pyplot as plt
import pickle
rng = np.random.default_rng(0)


low = 0.0
high = 1.0
domain = [[low, high], [low, high], [low, high]]
sphericities = []
counter = 0

while counter < 1000000:
    n = rng.poisson(lam=50000)
    counter += n
    print(n)
    points = rng.uniform(size=(n, 3), low=low, high=high)
    weights = np.zeros(n)
    lag = vs.Laguerre3D(points, weights, domain, periodic=True)
    volumes = lag.get_volumes()
    surface_areas = lag.get_surface_areas()
    sphericities.append(pi**(1./3)*np.power(6*volumes, 2./3)/surface_areas)

sphericities = np.concatenate(sphericities)[:1000000]
print("Mean value", np.mean(sphericities))

f = open("sphericities_1000000.pkl", "wb")
pickle.dump(sphericities, f)
f.close()

plt.figure(figsize=(4.0, 3.0))
plt.hist(sphericities, bins=100, ec='black', linewidth=0.2, density=True)
plt.ylabel("Density")
plt.xlabel("Deviation from spherical shape")
plt.title("Sphericity of a typical Voronoi cell")
plt.savefig("typical_sphericity_distribution.pdf")
#plt.show()
