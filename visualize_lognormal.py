import pickle
import matplotlib.pyplot as plt
import numpy as np
max_double = np.finfo(np.double).max*0.1


data = pickle.load(open("lognormal_0.4_50k.pkl", "rb"))
laguerre, section = data[0]
volumes = np.sort(laguerre.get_volumes())
m = 100
k = 1000
max_area = 5.3*10**(-5)
estimates = pickle.load(open("lognormal_sphere_estimates.pkl", "rb"))

estimate_pts = [np.append(np.append(0, est[0]**3), max_double) for est in estimates]
estimate_cdf = [np.append(np.append(0, est[1]), 1) for est in estimates]
x = np.linspace(0.01*10**(-5), max_area, k)
cdf_matrix = np.zeros((m, k))
for i in range(m):
    for j in range(k):
        ix = np.searchsorted(estimate_pts[i], x[j])-1
        print(ix)
        cdf_matrix[i, j] = estimate_cdf[i][ix]
mean_cdf = np.mean(cdf_matrix, axis=0)

plt.figure(figsize=(3.0, 2.4))
for i in range(m):
    plt.step(estimate_pts[i], estimate_cdf[i], where="post", alpha=0.3, c="tab:blue", rasterized=True)
plt.plot(x, mean_cdf, c="black", label="Average\n estimate")
plt.step(np.append(np.append(0, volumes), max_double), np.append(np.append(0, np.linspace(0, 1, len(volumes))), 1),
         where="post", label="3D Laguerre", c="red", linestyle="dashed")
plt.xlim([0, max_area])
plt.title(r"Estimates of volume CDF")
plt.xlabel(r"Volume")
plt.ylabel("CDF")
plt.legend()
plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(top=0.91)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(right=0.95)
#plt.show()
plt.savefig("sphere_estimates.pdf", dpi=1200)
