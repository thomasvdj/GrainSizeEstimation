import matplotlib.pyplot as plt
import vorostereology as vs
import pysizeunfolder as pu
import numpy as np
import pickle
rng = np.random.default_rng(1)
max_double = 1000000.0

octa_G_pts = pickle.load(open("octa_G_sample.pkl", "rb"))
cube_G_pts = pickle.load(open("cube_G_sample.pkl", "rb"))
dode_G_pts = pickle.load(open("dode_G_sample.pkl", "rb"))
kelvin_G_pts = pickle.load(open("kelvin_G_sample.pkl", "rb"))
tetra_G_pts = pickle.load(open("tetra_G_sample.pkl", "rb"))

low = 0.0
high = 100.0
domain = [[low, high], [low, high], [low, high]]
n = 78862
points = rng.uniform(size=(n, 3), low=low, high=high)
weights = np.zeros(n)
lag = vs.Laguerre3D(points, weights, domain, periodic=True)
volumes = np.sort(lag.get_volumes())

print("mean volume", np.mean(volumes))
coeffs = np.array([0.0, 0.0, 1.0])
offset = np.array([0.0, 0.0, 1.0])
section1 = lag.compute_section(coeffs, 50*offset)
areas = section1["areas"]
sqrt_sample = np.sqrt(np.sort(areas))
print("sample size", len(areas))

print("estimate 0 starting")
est_tetra = pu.estimate_size(areas, tetra_G_pts)
print("estimate 1 starting")
est_cube = pu.estimate_size(areas, cube_G_pts)
print("estimate 2 starting")
est_dode = pu.estimate_size(areas, dode_G_pts)
print("estimate 3 starting")
est_kelvin = pu.estimate_size(areas, kelvin_G_pts)
print("estimate 4 starting")
est_octa = pu.estimate_size(areas, octa_G_pts)
#est_sphere = pu.estimate_size(areas, None, sphere=True)
print("estimate 5 starting")

print("cube mean", np.dot(sqrt_sample**3, np.append(est_cube[1][0], np.diff(est_cube[1]))))
print("octa mean", np.dot(sqrt_sample**3, np.append(est_octa[1][0], np.diff(est_octa[1]))))
print("dode mean", np.dot(sqrt_sample**3, np.append(est_dode[1][0], np.diff(est_dode[1]))))
print("kelvin mean", np.dot(sqrt_sample**3, np.append(est_kelvin[1][0], np.diff(est_kelvin[1]))))
print("tetra mean", np.dot(sqrt_sample**3, np.append(est_tetra[1][0], np.diff(est_tetra[1]))))
#print("sphere mean", np.dot(sqrt_sample**3, np.append(est_sphere[1][0], np.diff(est_sphere[1]))))

plt.figure(figsize=(4.2, 3.0))
#plt.step(np.append(np.append(0, sqrt_sample**3), max_double), np.append(np.append(0, est_sphere), 1), where="post", label="Sphere")
plt.step(np.append(np.append(0, sqrt_sample**3), max_double), np.append(np.append(0, est_dode[1]), 1), where="post", label="Dodecahedron")
plt.step(np.append(np.append(0, sqrt_sample**3), max_double), np.append(np.append(0, est_kelvin[1]), 1), where="post", label="Kelvin cell")
plt.step(np.append(np.append(0, sqrt_sample**3), max_double), np.append(np.append(0, est_cube[1]), 1), where="post", label="Cube")
plt.step(np.append(np.append(0, sqrt_sample**3), max_double), np.append(np.append(0, est_octa[1]), 1), where="post", label="Octahedron")
plt.step(np.append(np.append(0, sqrt_sample**3), max_double), np.append(np.append(0, est_tetra[1]), 1), where="post", label="Tetrahedron")
plt.step(np.append(np.append(0, volumes), max_double), np.append(np.append(0, np.linspace(0, 1, len(volumes))), 1), where="post", label="3D Voronoi")
plt.title(r"Estimates of volume distribution function")
plt.xlabel(r"Volume")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.xlim([0, 35])
plt.subplots_adjust(bottom=0.16)
plt.subplots_adjust(top=0.91)
plt.subplots_adjust(left=0.13)
plt.subplots_adjust(right=0.97)
plt.savefig("additional_voronoi_estimates.pdf")


section_idx = 15
nv_estimates = []
fixed_dist = 100.0/101
for i in range(100):
    h1 = (0.5 + i)*fixed_dist
    h2 = (1.5 + i)*fixed_dist
    section1 = lag.compute_section(coeffs, h1*offset)
    section2 = lag.compute_section(coeffs, h2*offset)
    print(i)
    q1 = len(np.setdiff1d(section1["area_indices"], section2["area_indices"]))
    q2 = len(np.setdiff1d(section2["area_indices"], section1["area_indices"]))
    tot_area = np.sum(section1["areas"])
    nv = (q1 + q2)/(2*tot_area*fixed_dist)
    nv_estimates.append(nv)

print("N_v", np.mean(nv_estimates), "est mean volume", 1./np.mean(nv_estimates))

plt.figure(figsize=(3.0, 2.4))
plt.hist(1./np.array(nv_estimates), bins=30, ec='black', linewidth=0.2)
plt.axvline(x=12.68, color='red')
plt.title("Estimates of mean grain volume")
plt.xlabel("Volume")
plt.ylabel("Frequency")
plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(top=0.91)
plt.subplots_adjust(left=0.18)
plt.subplots_adjust(right=0.96)
plt.savefig("mean_volume_estimates.pdf")

