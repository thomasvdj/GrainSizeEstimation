import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import time
import pysizeunfolder as pu
mpl.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Sans serif']})
mpl.rc('text', usetex=True)
rng = np.random.default_rng(1)
max_double = np.finfo(np.double).max*0.1


# //////////////////////////////////// Read data from files /////////////////////////////////////////////////

col_names = ["identity", "phi1", "PHI", "phi2", "average_x", "average_y",
             "IQ", "CI", "average_fit", "phase", "type", "area", "diameter",
             "ar_ellipse", "major_axis_ellipse", "minor_axis_ellipse", "orientation_major_axis"]

data_types = {col_name: np.float64 for col_name in col_names}
data_types["identity"] = np.int64
data_types["phase"] = np.int64
data_types["type"] = np.int64

data_2d = pd.read_csv("DC06 ND_oppervlak grain_file-2D.txt", skiprows=17, delim_whitespace=True,
                      names=col_names, dtype=data_types)
data_2d.drop(data_2d[data_2d.type == 1].index, inplace=True)  # drop boundary grains
data_3d = pd.read_csv("DC06_GrainStats-3Ddata.txt", skiprows=1, delim_whitespace=True)

# There a lot of grains with the exact same measured area, we add a very tiny amount of noise
observed_areas = data_2d["area"].to_numpy() + np.abs(rng.normal(loc=0, scale=0.0001, size=len(data_2d["area"])))
# check that indeed very little noise was added
print(np.max(np.abs(observed_areas - data_2d["area"].to_numpy())))
volumes = np.sort(data_3d["VolumeSizes"].to_numpy())

print("Number of non-boundary grains:", len(np.unique(observed_areas)))
observed_areas = np.sort(observed_areas)

plt.figure(figsize=(4.0, 3.0))
plt.hist(observed_areas, bins=60, ec='black', linewidth=0.2)
plt.xlabel(r"Area $[\mu m^2]$")
plt.ylabel("Frequency")
plt.title("Observed section areas 2D EBSD")
plt.xlim([0, 0.6*np.max(observed_areas)])
plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(top=0.89)
plt.subplots_adjust(left=0.15)
plt.subplots_adjust(right=0.95)
plt.savefig("observed_areas.pdf")

# //////////////////////////////////

tetra_G_pts = pickle.load(open("tetra_G_sample.pkl", "rb"))
cube_G_pts = pickle.load(open("cube_G_sample.pkl", "rb"))
dode_G_pts = pickle.load(open("dode_G_sample.pkl", "rb"))
octa_G_pts = pickle.load(open("octa_G_sample.pkl", "rb"))
kelvin_G_pts = pickle.load(open("kelvin_G_sample.pkl", "rb"))
print("Finished loading data.")


start = time.time()
x_pts_tetra, est_tetra = pu.estimate_size(observed_areas, tetra_G_pts)
x_pts_cube, est_cube = pu.estimate_size(observed_areas, cube_G_pts)
x_pts_octa, est_octa = pu.estimate_size(observed_areas, octa_G_pts)
x_pts_dode, est_dode = pu.estimate_size(observed_areas, dode_G_pts)
x_pts_kelvin, est_kelvin = pu.estimate_size(observed_areas, kelvin_G_pts)
x_pts_sphere, est_sphere = pu.estimate_size(observed_areas, None, sphere=True)
end = time.time()
print("Time spent computing three estimates: ", end-start)

plt.figure(figsize=(3.4, 2.4))
plt.step(np.append(np.append(0, x_pts_sphere**3), max_double), np.append(np.append(0, est_sphere), 1), where="post", label="Sphere")
plt.step(np.append(np.append(0, x_pts_dode**3), max_double), np.append(np.append(0, est_dode), 1), where="post", label="Dodecahedron")
plt.step(np.append(np.append(0, x_pts_kelvin**3), max_double), np.append(np.append(0, est_kelvin), 1), where="post", label="Kelvin cell")
plt.step(np.append(np.append(0, x_pts_cube**3), max_double), np.append(np.append(0, est_cube), 1), where="post", label="Cube")
plt.step(np.append(np.append(0, x_pts_cube**3), max_double), np.append(np.append(0, est_octa), 1), where="post", label="Octahedron")
plt.step(np.append(np.append(0, x_pts_tetra**3), max_double), np.append(np.append(0, est_tetra), 1), where="post", label="Tetrahedron")
plt.step(np.append(np.append(0, volumes), max_double), np.append(np.append(0, np.linspace(0, 1, len(volumes))), 1), where="post", label="3D EBSD")
plt.title(r"Estimates of volume CDF")
plt.xlabel(r"Volume $[\mu m^3]$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.xlim([0, 0.3*np.max(volumes)])
plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(top=0.91)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(right=0.95)
plt.show()
#plt.savefig("if_steel.pdf")

