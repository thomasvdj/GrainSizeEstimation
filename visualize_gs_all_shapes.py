import pickle
import numpy as np
from math import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import pysizeunfolder as pu
mpl.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Sans serif']})
mpl.rc('text', usetex=True)


tetra_sqrt_areas = np.sqrt(pickle.load(open("tetra_G_sample.pkl", "rb")))
tetra_x, tetra_y = pu.approx_area_density(tetra_sqrt_areas, sqrt_data=True)

cube_sqrt_areas = np.sqrt(pickle.load(open("cube_G_sample.pkl", "rb")))
cube_x, cube_y = pu.approx_area_density(cube_sqrt_areas, sqrt_data=True)

dode_sqrt_areas = np.sqrt(pickle.load(open("dode_G_sample.pkl", "rb")))
dode_x, dode_y = pu.approx_area_density(dode_sqrt_areas, sqrt_data=True)

kelvin_sqrt_areas = np.sqrt(pickle.load(open("kelvin_G_sample.pkl", "rb")))
kelvin_x, kelvin_y = pu.approx_area_density(kelvin_sqrt_areas, sqrt_data=True)

octa_sqrt_areas = np.sqrt(pickle.load(open("octa_G_sample.pkl", "rb")))
octa_x, octa_y = pu.approx_area_density(octa_sqrt_areas, sqrt_data=True)


def gs_density(x, cs, maximum):
    y = cs(x)
    y[y < 0] = 0.0
    y[x > maximum] = 0.0
    y[x < 0] = 0.0
    return y


def gs_sphere(x):
    c = (9*pi/16)**(1./3)
    y = np.zeros(len(x))
    indices = x*x < c
    y[indices] = x[indices]/(c*np.sqrt(1-x[indices]*x[indices]/c))
    return y


sphere_x = np.linspace(0, (9*pi/16)**(1./6)-0.001, 10000)
sphere_y = gs_sphere(sphere_x)

plt.figure(figsize=(3.4, 2.4))
plt.plot(sphere_x, sphere_y, label="Sphere")
plt.plot(dode_x, dode_y, label="Dodecahedron")
plt.plot(kelvin_x, kelvin_y, label="Kelvin cell")
plt.plot(cube_x, cube_y, label="Cube")
plt.plot(octa_x, octa_y, label="Octahedron")
plt.plot(tetra_x, tetra_y, label="Tetrahedron")
plt.title(r"The function $g_K^S$ for various shapes")
plt.xlabel(r"Square root section area")
plt.ylabel("Density")
plt.legend(loc="upper left")
plt.xlim([0, 1.4])
plt.ylim([0, 8])
plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(left=0.12)
plt.subplots_adjust(right=0.97)
plt.show()

plt.figure(figsize=(3., 2.4))
plt.plot(sphere_x, sphere_y, label="Sphere")
plt.plot(dode_x, dode_y, label="Dodecahedron")
plt.plot(kelvin_x, kelvin_y, label="Kelvin cell")
plt.title(r"The function $g_K^S$ for various shapes")
plt.xlabel(r"Square root section area")
plt.ylabel("Density")
plt.legend(loc="upper left")
plt.xlim([0, 1.4])
plt.ylim([0, 8])
plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(left=0.13)
plt.subplots_adjust(right=0.97)
plt.savefig("gks_shapes1.pdf")
#plt.show(block=False)

plt.figure(figsize=(3., 2.4))
plt.plot(cube_x, cube_y, label="Cube")
plt.plot(octa_x, octa_y, label="Octahedron")
plt.plot(tetra_x, tetra_y, label="Tetrahedron")
plt.title(r"The function $g_K^S$ for various shapes")
plt.xlabel(r"Square root section area")
plt.ylabel("Density")
plt.legend(loc="upper left")
plt.xlim([0, 1.4])
plt.ylim([0, 8])
plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(left=0.13)
plt.subplots_adjust(right=0.97)
plt.savefig("gks_shapes2.pdf")
#plt.show()

