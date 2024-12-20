import pickle
import pysizeunfolder as pu
import numpy as np


data = pickle.load(open("lognormal_0.4_50k.pkl", "rb"))
octa_G_pts = pickle.load(open("octa_G_sample.pkl", "rb"))
cube_G_pts = pickle.load(open("cube_G_sample.pkl", "rb"))
dode_G_pts = pickle.load(open("dode_G_sample.pkl", "rb"))
kelvin_G_pts = pickle.load(open("kelvin_G_sample.pkl", "rb"))
tetra_G_pts = pickle.load(open("tetra_G_sample.pkl", "rb"))
print("Finished loading data.")


tetra_estimates = []
tetra_estimateshb = []
cube_estimates = []
cube_estimateshb = []
octa_estimates = []
octa_estimateshb = []
dode_estimates = []
dode_estimateshb = []
kelvin_estimates = []
kelvin_estimateshb = []
sphere_estimates = []
sphere_estimateshb = []

for idx, el in enumerate(data):
    laguerre, section = el
    sqrt_sample = np.sqrt(np.sort(section["areas"]))

    _, est_cube_hb = pu.estimate_size(section["areas"], cube_G_pts, debias=False)
    est_cube = pu.de_bias(sqrt_sample, est_cube_hb, cube_G_pts)

    _, est_octa_hb = pu.estimate_size(section["areas"], octa_G_pts, debias=False)
    est_octa = pu.de_bias(sqrt_sample, est_octa_hb, octa_G_pts)

    _, est_dode_hb = pu.estimate_size(section["areas"], dode_G_pts, debias=False)
    est_dode = pu.de_bias(sqrt_sample, est_dode_hb, dode_G_pts)

    _, est_kelvin_hb = pu.estimate_size(section["areas"], kelvin_G_pts, debias=False)
    est_kelvin = pu.de_bias(sqrt_sample, est_kelvin_hb, kelvin_G_pts)

    _, est_tetra_hb = pu.estimate_size(section["areas"], tetra_G_pts, debias=False)
    est_tetra = pu.estimate_size(sqrt_sample, est_tetra_hb, tetra_G_pts)

    _, est_sphere = pu.estimate_size(section["areas"], None, debias=True, sphere=True)
    _, est_sphere_hb = pu.estimate_size(section["areas"], None, debias=False, sphere=True)

    cube_estimates.append((sqrt_sample, est_cube))
    cube_estimateshb.append((sqrt_sample, est_cube_hb))
    tetra_estimates.append((sqrt_sample, est_tetra))
    tetra_estimateshb.append((sqrt_sample, est_tetra_hb))
    octa_estimates.append((sqrt_sample, est_octa))
    octa_estimateshb.append((sqrt_sample, est_octa_hb))
    dode_estimates.append((sqrt_sample, est_dode))
    dode_estimateshb.append((sqrt_sample, est_dode_hb))
    kelvin_estimates.append((sqrt_sample, est_kelvin))
    kelvin_estimateshb.append((sqrt_sample, est_kelvin_hb))
    sphere_estimates.append((sqrt_sample, est_sphere))
    sphere_estimateshb.append((sqrt_sample, est_sphere_hb))
    print("Finished iteration", idx)

f = open("lognormal_tetra_estimates.pkl", "wb")
pickle.dump(tetra_estimates, f)
f.close()

f = open("lognormal_tetra_estimateshb.pkl", "wb")
pickle.dump(tetra_estimateshb, f)
f.close()

f = open("lognormal_cube_estimateshb.pkl", "wb")
pickle.dump(cube_estimates, f)
f.close()

f = open("lognormal_cube_estimates.pkl", "wb")
pickle.dump(cube_estimateshb, f)
f.close()

f = open("lognormal_octa_estimateshb.pkl", "wb")
pickle.dump(octa_estimateshb, f)
f.close()

f = open("lognormal_octa_estimates.pkl", "wb")
pickle.dump(octa_estimates, f)
f.close()

f = open("lognormal_dode_estimateshb.pkl", "wb")
pickle.dump(dode_estimateshb, f)
f.close()

f = open("lognormal_dode_estimates.pkl", "wb")
pickle.dump(dode_estimates, f)
f.close()

f = open("lognormal_kelvin_estimates.pkl", "wb")
pickle.dump(kelvin_estimates, f)
f.close()

f = open("lognormal_kelvin_estimateshb.pkl", "wb")
pickle.dump(kelvin_estimateshb, f)
f.close()

f = open("lognormal_sphere_estimates.pkl", "wb")
pickle.dump(sphere_estimates, f)
f.close()

f = open("lognormal_sphere_estimateshb.pkl", "wb")
pickle.dump(sphere_estimateshb, f)
f.close()
