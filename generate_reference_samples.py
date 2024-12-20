import numpy as np
from pysizeunfolder import iur_3d_hull
from joblib import Parallel, delayed
from math import sqrt
import pickle
import os


n = 10000000
ss = np.random.SeedSequence(0)
num_cpus = os.cpu_count()
child_seeds = ss.spawn(num_cpus)
streams = [np.random.default_rng(s) for s in child_seeds]

block_size = n // num_cpus
remainder = n - num_cpus*block_size
sizes = [block_size]*num_cpus
sizes[-1] += remainder

points = np.array([[0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
                   [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]])
res = Parallel(n_jobs=num_cpus)(delayed(iur_3d_hull)(points, sizes[i], False, streams[i]) for i in range(num_cpus))
areas = np.concatenate(res)
f = open("cube_G_sample.pkl", "wb")
pickle.dump(areas, f)
f.close()

points = np.array([[0., 0., 1.0], [0., 0., -1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.],
                   [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
res = Parallel(n_jobs=num_cpus)(delayed(iur_3d_hull)(points, sizes[i], False, streams[i], True) for i in range(num_cpus))
areas = np.concatenate(res)
f = open("octa_G_sample.pkl", "wb")
pickle.dump(areas, f)
f.close()

phi = (1 + sqrt(5))*0.5
points = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, -1, 1],
                   [0, phi, 1./phi], [0, -phi, 1./phi], [0, phi, -1./phi], [0, -phi, -1./phi],
                   [1./phi, 0, phi], [-1./phi, 0, phi], [1./phi, 0, -phi], [-1./phi, 0, -phi],
                   [phi, 1./phi, 0], [-phi, 1./phi, 0], [phi, -1./phi, 0], [-phi, -1./phi, 0]])
res = Parallel(n_jobs=num_cpus)(delayed(iur_3d_hull)(points, sizes[i], False, streams[i], True) for i in range(num_cpus))
areas = np.concatenate(res)
f = open("dode_G_sample.pkl", "wb")
pickle.dump(areas, f)
f.close()

points = np.array([[0, 1, 2], [0, -1, 2], [0, 1, -2], [0, -1, -2],
                   [1, 0, 2], [-1, 0, 2], [1, 0, -2], [-1, 0, -2],
                   [1, 2, 0], [-1, 2, 0], [1, -2, 0], [-1, -2, 0],
                   [0, 2, 1], [0, -2, 1], [0, 2, -1], [0, -2, -1],
                   [2, 0, 1], [-2, 0, 1], [2, 0, -1], [-2, 0, -1],
                   [2, 1, 0], [-2, 1, 0], [2, -1, 0], [-2, -1, 0]], dtype=np.double)
res = Parallel(n_jobs=num_cpus)(delayed(iur_3d_hull)(points, sizes[i], False, streams[i], True) for i in range(num_cpus))
areas = np.concatenate(res)
f = open("kelvin_G_sample.pkl", "wb")
pickle.dump(areas, f)
f.close()

points = np.array([[1, 0, -1./sqrt(2)], [-1, 0, -1./sqrt(2)], [0, 1, 1./sqrt(2)], [0, -1, 1./sqrt(2)]], dtype=np.double)
res = Parallel(n_jobs=num_cpus)(delayed(iur_3d_hull)(points, sizes[i], False, streams[i], True) for i in range(num_cpus))
areas = np.concatenate(res)
f = open("tetra_G_sample.pkl", "wb")
pickle.dump(areas, f)
f.close()

