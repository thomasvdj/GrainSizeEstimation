import numpy as np
import vorostereology as vs
import multiprocessing as mp
import pickle


sigma = 0.4
mu = -1*sigma*sigma/2


def generate_section(rng, n, i):
    target_volumes = rng.lognormal(mean=mu, sigma=sigma, size=n)
    target_volumes = target_volumes / np.sum(target_volumes)

    res, flag = vs.compute_centroidal_laguerre3d(target_volumes, periodic=True, rng=rng)

    if flag:
        coeffs = np.array([0.0, 0.0, 1.0])
        offset = np.array([0.5, 0.5, 0.5])
        offset[2] = rng.uniform()
        cross_section = res.compute_section(coeffs, offset)
        print("Finished section", i)
        return res, cross_section
    else:
        print("Failed section", i, "Trying again")
        return generate_section(rng, n, i)


def main():
    num_repitions = 100
    rngs = [np.random.default_rng(i) for i in range(num_repitions)]
    pool = mp.Pool(processes=3)

    n = 50000
    results = [pool.apply_async(generate_section, args=(rngs[j], n, j)) for j in range(num_repitions)]
    combined_result = [p.get() for p in results]

    f = open("lognormal_0.4_50k.pkl", "wb")
    pickle.dump(combined_result, f)
    f.close()


if __name__ == "__main__":
    main()

