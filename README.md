# GrainSizeEstimation
A collection of Python scripts used for simulating microstructures and estimating grain size distributions. These scripts were used in the paper: "Estimation of 3D grain size distributions from 2D sections in real and simulated microstructures", by: T. van der Jagt, M. Vittorietti, K. Sedighiani, C. Bos, G. Jongbloed. Hence, these scripts may be used to reproduce the results in this paper.

# Dependencies
These scripts rely on particular on the Python packages [pysizeunfolder](https://github.com/thomasvdj/pysizeunfolder) and [vorostereology](https://github.com/thomasvdj/vorostereology).

# Preparation
As a first step, you should run the script: generate_reference_samples.py. Because this file generates a set of files used by almost all of the other scripts in this repository.
Once finished, you may run visualize_gs_all_shapes.py to plot the results.

# Simulating Laguerre-Voronoi diagrams and estimating grain size distributions
It is recommended to run the following scripts in the following order:
- generate_lognormal.py Generates 100 Laguerre-Voronoi diagrams and cross sections
- estimate_volume_dist.py Computes 100 estimates (for each shape) of grain size distributions based on the cross sections
- visualize_lognormal.py Visualize the 100 (biased) grain volume estimates for a given shape
- compute_errors.py Compute supremum errors for all estimates of all shapes
- compute_l1_errors.py Compute L1 errors for all estimates of all shapes
- summary_table.py Summarize the computed errors in a table

# Other scripts
- sphericity_simulation.py Keeps generating Poisson-Voronoi diagrams until 1 million cells have been simulated. For each cell the sphericity is computed.
- disector.py Generates a Poisson-Voronoi diagram, takes 100 pairs of parallel sections and uses the disector to estimate the mean grain volume.
- if_steel.py Estimates the grain size distribution for an IF-steel data set. (Data set not in repository)

