import numpy as np
import pickle
import pandas as pd


shapes = ["sphere", "dode", "kelvin", "octa", "cube", "tetra"]
shape_names = ["sphere", "dodecahedron", "Kelvin cell", "octahedron", "cube", "tetrahedron"]
error_hb_column = []
ci_hb_column1 = []
ci_hb_column2 = []
error_h_column = []
ci_h_column1 = []
ci_h_column2 = []
padding_column = []

for shape in shapes:
    print("Shape:", shape)

    errors = pickle.load(open(shape + "_hb_l1_errors.pkl", "rb"))
    errors2 = pickle.load(open(shape + "_h_l1_errors.pkl", "rb"))

    error_hb_column.append(np.mean(errors))
    ci_hb_column1.append(np.quantile(errors, 0.025))
    ci_hb_column2.append(np.quantile(errors, 0.975))

    error_h_column.append(np.mean(errors2))
    ci_h_column1.append(np.quantile(errors2, 0.025))
    ci_h_column2.append(np.quantile(errors2, 0.975))
    padding_column.append(" ")

data = {"shape": shape_names, "mean error": error_h_column, "lower": ci_h_column1,
        "upper": ci_h_column2, "mean error2": error_hb_column, "lower2": ci_hb_column1, "upper2": ci_hb_column2,
        "padding": padding_column}

df = pd.DataFrame(data)
names = ["shape", "mean error", "lower", "upper", "mean error", "lower", "upper", " "]
print(df.to_latex(header=names, index=False, escape=False, float_format="%.3e"))

