# %%
import numpy as np
import pandas as pd

# %%
path = "./data/processed/dataset_features.npz"

# %%
with np.load(path) as dados:
    X = dados['X']
    y = dados['y']

print(y)