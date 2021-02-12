#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:10:04 2021

@author: ostadgeorge
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn import manifold

data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
pixel_vals, targets = data
pixel_vals = pixel_vals.to_numpy()
targets = targets.astype(int)
n, d = pixel_vals.shape

random_image = pixel_vals[np.random.randint(0, n), :].reshape(28, 28)
plt.imshow(random_image, cmap='gray')

tsne = manifold.TSNE(n_components=2)
transformer = tsne.fit_transform(pixel_vals[:500, :])

tsne_df = pd.DataFrame(np.column_stack((transformer, targets[:500])), columns=["x", "y", "target"])
tsne_df.loc[:, "target"] = tsne_df["target"].astype(int)

tsne_df.head()

grid = sns.FacetGrid(tsne_df, hue="target", size=8)
grid.map(plt.scatter, "x", "y").add_legend()
