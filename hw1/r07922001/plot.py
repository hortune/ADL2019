import seaborn
import numpy as np
utterance = "<BOS> anyone knows about a good file sharing webserver program ? <EOS>".split()

weight = np.load("tmp.npy")

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=weight,
                         columns=utterance,
                         index = utterance)

# Compute the correlation matrix
corr = d.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.savefig('123.png')


