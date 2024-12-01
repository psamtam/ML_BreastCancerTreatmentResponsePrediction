import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def df_to_corr_matrix(df, sep=200, annot=True, size_factor=1):
  correlation_matrix = df.corr()

  mask = np.zeros_like(correlation_matrix)
  print(correlation_matrix.shape)
  mask[np.triu_indices_from(mask)] = True
  plt.figure(figsize = (size_factor*10,size_factor*8))
  cmap = sns.diverging_palette(260, 10, sep=sep, as_cmap=True)
  sns.heatmap(correlation_matrix, cmap = cmap, mask=mask, vmin=-1, vmax=1, annot=annot)
  plt.show()