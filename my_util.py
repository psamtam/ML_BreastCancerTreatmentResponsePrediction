import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import odr
import seaborn as sns
from enum import Enum



def df_to_corr_matrix(df, sep=200, annot=True, size_factor=1):
  correlation_matrix = df.corr()

  mask = np.zeros_like(correlation_matrix)
  print(correlation_matrix.shape)
  mask[np.triu_indices_from(mask)] = True
  plt.figure(figsize = (size_factor*10,size_factor*8))
  cmap = sns.diverging_palette(260, 10, sep=sep, as_cmap=True)
  sns.heatmap(correlation_matrix, cmap = cmap, mask=mask, vmin=-1, vmax=1, annot=annot)
  plt.show()

def remove_outliers(X, y, selected_features):
  class ReplaceMethod(Enum):
      SMALLEST = 0
      LARGEST = 1

  replace_list = [
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_firstorder_10Percentile", "outcome": 0, "threshold": 1.678166},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_firstorder_10Percentile", "outcome": 1, "threshold": -0.617352},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_ngtdm_Busyness", "outcome": 1, "threshold": 838.677442},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_gldm_DependenceEntropy", "outcome": 1, "threshold": 2.478963},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_firstorder_Skewness", "outcome": 1, "threshold": 0.545105},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_firstorder_Skewness", "outcome": 0, "threshold": 0.767485},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_firstorder_Skewness", "outcome": 1, "threshold": -0.995207},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_glrlm_ShortRunHighGrayLevelEmphasis", "outcome": 1, "threshold": 0.363247},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glrlm_ShortRunHighGrayLevelEmphasis", "outcome": 0, "threshold": 0.838612},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_gldm_SmallDependenceEmphasis", "outcome": 0, "threshold": 0.011321},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_shape_MajorAxisLength", "outcome": 0, "threshold": 162.863366},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glrlm_LongRunLowGrayLevelEmphasis", "outcome": 1, "threshold": 39.241005},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glrlm_LongRunLowGrayLevelEmphasis", "outcome": 0, "threshold": 90.572934},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_firstorder_Minimum", "outcome": 0, "threshold": -2.346176},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_firstorder_Minimum", "outcome": 1, "threshold": -2.052288},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_shape_Maximum2DDiameterRow", "outcome": 1, "threshold": 77.252832},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_shape_Maximum2DDiameterRow", "outcome": 0, "threshold": 124.193398},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_shape_SurfaceVolumeRatio", "outcome": 0, "threshold": 0.723904},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_shape_SurfaceVolumeRatio", "outcome": 1, "threshold": 0.772898},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_shape_SurfaceVolumeRatio", "outcome": 1, "threshold": 0.215198},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_shape_LeastAxisLength", "outcome": 0, "threshold": 52.226330},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_shape_LeastAxisLength", "outcome": 1, "threshold": 41.589009},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_shape_LeastAxisLength", "outcome": 1, "threshold": 8.531971},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_glcm_Autocorrelation", "outcome": 0, "threshold": 3.040814},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_glcm_Autocorrelation", "outcome": 1, "threshold": 3.297653},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glszm_SizeZoneNonUniformityNormalized", "outcome": 0, "threshold": 0.653333},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glszm_SizeZoneNonUniformityNormalized", "outcome": 1, "threshold": 0.500000},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glszm_SmallAreaEmphasis", "outcome": 1, "threshold": 0.643301},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_shape_Elongation", "outcome": 0, "threshold": 0.299156},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_shape_Elongation", "outcome": 1, "threshold": 0.350000},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_firstorder_Kurtosis", "outcome": 1, "threshold": 4.760064},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_firstorder_Kurtosis", "outcome": 0, "threshold": 5.157534},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glszm_GrayLevelNonUniformity", "outcome": 1, "threshold": 204.009709},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glszm_GrayLevelNonUniformity", "outcome": 0, "threshold": 290.006849},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_glcm_Imc1", "outcome": 1, "threshold": -0.399542},
      {"replace_method": ReplaceMethod.SMALLEST, "feature": "original_glcm_Imc1", "outcome": 0, "threshold": -0.451831},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_firstorder_90Percentile", "outcome": 0, "threshold": 4.883197},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_firstorder_90Percentile", "outcome": 1, "threshold": 4.087205},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glcm_Correlation", "outcome": 1, "threshold": 0.569794},
      {"replace_method": ReplaceMethod.LARGEST, "feature": "original_glcm_Correlation", "outcome": 0, "threshold": 0.704593},
  ]

  data = pd.concat([y, X], axis=1)

  for task in replace_list:
      outcome = task["outcome"]
      feature = task["feature"]
      threshold = task["threshold"]
      num_of_replace = 0
      if feature not in X.columns:
          continue
      if task["replace_method"] == ReplaceMethod.LARGEST:
          num_of_replace = len(data.loc[(data["pCR (outcome)"] == outcome) & (data[feature] > threshold), feature])
          data.loc[(data["pCR (outcome)"] == outcome) & (data[feature] > threshold), feature] = threshold
      elif task["replace_method"] == ReplaceMethod.SMALLEST:
          num_of_replace = len(data.loc[(data["pCR (outcome)"] == outcome) & (data[feature] < threshold), feature])
          data.loc[(data["pCR (outcome)"] == outcome) & (data[feature] < threshold), feature] = threshold
      print(f"Replaced {num_of_replace} records in {feature}[{outcome}] to {threshold}")

  X = data[selected_features]
  y = data["pCR (outcome)"]

  return X, y