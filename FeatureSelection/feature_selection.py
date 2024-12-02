import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath('../'))  # Adjust the path as needed

from my_util import df_to_corr_matrix

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif


from joblib import Parallel, delayed

from pickle import dump , load


TARGET_NUM_OF_FEATURES = 35

# Read data
training_file = "../TrainDataset2024.xls"

data = pd.read_excel(training_file)
data.drop(["ID", "RelapseFreeSurvival (outcome)"], axis=1, inplace=True)
data_no_na = data.replace(999, np.nan)
data_no_na.dropna(ignore_index=True, inplace=True)
X = data_no_na.drop('pCR (outcome)', axis=1)
y = data_no_na['pCR (outcome)']


# Drop highly correlated features
CORR_THRESHOLD = 0.9
# Create a correlation matrix
correlation_matrix = X.corr()

highly_correlated_features = set()

for i in range(len(correlation_matrix.columns)):
  for j in range(i):
    if abs(correlation_matrix.iloc[i, j]) > CORR_THRESHOLD:
        highly_correlated_features.add(correlation_matrix.columns[i])

X_no_highly_correlated = X.drop(columns=highly_correlated_features)

scaler = StandardScaler()
Xs = scaler.fit_transform(X_no_highly_correlated)
Xs = pd.DataFrame(Xs, columns=X_no_highly_correlated.columns)


def process_k_best(K, i):
    k_best = SelectKBest(score_func=mutual_info_classif, k=K)
    Xs_k_best = k_best.fit_transform(Xs, y)
    return k_best.get_feature_names_out()


# find features
features = {}

# Run in parallel
for K in range(1, TARGET_NUM_OF_FEATURES + 5):
    best = {}
    results = Parallel(n_jobs=-1)(delayed(process_k_best)(K, i) for i in range(K + 5))
    
    for feature_list in results:
        for feature in feature_list:
            if feature in best:
                best[feature] += 1
            else:
                best[feature] = 1

    sorted_best = dict(sorted(best.items(), key=lambda item: item[1], reverse=True))
    
    # Update features based on the counts
    for key in best:
        if best[key] > (K - 2):
            features[key] = features.get(key, 0) + 1


sorted_features = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
feature_names = list(sorted_features.keys())

num_max = sum(np.array(list(sorted_best.values())) == max(sorted_best.values()))

num_of_features = max(num_max, TARGET_NUM_OF_FEATURES)

important_features = ["Gene", "ER", "HER2"]

selected_features = list(set(important_features + feature_names[:num_of_features]))

num_of_features = len(selected_features)

print(f"Best {num_of_features} features are: ")
print(selected_features)

with open(f"pkl/{num_of_features}_selected_features.pkl", "wb") as file:
    dump(selected_features, file)