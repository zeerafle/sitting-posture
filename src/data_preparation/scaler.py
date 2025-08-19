import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray):
        # add tiny noise to zero‐std columns
        stds = X.std(axis=0)
        zeros = np.where(stds == 0)[0]
        for i in zeros:
            X[:,i] += np.random.normal(0,1e-3,size=X.shape[0])
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray):
        return self.scaler.transform(X)
