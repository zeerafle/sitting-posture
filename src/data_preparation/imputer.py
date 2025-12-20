import pandas as pd
from sklearn.impute import SimpleImputer

class Imputer:
    def __init__(self):
        self.group_imp = {}
        self.overall_imp = None

    def fit_transform(self, df: pd.DataFrame, feature_cols):
        X = df.copy()
        # group‐wise means
        for cls, sub in X.groupby("class_name"):
            imp = SimpleImputer(strategy="mean")
            cols = feature_cols
            imp.fit(sub[cols])
            X.loc[sub.index, cols] = imp.transform(sub[cols])
            self.group_imp[cls] = (imp, cols)
        # overall for any leftover
        if X[feature_cols].isna().any().any():
            self.overall_imp = SimpleImputer(strategy="mean")
            X[feature_cols] = self.overall_imp.fit_transform(X[feature_cols])
        return X

    def transform(self, df: pd.DataFrame, feature_cols):
        X = df.copy()
        for cls, (imp, cols) in self.group_imp.items():
            idx = X["class_name"]==cls
            X.loc[idx, cols] = imp.transform(X.loc[idx, cols])
        if self.overall_imp:
            X[feature_cols] = self.overall_imp.transform(X[feature_cols])
        return X
