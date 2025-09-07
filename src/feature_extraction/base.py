from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class FeatureExtractor(ABC):
    """Base class for all feature extractors."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract(self, landmarks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from landmark data.

        Args:
            landmarks_df: DataFrame with landmark coordinates

        Returns:
            DataFrame with extracted features
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        raise NotImplementedError


class FeatureExtractionPipeline:
    """Pipeline for applying multiple feature extractors."""

    def __init__(self, extractors: List[FeatureExtractor]):
        self.extractors = extractors

    def extract_all(self, landmarks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature extractors to the landmark data.

        Notes:
            - This implementation expects and returns pandas.DataFrame.
            - Feature DataFrames returned by extractors must have the same number
              of rows as `landmarks_df`.
        """
        if not isinstance(landmarks_df, pd.DataFrame):
            raise TypeError(
                "FeatureExtractionPipeline expects a pandas.DataFrame. "
                "Please convert your input to pandas before calling extract_all."
            )

        # Work on a copy to avoid mutating user input
        result_df = landmarks_df.copy(deep=True)

        for extractor in self.extractors:
            print(f"Extracting {extractor.name} features...")
            features_df = extractor.extract(landmarks_df)

            if not isinstance(features_df, pd.DataFrame):
                raise TypeError(
                    f"Extractor '{extractor.name}' must return a pandas.DataFrame, "
                    f"got {type(features_df)}"
                )

            if len(features_df) != len(result_df):
                raise ValueError(
                    f"Extractor '{extractor.name}' returned {len(features_df)} rows, "
                    f"but input has {len(result_df)} rows."
                )

            # Align indices to avoid accidental misalignment during concat
            features_df = features_df.set_index(result_df.index)
            result_df = pd.concat([result_df, features_df], axis=1)

        return result_df

    def get_all_feature_names(self) -> List[str]:
        """Get all feature names from all extractors."""
        all_features = []
        for extractor in self.extractors:
            all_features.extend(extractor.get_feature_names())
        return all_features
