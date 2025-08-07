from abc import ABC, abstractmethod
import polars as pl
import numpy as np
from typing import Dict, Any, List


class FeatureExtractor(ABC):
    """Base class for all feature extractors."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def extract(self, landmarks_df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract features from landmark data.
        
        Args:
            landmarks_df: DataFrame with landmark coordinates
            
        Returns:
            DataFrame with extracted features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        pass


class FeatureExtractionPipeline:
    """Pipeline for applying multiple feature extractors."""
    
    def __init__(self, extractors: List[FeatureExtractor]):
        self.extractors = extractors
    
    def extract_all(self, landmarks_df: pl.DataFrame) -> pl.DataFrame:
        """Apply all feature extractors to the landmark data."""
        result_df = landmarks_df.clone()
        
        for extractor in self.extractors:
            print(f"Extracting {extractor.name} features...")
            features_df = extractor.extract(landmarks_df)
            result_df = result_df.hstack(features_df)
        
        return result_df
    
    def get_all_feature_names(self) -> List[str]:
        """Get all feature names from all extractors."""
        all_features = []
        for extractor in self.extractors:
            all_features.extend(extractor.get_feature_names())
        return all_features