"""
Feature transformers for Feluda.

This module provides transformers for machine learning features.
"""

import abc
import enum
import json
import logging
import os
import pickle
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class FeatureTransformer(abc.ABC):
    """
    Base class for feature transformers.
    
    This class defines the interface for feature transformers.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a feature transformer.
        
        Args:
            name: The transformer name.
            config: The transformer configuration.
        """
        self.name = name
        self.config = config
        self.model = None
    
    @abc.abstractmethod
    def fit(self, data: Any) -> None:
        """
        Fit the transformer to the data.
        
        Args:
            data: The data to fit the transformer to.
        """
        pass
    
    @abc.abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transform the data.
        
        Args:
            data: The data to transform.
            
        Returns:
            The transformed data.
        """
        pass
    
    @abc.abstractmethod
    def inverse_transform(self, data: Any) -> Any:
        """
        Inverse transform the data.
        
        Args:
            data: The data to inverse transform.
            
        Returns:
            The inverse transformed data.
        """
        pass
    
    def fit_transform(self, data: Any) -> Any:
        """
        Fit the transformer to the data and transform it.
        
        Args:
            data: The data to fit the transformer to and transform.
            
        Returns:
            The transformed data.
        """
        self.fit(data)
        return self.transform(data)
    
    def save(self, path: str) -> None:
        """
        Save the transformer to a file.
        
        Args:
            path: The path to save the transformer to.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the transformer
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "FeatureTransformer":
        """
        Load a transformer from a file.
        
        Args:
            path: The path to load the transformer from.
            
        Returns:
            The loaded transformer.
        """
        # Load the transformer
        with open(path, "rb") as f:
            return pickle.load(f)


class StandardScalerTransformer(FeatureTransformer):
    """
    Standard scaler transformer.
    
    This class implements a transformer that standardizes features by removing the mean and scaling to unit variance.
    """
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> None:
        """
        Fit the transformer to the data.
        
        Args:
            data: The data to fit the transformer to.
        """
        # Create the model
        self.model = StandardScaler()
        
        # Fit the model
        self.model.fit(data.reshape(-1, 1) if isinstance(data, np.ndarray) and data.ndim == 1 else data)
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Transform the data.
        
        Args:
            data: The data to transform.
            
        Returns:
            The transformed data.
        """
        if self.model is None:
            raise ValueError("Transformer not fitted")
        
        # Transform the data
        return self.model.transform(data.reshape(-1, 1) if isinstance(data, np.ndarray) and data.ndim == 1 else data)
    
    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Inverse transform the data.
        
        Args:
            data: The data to inverse transform.
            
        Returns:
            The inverse transformed data.
        """
        if self.model is None:
            raise ValueError("Transformer not fitted")
        
        # Inverse transform the data
        return self.model.inverse_transform(data.reshape(-1, 1) if isinstance(data, np.ndarray) and data.ndim == 1 else data)


class MinMaxScalerTransformer(FeatureTransformer):
    """
    Min-max scaler transformer.
    
    This class implements a transformer that scales features to a given range.
    """
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> None:
        """
        Fit the transformer to the data.
        
        Args:
            data: The data to fit the transformer to.
        """
        # Get the feature range
        feature_range = self.config.get("feature_range", (0, 1))
        
        # Create the model
        self.model = MinMaxScaler(feature_range=feature_range)
        
        # Fit the model
        self.model.fit(data.reshape(-1, 1) if isinstance(data, np.ndarray) and data.ndim == 1 else data)
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Transform the data.
        
        Args:
            data: The data to transform.
            
        Returns:
            The transformed data.
        """
        if self.model is None:
            raise ValueError("Transformer not fitted")
        
        # Transform the data
        return self.model.transform(data.reshape(-1, 1) if isinstance(data, np.ndarray) and data.ndim == 1 else data)
    
    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Inverse transform the data.
        
        Args:
            data: The data to inverse transform.
            
        Returns:
            The inverse transformed data.
        """
        if self.model is None:
            raise ValueError("Transformer not fitted")
        
        # Inverse transform the data
        return self.model.inverse_transform(data.reshape(-1, 1) if isinstance(data, np.ndarray) and data.ndim == 1 else data)


class OneHotEncoderTransformer(FeatureTransformer):
    """
    One-hot encoder transformer.
    
    This class implements a transformer that encodes categorical features as a one-hot numeric array.
    """
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> None:
        """
        Fit the transformer to the data.
        
        Args:
            data: The data to fit the transformer to.
        """
        # Get the configuration
        sparse = self.config.get("sparse", False)
        
        # Create the model
        self.model = OneHotEncoder(sparse=sparse)
        
        # Fit the model
        self.model.fit(data.reshape(-1, 1) if isinstance(data, np.ndarray) and data.ndim == 1 else data)
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Transform the data.
        
        Args:
            data: The data to transform.
            
        Returns:
            The transformed data.
        """
        if self.model is None:
            raise ValueError("Transformer not fitted")
        
        # Transform the data
        return self.model.transform(data.reshape(-1, 1) if isinstance(data, np.ndarray) and data.ndim == 1 else data)
    
    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Inverse transform the data.
        
        Args:
            data: The data to inverse transform.
            
        Returns:
            The inverse transformed data.
        """
        if self.model is None:
            raise ValueError("Transformer not fitted")
        
        # Inverse transform the data
        return self.model.inverse_transform(data)


class LabelEncoderTransformer(FeatureTransformer):
    """
    Label encoder transformer.
    
    This class implements a transformer that encodes labels with values between 0 and n_classes-1.
    """
    
    def fit(self, data: Union[np.ndarray, pd.Series]) -> None:
        """
        Fit the transformer to the data.
        
        Args:
            data: The data to fit the transformer to.
        """
        # Create the model
        self.model = LabelEncoder()
        
        # Fit the model
        self.model.fit(data)
    
    def transform(self, data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Transform the data.
        
        Args:
            data: The data to transform.
            
        Returns:
            The transformed data.
        """
        if self.model is None:
            raise ValueError("Transformer not fitted")
        
        # Transform the data
        return self.model.transform(data)
    
    def inverse_transform(self, data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Inverse transform the data.
        
        Args:
            data: The data to inverse transform.
            
        Returns:
            The inverse transformed data.
        """
        if self.model is None:
            raise ValueError("Transformer not fitted")
        
        # Inverse transform the data
        return self.model.inverse_transform(data)


# Dictionary of transformer classes
_transformer_classes = {
    "standard_scaler": StandardScalerTransformer,
    "minmax_scaler": MinMaxScalerTransformer,
    "onehot_encoder": OneHotEncoderTransformer,
    "label_encoder": LabelEncoderTransformer,
}


def get_transformer(name: str, type: str, config: Optional[Dict[str, Any]] = None) -> FeatureTransformer:
    """
    Get a transformer.
    
    Args:
        name: The transformer name.
        type: The transformer type.
        config: The transformer configuration.
        
    Returns:
        The transformer.
    """
    # Get the transformer class
    transformer_class = _transformer_classes.get(type)
    
    if not transformer_class:
        raise ValueError(f"Unknown transformer type: {type}")
    
    # Create the transformer
    return transformer_class(name, config or {})
