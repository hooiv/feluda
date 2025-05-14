"""
Drift detection module for Feluda.

This module provides drift detection for data validation.
"""

import abc
import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from feluda.config import get_config
from feluda.monitoring.alerts import AlertLevel, get_alert_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class DriftType(str, enum.Enum):
    """Enum for drift types."""
    
    STATISTICAL = "statistical"
    DISTRIBUTION = "distribution"
    CONCEPT = "concept"
    FEATURE = "feature"
    PREDICTION = "prediction"


class DriftResult(BaseModel):
    """
    Drift result.
    
    This class represents the result of a drift detection.
    """
    
    id: str = Field(..., description="The drift result ID")
    detector_id: str = Field(..., description="The drift detector ID")
    drift_type: DriftType = Field(..., description="The drift type")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    p_value: Optional[float] = Field(None, description="The p-value")
    threshold: Optional[float] = Field(None, description="The threshold")
    metrics: Dict[str, float] = Field(default_factory=dict, description="The drift metrics")
    details: Dict[str, Any] = Field(default_factory=dict, description="The drift details")
    timestamp: float = Field(..., description="The drift detection timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the drift result to a dictionary.
        
        Returns:
            A dictionary representation of the drift result.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftResult":
        """
        Create a drift result from a dictionary.
        
        Args:
            data: The dictionary to create the drift result from.
            
        Returns:
            A drift result.
        """
        return cls(**data)


class DriftDetector(abc.ABC):
    """
    Base class for drift detectors.
    
    This class defines the interface for drift detectors.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        drift_type: DriftType,
        reference_data: Any,
        threshold: float = 0.05,
    ):
        """
        Initialize a drift detector.
        
        Args:
            id: The drift detector ID.
            name: The drift detector name.
            drift_type: The drift type.
            reference_data: The reference data.
            threshold: The drift threshold.
        """
        self.id = id
        self.name = name
        self.drift_type = drift_type
        self.reference_data = reference_data
        self.threshold = threshold
    
    @abc.abstractmethod
    def detect_drift(self, current_data: Any) -> DriftResult:
        """
        Detect drift.
        
        Args:
            current_data: The current data.
            
        Returns:
            A drift result.
        """
        pass


class StatisticalDriftDetector(DriftDetector):
    """
    Statistical drift detector.
    
    This class implements a drift detector that uses statistical tests.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        reference_data: Any,
        threshold: float = 0.05,
        test: str = "ks",
    ):
        """
        Initialize a statistical drift detector.
        
        Args:
            id: The drift detector ID.
            name: The drift detector name.
            reference_data: The reference data.
            threshold: The drift threshold.
            test: The statistical test to use.
        """
        super().__init__(id, name, DriftType.STATISTICAL, reference_data, threshold)
        self.test = test
    
    def detect_drift(self, current_data: Any) -> DriftResult:
        """
        Detect drift using a statistical test.
        
        Args:
            current_data: The current data.
            
        Returns:
            A drift result.
        """
        # Convert data to numpy arrays
        reference_data = np.array(self.reference_data).flatten()
        current_data = np.array(current_data).flatten()
        
        # Perform the statistical test
        if self.test == "ks":
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(reference_data, current_data)
        elif self.test == "t":
            # T-test
            statistic, p_value = stats.ttest_ind(reference_data, current_data)
        elif self.test == "chi2":
            # Chi-squared test
            statistic, p_value = stats.chisquare(current_data, reference_data)
        else:
            raise ValueError(f"Unsupported statistical test: {self.test}")
        
        # Check if drift is detected
        drift_detected = p_value < self.threshold
        
        # Create the drift result
        return DriftResult(
            id=str(uuid.uuid4()),
            detector_id=self.id,
            drift_type=self.drift_type,
            drift_detected=drift_detected,
            p_value=p_value,
            threshold=self.threshold,
            metrics={
                "statistic": statistic,
                "p_value": p_value,
            },
            details={
                "test": self.test,
                "reference_data_mean": float(np.mean(reference_data)),
                "reference_data_std": float(np.std(reference_data)),
                "current_data_mean": float(np.mean(current_data)),
                "current_data_std": float(np.std(current_data)),
            },
            timestamp=time.time(),
        )


class DistributionDriftDetector(DriftDetector):
    """
    Distribution drift detector.
    
    This class implements a drift detector that compares distributions.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        reference_data: Any,
        threshold: float = 0.05,
        bins: int = 10,
    ):
        """
        Initialize a distribution drift detector.
        
        Args:
            id: The drift detector ID.
            name: The drift detector name.
            reference_data: The reference data.
            threshold: The drift threshold.
            bins: The number of bins for histogram comparison.
        """
        super().__init__(id, name, DriftType.DISTRIBUTION, reference_data, threshold)
        self.bins = bins
    
    def detect_drift(self, current_data: Any) -> DriftResult:
        """
        Detect drift by comparing distributions.
        
        Args:
            current_data: The current data.
            
        Returns:
            A drift result.
        """
        # Convert data to numpy arrays
        reference_data = np.array(self.reference_data).flatten()
        current_data = np.array(current_data).flatten()
        
        # Compute histograms
        reference_hist, bin_edges = np.histogram(reference_data, bins=self.bins, density=True)
        current_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)
        
        # Compute Jensen-Shannon divergence
        js_divergence = stats.entropy(reference_hist, current_hist)
        
        # Check if drift is detected
        drift_detected = js_divergence > self.threshold
        
        # Create the drift result
        return DriftResult(
            id=str(uuid.uuid4()),
            detector_id=self.id,
            drift_type=self.drift_type,
            drift_detected=drift_detected,
            p_value=None,
            threshold=self.threshold,
            metrics={
                "js_divergence": js_divergence,
            },
            details={
                "bins": self.bins,
                "reference_hist": reference_hist.tolist(),
                "current_hist": current_hist.tolist(),
                "bin_edges": bin_edges.tolist(),
            },
            timestamp=time.time(),
        )


class FeatureDriftDetector(DriftDetector):
    """
    Feature drift detector.
    
    This class implements a drift detector that detects drift in features.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        reference_data: pd.DataFrame,
        threshold: float = 0.05,
        test: str = "ks",
    ):
        """
        Initialize a feature drift detector.
        
        Args:
            id: The drift detector ID.
            name: The drift detector name.
            reference_data: The reference data.
            threshold: The drift threshold.
            test: The statistical test to use.
        """
        super().__init__(id, name, DriftType.FEATURE, reference_data, threshold)
        self.test = test
    
    def detect_drift(self, current_data: pd.DataFrame) -> DriftResult:
        """
        Detect drift in features.
        
        Args:
            current_data: The current data.
            
        Returns:
            A drift result.
        """
        # Check if the data is a DataFrame
        if not isinstance(current_data, pd.DataFrame):
            raise ValueError("Current data must be a pandas DataFrame")
        
        # Check if the reference data is a DataFrame
        if not isinstance(self.reference_data, pd.DataFrame):
            raise ValueError("Reference data must be a pandas DataFrame")
        
        # Check if the columns match
        if set(current_data.columns) != set(self.reference_data.columns):
            raise ValueError("Current data columns do not match reference data columns")
        
        # Detect drift for each feature
        feature_results = {}
        
        for column in self.reference_data.columns:
            # Create a statistical drift detector for the feature
            detector = StatisticalDriftDetector(
                id=f"{self.id}_{column}",
                name=f"{self.name}_{column}",
                reference_data=self.reference_data[column],
                threshold=self.threshold,
                test=self.test,
            )
            
            # Detect drift
            result = detector.detect_drift(current_data[column])
            
            # Store the result
            feature_results[column] = result.to_dict()
        
        # Check if drift is detected in any feature
        drift_detected = any(result["drift_detected"] for result in feature_results.values())
        
        # Create the drift result
        return DriftResult(
            id=str(uuid.uuid4()),
            detector_id=self.id,
            drift_type=self.drift_type,
            drift_detected=drift_detected,
            p_value=None,
            threshold=self.threshold,
            metrics={
                "num_features": len(feature_results),
                "num_drifted_features": sum(1 for result in feature_results.values() if result["drift_detected"]),
            },
            details={
                "feature_results": feature_results,
            },
            timestamp=time.time(),
        )


class DriftManager:
    """
    Drift manager.
    
    This class is responsible for managing drift detectors and drift detection.
    """
    
    def __init__(self):
        """
        Initialize the drift manager.
        """
        self.detectors: Dict[str, DriftDetector] = {}
        self.results: Dict[str, DriftResult] = {}
        self.lock = threading.RLock()
    
    def register_detector(self, detector: DriftDetector) -> None:
        """
        Register a drift detector.
        
        Args:
            detector: The drift detector to register.
        """
        with self.lock:
            self.detectors[detector.id] = detector
    
    def get_detector(self, detector_id: str) -> Optional[DriftDetector]:
        """
        Get a drift detector by ID.
        
        Args:
            detector_id: The drift detector ID.
            
        Returns:
            The drift detector, or None if the detector is not found.
        """
        with self.lock:
            return self.detectors.get(detector_id)
    
    def get_detectors(self) -> Dict[str, DriftDetector]:
        """
        Get all drift detectors.
        
        Returns:
            A dictionary mapping drift detector IDs to drift detectors.
        """
        with self.lock:
            return self.detectors.copy()
    
    def detect_drift(self, detector_id: str, current_data: Any) -> DriftResult:
        """
        Detect drift.
        
        Args:
            detector_id: The drift detector ID.
            current_data: The current data.
            
        Returns:
            A drift result.
        """
        with self.lock:
            # Get the drift detector
            detector = self.get_detector(detector_id)
            
            if not detector:
                raise ValueError(f"Drift detector {detector_id} not found")
            
            # Detect drift
            result = detector.detect_drift(current_data)
            
            # Store the result
            self.results[result.id] = result
            
            # Create an alert if drift is detected
            if result.drift_detected:
                alert_manager = get_alert_manager()
                
                alert_manager.create_alert(
                    rule_id="drift_detected",
                    level=AlertLevel.WARNING,
                    message=f"Drift detected by detector {detector.name}",
                    details=result.to_dict(),
                )
            
            return result
    
    def get_result(self, result_id: str) -> Optional[DriftResult]:
        """
        Get a drift result by ID.
        
        Args:
            result_id: The drift result ID.
            
        Returns:
            The drift result, or None if the result is not found.
        """
        with self.lock:
            return self.results.get(result_id)
    
    def get_results(self, detector_id: Optional[str] = None) -> Dict[str, DriftResult]:
        """
        Get drift results.
        
        Args:
            detector_id: The drift detector ID. If None, get all results.
            
        Returns:
            A dictionary mapping drift result IDs to drift results.
        """
        with self.lock:
            if detector_id:
                return {
                    result_id: result
                    for result_id, result in self.results.items()
                    if result.detector_id == detector_id
                }
            
            return self.results.copy()


# Global drift manager instance
_drift_manager = None
_drift_manager_lock = threading.RLock()


def get_drift_manager() -> DriftManager:
    """
    Get the global drift manager instance.
    
    Returns:
        The global drift manager instance.
    """
    global _drift_manager
    
    with _drift_manager_lock:
        if _drift_manager is None:
            _drift_manager = DriftManager()
        
        return _drift_manager
