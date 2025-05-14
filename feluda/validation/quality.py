"""
Data quality module for Feluda.

This module provides data quality checks for data validation.
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

from feluda.config import get_config
from feluda.monitoring.alerts import AlertLevel, get_alert_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class QualityCheckResult(BaseModel):
    """
    Quality check result.
    
    This class represents the result of a quality check.
    """
    
    id: str = Field(..., description="The quality check result ID")
    check_id: str = Field(..., description="The quality check ID")
    check_name: str = Field(..., description="The quality check name")
    passed: bool = Field(..., description="Whether the check passed")
    metrics: Dict[str, float] = Field(default_factory=dict, description="The quality metrics")
    details: Dict[str, Any] = Field(default_factory=dict, description="The quality details")
    timestamp: float = Field(..., description="The quality check timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the quality check result to a dictionary.
        
        Returns:
            A dictionary representation of the quality check result.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityCheckResult":
        """
        Create a quality check result from a dictionary.
        
        Args:
            data: The dictionary to create the quality check result from.
            
        Returns:
            A quality check result.
        """
        return cls(**data)


class QualityCheck(abc.ABC):
    """
    Base class for quality checks.
    
    This class defines the interface for quality checks.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, id: str, name: str):
        """
        Initialize a quality check.
        
        Args:
            id: The quality check ID.
            name: The quality check name.
        """
        self.id = id
        self.name = name
    
    @abc.abstractmethod
    def check(self, data: Any) -> QualityCheckResult:
        """
        Perform a quality check.
        
        Args:
            data: The data to check.
            
        Returns:
            A quality check result.
        """
        pass


class MissingValuesCheck(QualityCheck):
    """
    Missing values check.
    
    This class implements a quality check that checks for missing values.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        threshold: float = 0.1,
        columns: Optional[List[str]] = None,
    ):
        """
        Initialize a missing values check.
        
        Args:
            id: The quality check ID.
            name: The quality check name.
            threshold: The maximum allowed percentage of missing values.
            columns: The columns to check. If None, check all columns.
        """
        super().__init__(id, name)
        self.threshold = threshold
        self.columns = columns
    
    def check(self, data: pd.DataFrame) -> QualityCheckResult:
        """
        Check for missing values.
        
        Args:
            data: The data to check.
            
        Returns:
            A quality check result.
        """
        # Check if the data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        # Get the columns to check
        columns = self.columns or data.columns
        
        # Check for missing values
        missing_values = data[columns].isnull().sum()
        missing_percentages = missing_values / len(data)
        
        # Check if the missing values are below the threshold
        passed = all(percentage <= self.threshold for percentage in missing_percentages)
        
        # Create the quality check result
        return QualityCheckResult(
            id=str(uuid.uuid4()),
            check_id=self.id,
            check_name=self.name,
            passed=passed,
            metrics={
                "max_missing_percentage": float(missing_percentages.max()),
                "mean_missing_percentage": float(missing_percentages.mean()),
            },
            details={
                "threshold": self.threshold,
                "missing_values": missing_values.to_dict(),
                "missing_percentages": missing_percentages.to_dict(),
            },
            timestamp=time.time(),
        )


class DuplicatesCheck(QualityCheck):
    """
    Duplicates check.
    
    This class implements a quality check that checks for duplicate rows.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        threshold: float = 0.1,
        columns: Optional[List[str]] = None,
    ):
        """
        Initialize a duplicates check.
        
        Args:
            id: The quality check ID.
            name: The quality check name.
            threshold: The maximum allowed percentage of duplicate rows.
            columns: The columns to check for duplicates. If None, check all columns.
        """
        super().__init__(id, name)
        self.threshold = threshold
        self.columns = columns
    
    def check(self, data: pd.DataFrame) -> QualityCheckResult:
        """
        Check for duplicate rows.
        
        Args:
            data: The data to check.
            
        Returns:
            A quality check result.
        """
        # Check if the data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        # Get the columns to check
        columns = self.columns or data.columns
        
        # Check for duplicates
        duplicates = data.duplicated(subset=columns)
        num_duplicates = duplicates.sum()
        duplicate_percentage = num_duplicates / len(data)
        
        # Check if the duplicate percentage is below the threshold
        passed = duplicate_percentage <= self.threshold
        
        # Create the quality check result
        return QualityCheckResult(
            id=str(uuid.uuid4()),
            check_id=self.id,
            check_name=self.name,
            passed=passed,
            metrics={
                "duplicate_percentage": float(duplicate_percentage),
                "num_duplicates": int(num_duplicates),
            },
            details={
                "threshold": self.threshold,
                "columns": list(columns),
            },
            timestamp=time.time(),
        )


class OutliersCheck(QualityCheck):
    """
    Outliers check.
    
    This class implements a quality check that checks for outliers.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        threshold: float = 0.1,
        columns: Optional[List[str]] = None,
        method: str = "zscore",
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
    ):
        """
        Initialize an outliers check.
        
        Args:
            id: The quality check ID.
            name: The quality check name.
            threshold: The maximum allowed percentage of outliers.
            columns: The columns to check for outliers. If None, check all numeric columns.
            method: The outlier detection method. One of "zscore" or "iqr".
            zscore_threshold: The Z-score threshold for outlier detection.
            iqr_multiplier: The IQR multiplier for outlier detection.
        """
        super().__init__(id, name)
        self.threshold = threshold
        self.columns = columns
        self.method = method
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
    
    def check(self, data: pd.DataFrame) -> QualityCheckResult:
        """
        Check for outliers.
        
        Args:
            data: The data to check.
            
        Returns:
            A quality check result.
        """
        # Check if the data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        # Get the columns to check
        if self.columns:
            columns = self.columns
        else:
            # Get numeric columns
            columns = data.select_dtypes(include=np.number).columns
        
        # Check for outliers
        outlier_counts = {}
        outlier_percentages = {}
        
        for column in columns:
            # Get the column data
            column_data = data[column].dropna()
            
            # Detect outliers
            if self.method == "zscore":
                # Z-score method
                z_scores = np.abs((column_data - column_data.mean()) / column_data.std())
                outliers = z_scores > self.zscore_threshold
            elif self.method == "iqr":
                # IQR method
                q1 = column_data.quantile(0.25)
                q3 = column_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - self.iqr_multiplier * iqr
                upper_bound = q3 + self.iqr_multiplier * iqr
                outliers = (column_data < lower_bound) | (column_data > upper_bound)
            else:
                raise ValueError(f"Unsupported outlier detection method: {self.method}")
            
            # Count outliers
            outlier_count = outliers.sum()
            outlier_percentage = outlier_count / len(column_data)
            
            outlier_counts[column] = int(outlier_count)
            outlier_percentages[column] = float(outlier_percentage)
        
        # Check if the outlier percentages are below the threshold
        passed = all(percentage <= self.threshold for percentage in outlier_percentages.values())
        
        # Create the quality check result
        return QualityCheckResult(
            id=str(uuid.uuid4()),
            check_id=self.id,
            check_name=self.name,
            passed=passed,
            metrics={
                "max_outlier_percentage": float(max(outlier_percentages.values())) if outlier_percentages else 0.0,
                "mean_outlier_percentage": float(np.mean(list(outlier_percentages.values()))) if outlier_percentages else 0.0,
            },
            details={
                "threshold": self.threshold,
                "method": self.method,
                "zscore_threshold": self.zscore_threshold if self.method == "zscore" else None,
                "iqr_multiplier": self.iqr_multiplier if self.method == "iqr" else None,
                "outlier_counts": outlier_counts,
                "outlier_percentages": outlier_percentages,
            },
            timestamp=time.time(),
        )


class SchemaCheck(QualityCheck):
    """
    Schema check.
    
    This class implements a quality check that checks if data conforms to a schema.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        schema: Any,
    ):
        """
        Initialize a schema check.
        
        Args:
            id: The quality check ID.
            name: The quality check name.
            schema: The schema to check against.
        """
        super().__init__(id, name)
        self.schema = schema
    
    def check(self, data: Any) -> QualityCheckResult:
        """
        Check if data conforms to a schema.
        
        Args:
            data: The data to check.
            
        Returns:
            A quality check result.
        """
        # Validate the data against the schema
        is_valid, error_message = self.schema.validate(data)
        
        # Create the quality check result
        return QualityCheckResult(
            id=str(uuid.uuid4()),
            check_id=self.id,
            check_name=self.name,
            passed=is_valid,
            metrics={},
            details={
                "error_message": error_message,
            },
            timestamp=time.time(),
        )


class QualityManager:
    """
    Quality manager.
    
    This class is responsible for managing quality checks and quality check results.
    """
    
    def __init__(self):
        """
        Initialize the quality manager.
        """
        self.checks: Dict[str, QualityCheck] = {}
        self.results: Dict[str, QualityCheckResult] = {}
        self.lock = threading.RLock()
    
    def register_check(self, check: QualityCheck) -> None:
        """
        Register a quality check.
        
        Args:
            check: The quality check to register.
        """
        with self.lock:
            self.checks[check.id] = check
    
    def get_check(self, check_id: str) -> Optional[QualityCheck]:
        """
        Get a quality check by ID.
        
        Args:
            check_id: The quality check ID.
            
        Returns:
            The quality check, or None if the check is not found.
        """
        with self.lock:
            return self.checks.get(check_id)
    
    def get_checks(self) -> Dict[str, QualityCheck]:
        """
        Get all quality checks.
        
        Returns:
            A dictionary mapping quality check IDs to quality checks.
        """
        with self.lock:
            return self.checks.copy()
    
    def run_check(self, check_id: str, data: Any) -> QualityCheckResult:
        """
        Run a quality check.
        
        Args:
            check_id: The quality check ID.
            data: The data to check.
            
        Returns:
            A quality check result.
        """
        with self.lock:
            # Get the quality check
            check = self.get_check(check_id)
            
            if not check:
                raise ValueError(f"Quality check {check_id} not found")
            
            # Run the check
            result = check.check(data)
            
            # Store the result
            self.results[result.id] = result
            
            # Create an alert if the check failed
            if not result.passed:
                alert_manager = get_alert_manager()
                
                alert_manager.create_alert(
                    rule_id="quality_check_failed",
                    level=AlertLevel.WARNING,
                    message=f"Quality check {check.name} failed",
                    details=result.to_dict(),
                )
            
            return result
    
    def run_checks(self, data: Any, check_ids: Optional[List[str]] = None) -> Dict[str, QualityCheckResult]:
        """
        Run multiple quality checks.
        
        Args:
            data: The data to check.
            check_ids: The quality check IDs. If None, run all checks.
            
        Returns:
            A dictionary mapping quality check IDs to quality check results.
        """
        with self.lock:
            # Get the quality checks to run
            if check_ids:
                checks = {
                    check_id: self.get_check(check_id)
                    for check_id in check_ids
                    if self.get_check(check_id)
                }
            else:
                checks = self.get_checks()
            
            # Run the checks
            results = {}
            
            for check_id, check in checks.items():
                try:
                    result = self.run_check(check_id, data)
                    results[check_id] = result
                except Exception as e:
                    log.error(f"Error running quality check {check_id}: {e}")
            
            return results
    
    def get_result(self, result_id: str) -> Optional[QualityCheckResult]:
        """
        Get a quality check result by ID.
        
        Args:
            result_id: The quality check result ID.
            
        Returns:
            The quality check result, or None if the result is not found.
        """
        with self.lock:
            return self.results.get(result_id)
    
    def get_results(self, check_id: Optional[str] = None) -> Dict[str, QualityCheckResult]:
        """
        Get quality check results.
        
        Args:
            check_id: The quality check ID. If None, get all results.
            
        Returns:
            A dictionary mapping quality check result IDs to quality check results.
        """
        with self.lock:
            if check_id:
                return {
                    result_id: result
                    for result_id, result in self.results.items()
                    if result.check_id == check_id
                }
            
            return self.results.copy()


# Global quality manager instance
_quality_manager = None
_quality_manager_lock = threading.RLock()


def get_quality_manager() -> QualityManager:
    """
    Get the global quality manager instance.
    
    Returns:
        The global quality manager instance.
    """
    global _quality_manager
    
    with _quality_manager_lock:
        if _quality_manager is None:
            _quality_manager = QualityManager()
        
        return _quality_manager
