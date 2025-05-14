"""
Pipeline module for Feluda.

This module provides a data pipeline for processing large datasets.
"""

import abc
import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import pandas as pd
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.data_pipeline.connectors import Connector, get_connector_manager
from feluda.data_pipeline.processors import DataProcessor, get_processor_manager
from feluda.data_pipeline.transformers import DataTransformer, get_transformer_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class PipelineStage(BaseModel):
    """
    Pipeline stage.
    
    This class represents a stage in a data pipeline.
    """
    
    id: str = Field(..., description="The stage ID")
    name: str = Field(..., description="The stage name")
    description: Optional[str] = Field(None, description="The stage description")
    processor_name: str = Field(..., description="The processor name")
    input_connector_name: Optional[str] = Field(None, description="The input connector name")
    output_connector_name: Optional[str] = Field(None, description="The output connector name")
    dependencies: List[str] = Field(default_factory=list, description="The stage dependencies")
    config: Dict[str, Any] = Field(default_factory=dict, description="The stage configuration")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pipeline stage to a dictionary.
        
        Returns:
            A dictionary representation of the pipeline stage.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStage":
        """
        Create a pipeline stage from a dictionary.
        
        Args:
            data: The dictionary to create the pipeline stage from.
            
        Returns:
            A pipeline stage.
        """
        return cls(**data)


class DataPipeline(BaseModel):
    """
    Data pipeline.
    
    This class represents a data pipeline.
    """
    
    id: str = Field(..., description="The pipeline ID")
    name: str = Field(..., description="The pipeline name")
    description: Optional[str] = Field(None, description="The pipeline description")
    stages: Dict[str, PipelineStage] = Field(..., description="The pipeline stages")
    config: Dict[str, Any] = Field(default_factory=dict, description="The pipeline configuration")
    
    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the data pipeline to a dictionary.
        
        Returns:
            A dictionary representation of the data pipeline.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "stages": {
                stage_id: stage.to_dict()
                for stage_id, stage in self.stages.items()
            },
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPipeline":
        """
        Create a data pipeline from a dictionary.
        
        Args:
            data: The dictionary to create the data pipeline from.
            
        Returns:
            A data pipeline.
        """
        stages = {
            stage_id: PipelineStage.from_dict(stage_data)
            for stage_id, stage_data in data.get("stages", {}).items()
        }
        
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            stages=stages,
            config=data.get("config", {}),
        )
    
    def get_stage(self, stage_id: str) -> Optional[PipelineStage]:
        """
        Get a stage by ID.
        
        Args:
            stage_id: The stage ID.
            
        Returns:
            The stage, or None if the stage is not found.
        """
        return self.stages.get(stage_id)
    
    def get_dependencies(self, stage_id: str) -> List[PipelineStage]:
        """
        Get the dependencies of a stage.
        
        Args:
            stage_id: The stage ID.
            
        Returns:
            The stage dependencies.
        """
        stage = self.get_stage(stage_id)
        
        if not stage:
            return []
        
        return [self.get_stage(dep_id) for dep_id in stage.dependencies if self.get_stage(dep_id)]
    
    def get_dependents(self, stage_id: str) -> List[PipelineStage]:
        """
        Get the dependents of a stage.
        
        Args:
            stage_id: The stage ID.
            
        Returns:
            The stage dependents.
        """
        return [
            stage for stage in self.stages.values()
            if stage_id in stage.dependencies
        ]
    
    def get_root_stages(self) -> List[PipelineStage]:
        """
        Get the root stages of the pipeline.
        
        Returns:
            The root stages.
        """
        return [
            stage for stage in self.stages.values()
            if not stage.dependencies
        ]
    
    def get_leaf_stages(self) -> List[PipelineStage]:
        """
        Get the leaf stages of the pipeline.
        
        Returns:
            The leaf stages.
        """
        return [
            stage for stage in self.stages.values()
            if not self.get_dependents(stage.id)
        ]
    
    def validate(self) -> bool:
        """
        Validate the pipeline.
        
        Returns:
            True if the pipeline is valid, False otherwise.
        """
        # Check for cycles
        visited = set()
        path = set()
        
        def has_cycle(stage_id: str) -> bool:
            if stage_id in path:
                return True
            
            if stage_id in visited:
                return False
            
            visited.add(stage_id)
            path.add(stage_id)
            
            stage = self.get_stage(stage_id)
            
            if not stage:
                return False
            
            for dep_id in stage.dependencies:
                if has_cycle(dep_id):
                    return True
            
            path.remove(stage_id)
            return False
        
        for stage_id in self.stages:
            if has_cycle(stage_id):
                return False
        
        return True
    
    def execute(self, data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Args:
            data: The input data. If None, data is read from the input connectors.
            
        Returns:
            A dictionary mapping stage IDs to results.
        """
        # Validate the pipeline
        if not self.validate():
            raise ValueError("Invalid pipeline")
        
        # Get the managers
        connector_manager = get_connector_manager()
        processor_manager = get_processor_manager()
        
        # Execute the pipeline
        results = {}
        executed_stages = set()
        
        while len(executed_stages) < len(self.stages):
            # Find stages that can be executed
            executable_stages = []
            
            for stage_id, stage in self.stages.items():
                if stage_id in executed_stages:
                    continue
                
                # Check if all dependencies have been executed
                if all(dep_id in executed_stages for dep_id in stage.dependencies):
                    executable_stages.append(stage)
            
            if not executable_stages:
                # No more stages can be executed
                break
            
            # Execute the stages
            for stage in executable_stages:
                try:
                    # Get the processor
                    processor = processor_manager.get_processor(stage.processor_name)
                    
                    if not processor:
                        log.error(f"Processor {stage.processor_name} not found")
                        results[stage.id] = None
                        executed_stages.add(stage.id)
                        continue
                    
                    # Get the input data
                    stage_data = data
                    
                    if stage.input_connector_name:
                        # Read data from the input connector
                        input_connector = connector_manager.get_connector(stage.input_connector_name)
                        
                        if not input_connector:
                            log.error(f"Input connector {stage.input_connector_name} not found")
                            results[stage.id] = None
                            executed_stages.add(stage.id)
                            continue
                        
                        stage_data = input_connector.read()
                    elif stage.dependencies:
                        # Get data from dependencies
                        stage_data = [results[dep_id] for dep_id in stage.dependencies]
                        
                        if len(stage_data) == 1:
                            stage_data = stage_data[0]
                    
                    # Process the data
                    result = processor.process(stage_data)
                    
                    # Write the result to the output connector
                    if stage.output_connector_name:
                        output_connector = connector_manager.get_connector(stage.output_connector_name)
                        
                        if not output_connector:
                            log.error(f"Output connector {stage.output_connector_name} not found")
                        else:
                            output_connector.write(result)
                    
                    # Store the result
                    results[stage.id] = result
                    executed_stages.add(stage.id)
                
                except Exception as e:
                    log.error(f"Error executing stage {stage.id}: {e}")
                    results[stage.id] = None
                    executed_stages.add(stage.id)
        
        return results


class PipelineManager:
    """
    Pipeline manager.
    
    This class is responsible for managing data pipelines.
    """
    
    def __init__(self):
        """
        Initialize the pipeline manager.
        """
        self.pipelines: Dict[str, DataPipeline] = {}
        self.lock = threading.RLock()
    
    def register_pipeline(self, pipeline: DataPipeline) -> None:
        """
        Register a pipeline.
        
        Args:
            pipeline: The pipeline to register.
        """
        with self.lock:
            self.pipelines[pipeline.id] = pipeline
    
    def get_pipeline(self, pipeline_id: str) -> Optional[DataPipeline]:
        """
        Get a pipeline by ID.
        
        Args:
            pipeline_id: The pipeline ID.
            
        Returns:
            The pipeline, or None if the pipeline is not found.
        """
        with self.lock:
            return self.pipelines.get(pipeline_id)
    
    def get_pipelines(self) -> Dict[str, DataPipeline]:
        """
        Get all pipelines.
        
        Returns:
            A dictionary mapping pipeline IDs to pipelines.
        """
        with self.lock:
            return self.pipelines.copy()
    
    def create_pipeline(self, name: str, description: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> DataPipeline:
        """
        Create a pipeline.
        
        Args:
            name: The pipeline name.
            description: The pipeline description.
            config: The pipeline configuration.
            
        Returns:
            The created pipeline.
        """
        with self.lock:
            pipeline = DataPipeline(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                stages={},
                config=config or {},
            )
            
            self.register_pipeline(pipeline)
            return pipeline
    
    def add_stage(
        self,
        pipeline_id: str,
        name: str,
        processor_name: str,
        input_connector_name: Optional[str] = None,
        output_connector_name: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Optional[PipelineStage]:
        """
        Add a stage to a pipeline.
        
        Args:
            pipeline_id: The pipeline ID.
            name: The stage name.
            processor_name: The processor name.
            input_connector_name: The input connector name.
            output_connector_name: The output connector name.
            dependencies: The stage dependencies.
            config: The stage configuration.
            description: The stage description.
            
        Returns:
            The created stage, or None if the pipeline is not found.
        """
        with self.lock:
            pipeline = self.get_pipeline(pipeline_id)
            
            if not pipeline:
                return None
            
            stage = PipelineStage(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                processor_name=processor_name,
                input_connector_name=input_connector_name,
                output_connector_name=output_connector_name,
                dependencies=dependencies or [],
                config=config or {},
            )
            
            pipeline.stages[stage.id] = stage
            return stage
    
    def remove_stage(self, pipeline_id: str, stage_id: str) -> bool:
        """
        Remove a stage from a pipeline.
        
        Args:
            pipeline_id: The pipeline ID.
            stage_id: The stage ID.
            
        Returns:
            True if the stage was removed, False otherwise.
        """
        with self.lock:
            pipeline = self.get_pipeline(pipeline_id)
            
            if not pipeline:
                return False
            
            if stage_id not in pipeline.stages:
                return False
            
            # Remove the stage
            del pipeline.stages[stage_id]
            
            # Remove the stage from dependencies
            for stage in pipeline.stages.values():
                if stage_id in stage.dependencies:
                    stage.dependencies.remove(stage_id)
            
            return True
    
    def execute_pipeline(self, pipeline_id: str, data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute a pipeline.
        
        Args:
            pipeline_id: The pipeline ID.
            data: The input data. If None, data is read from the input connectors.
            
        Returns:
            A dictionary mapping stage IDs to results.
        """
        with self.lock:
            pipeline = self.get_pipeline(pipeline_id)
            
            if not pipeline:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            return pipeline.execute(data)


# Global pipeline manager instance
_pipeline_manager = None
_pipeline_manager_lock = threading.RLock()


def get_pipeline_manager() -> PipelineManager:
    """
    Get the global pipeline manager instance.
    
    Returns:
        The global pipeline manager instance.
    """
    global _pipeline_manager
    
    with _pipeline_manager_lock:
        if _pipeline_manager is None:
            _pipeline_manager = PipelineManager()
        
        return _pipeline_manager
