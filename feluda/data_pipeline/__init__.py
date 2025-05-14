"""
Data pipeline for Feluda.

This module provides a data pipeline for processing large datasets.
"""

from feluda.data_pipeline.connectors import (
    Connector,
    ConnectorManager,
    FileConnector,
    KafkaConnector,
    S3Connector,
    SQLConnector,
    get_connector_manager,
)
from feluda.data_pipeline.processors import (
    BatchProcessor,
    DataProcessor,
    ProcessorManager,
    StreamProcessor,
    get_processor_manager,
)
from feluda.data_pipeline.transformers import (
    DataTransformer,
    FilterTransformer,
    JoinTransformer,
    MapTransformer,
    ReduceTransformer,
    TransformerManager,
    get_transformer_manager,
)
from feluda.data_pipeline.pipeline import (
    DataPipeline,
    PipelineManager,
    PipelineStage,
    get_pipeline_manager,
)

__all__ = [
    "BatchProcessor",
    "Connector",
    "ConnectorManager",
    "DataPipeline",
    "DataProcessor",
    "DataTransformer",
    "FileConnector",
    "FilterTransformer",
    "JoinTransformer",
    "KafkaConnector",
    "MapTransformer",
    "PipelineManager",
    "PipelineStage",
    "ProcessorManager",
    "ReduceTransformer",
    "S3Connector",
    "SQLConnector",
    "StreamProcessor",
    "TransformerManager",
    "get_connector_manager",
    "get_pipeline_manager",
    "get_processor_manager",
    "get_transformer_manager",
]
