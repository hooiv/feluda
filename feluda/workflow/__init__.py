"""
Workflow engine for Feluda.

This module provides a workflow engine for complex processing pipelines.
"""

from feluda.workflow.engine import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    Task,
    TaskDefinition,
    TaskStatus,
    Workflow,
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowStatus,
    get_workflow_engine,
)
from feluda.workflow.operators import (
    BranchOperator,
    ConditionalOperator,
    JoinOperator,
    MapOperator,
    OperatorTask,
    ParallelOperator,
    ReduceOperator,
    SequentialOperator,
)
from feluda.workflow.storage import (
    StorageBackend,
    StorageManager,
    get_storage_manager,
)

__all__ = [
    "BranchOperator",
    "ConditionalOperator",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionStatus",
    "JoinOperator",
    "MapOperator",
    "OperatorTask",
    "ParallelOperator",
    "ReduceOperator",
    "SequentialOperator",
    "StorageBackend",
    "StorageManager",
    "Task",
    "TaskDefinition",
    "TaskStatus",
    "Workflow",
    "WorkflowDefinition",
    "WorkflowEngine",
    "WorkflowStatus",
    "get_storage_manager",
    "get_workflow_engine",
]
