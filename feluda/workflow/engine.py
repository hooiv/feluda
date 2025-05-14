"""
Workflow engine for Feluda.

This module provides a workflow engine for complex processing pipelines.
"""

import abc
import datetime
import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.observability import get_logger
from feluda.workflow.storage import StorageManager, get_storage_manager

log = get_logger(__name__)


class TaskStatus(str, enum.Enum):
    """Enum for task status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(str, enum.Enum):
    """Enum for workflow status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStatus(str, enum.Enum):
    """Enum for execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskDefinition(BaseModel):
    """
    Task definition.
    
    This class defines a task in a workflow.
    """
    
    id: str = Field(..., description="The task ID")
    name: str = Field(..., description="The task name")
    description: Optional[str] = Field(None, description="The task description")
    type: str = Field(..., description="The task type")
    config: Dict[str, Any] = Field(default_factory=dict, description="The task configuration")
    dependencies: List[str] = Field(default_factory=list, description="The task dependencies")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task definition to a dictionary.
        
        Returns:
            A dictionary representation of the task definition.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDefinition":
        """
        Create a task definition from a dictionary.
        
        Args:
            data: The dictionary to create the task definition from.
            
        Returns:
            A task definition.
        """
        return cls(**data)


class WorkflowDefinition(BaseModel):
    """
    Workflow definition.
    
    This class defines a workflow.
    """
    
    id: str = Field(..., description="The workflow ID")
    name: str = Field(..., description="The workflow name")
    description: Optional[str] = Field(None, description="The workflow description")
    tasks: Dict[str, TaskDefinition] = Field(..., description="The workflow tasks")
    config: Dict[str, Any] = Field(default_factory=dict, description="The workflow configuration")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workflow definition to a dictionary.
        
        Returns:
            A dictionary representation of the workflow definition.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        """
        Create a workflow definition from a dictionary.
        
        Args:
            data: The dictionary to create the workflow definition from.
            
        Returns:
            A workflow definition.
        """
        return cls(**data)
    
    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        """
        Get a task by ID.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task, or None if the task is not found.
        """
        return self.tasks.get(task_id)
    
    def get_dependencies(self, task_id: str) -> List[TaskDefinition]:
        """
        Get the dependencies of a task.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task dependencies.
        """
        task = self.get_task(task_id)
        
        if not task:
            return []
        
        return [self.get_task(dep_id) for dep_id in task.dependencies if self.get_task(dep_id)]
    
    def get_dependents(self, task_id: str) -> List[TaskDefinition]:
        """
        Get the dependents of a task.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task dependents.
        """
        return [
            task for task in self.tasks.values()
            if task_id in task.dependencies
        ]
    
    def get_root_tasks(self) -> List[TaskDefinition]:
        """
        Get the root tasks of the workflow.
        
        Returns:
            The root tasks.
        """
        return [
            task for task in self.tasks.values()
            if not task.dependencies
        ]
    
    def get_leaf_tasks(self) -> List[TaskDefinition]:
        """
        Get the leaf tasks of the workflow.
        
        Returns:
            The leaf tasks.
        """
        return [
            task for task in self.tasks.values()
            if not self.get_dependents(task.id)
        ]
    
    def validate(self) -> bool:
        """
        Validate the workflow definition.
        
        Returns:
            True if the workflow definition is valid, False otherwise.
        """
        # Check for cycles
        visited = set()
        path = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in path:
                return True
            
            if task_id in visited:
                return False
            
            visited.add(task_id)
            path.add(task_id)
            
            task = self.get_task(task_id)
            
            if not task:
                return False
            
            for dep_id in task.dependencies:
                if has_cycle(dep_id):
                    return True
            
            path.remove(task_id)
            return False
        
        for task_id in self.tasks:
            if has_cycle(task_id):
                return False
        
        return True


class ExecutionResult(BaseModel):
    """
    Execution result.
    
    This class represents the result of a task execution.
    """
    
    status: ExecutionStatus = Field(..., description="The execution status")
    result: Optional[Any] = Field(None, description="The execution result")
    error: Optional[str] = Field(None, description="The execution error")
    start_time: Optional[datetime.datetime] = Field(None, description="The execution start time")
    end_time: Optional[datetime.datetime] = Field(None, description="The execution end time")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the execution result to a dictionary.
        
        Returns:
            A dictionary representation of the execution result.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """
        Create an execution result from a dictionary.
        
        Args:
            data: The dictionary to create the execution result from.
            
        Returns:
            An execution result.
        """
        return cls(**data)


class ExecutionContext(BaseModel):
    """
    Execution context.
    
    This class represents the context of a task execution.
    """
    
    workflow_id: str = Field(..., description="The workflow ID")
    execution_id: str = Field(..., description="The execution ID")
    task_id: str = Field(..., description="The task ID")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="The task inputs")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="The task outputs")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="The task parameters")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the execution context to a dictionary.
        
        Returns:
            A dictionary representation of the execution context.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionContext":
        """
        Create an execution context from a dictionary.
        
        Args:
            data: The dictionary to create the execution context from.
            
        Returns:
            An execution context.
        """
        return cls(**data)


class Task(abc.ABC):
    """
    Base class for tasks.
    
    This class defines the interface for tasks.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, definition: TaskDefinition):
        """
        Initialize a task.
        
        Args:
            definition: The task definition.
        """
        self.definition = definition
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
    
    @property
    def id(self) -> str:
        """
        Get the task ID.
        
        Returns:
            The task ID.
        """
        return self.definition.id
    
    @property
    def name(self) -> str:
        """
        Get the task name.
        
        Returns:
            The task name.
        """
        return self.definition.name
    
    @property
    def description(self) -> Optional[str]:
        """
        Get the task description.
        
        Returns:
            The task description.
        """
        return self.definition.description
    
    @property
    def type(self) -> str:
        """
        Get the task type.
        
        Returns:
            The task type.
        """
        return self.definition.type
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the task configuration.
        
        Returns:
            The task configuration.
        """
        return self.definition.config
    
    @property
    def dependencies(self) -> List[str]:
        """
        Get the task dependencies.
        
        Returns:
            The task dependencies.
        """
        return self.definition.dependencies
    
    @abc.abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute the task.
        
        Args:
            context: The execution context.
            
        Returns:
            The execution result.
        """
        pass


class Workflow:
    """
    Workflow.
    
    This class represents a workflow.
    """
    
    def __init__(self, definition: WorkflowDefinition, tasks: Optional[Dict[str, Task]] = None):
        """
        Initialize a workflow.
        
        Args:
            definition: The workflow definition.
            tasks: The workflow tasks.
        """
        self.definition = definition
        self.tasks = tasks or {}
        self.status = WorkflowStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.execution_id = str(uuid.uuid4())
    
    @property
    def id(self) -> str:
        """
        Get the workflow ID.
        
        Returns:
            The workflow ID.
        """
        return self.definition.id
    
    @property
    def name(self) -> str:
        """
        Get the workflow name.
        
        Returns:
            The workflow name.
        """
        return self.definition.name
    
    @property
    def description(self) -> Optional[str]:
        """
        Get the workflow description.
        
        Returns:
            The workflow description.
        """
        return self.definition.description
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the workflow configuration.
        
        Returns:
            The workflow configuration.
        """
        return self.definition.config
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task, or None if the task is not found.
        """
        return self.tasks.get(task_id)
    
    def get_dependencies(self, task_id: str) -> List[Task]:
        """
        Get the dependencies of a task.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task dependencies.
        """
        task = self.get_task(task_id)
        
        if not task:
            return []
        
        return [self.get_task(dep_id) for dep_id in task.dependencies if self.get_task(dep_id)]
    
    def get_dependents(self, task_id: str) -> List[Task]:
        """
        Get the dependents of a task.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task dependents.
        """
        return [
            task for task in self.tasks.values()
            if task_id in task.dependencies
        ]
    
    def get_root_tasks(self) -> List[Task]:
        """
        Get the root tasks of the workflow.
        
        Returns:
            The root tasks.
        """
        return [
            task for task in self.tasks.values()
            if not task.dependencies
        ]
    
    def get_leaf_tasks(self) -> List[Task]:
        """
        Get the leaf tasks of the workflow.
        
        Returns:
            The leaf tasks.
        """
        return [
            task for task in self.tasks.values()
            if not self.get_dependents(task.id)
        ]
    
    def is_task_ready(self, task_id: str) -> bool:
        """
        Check if a task is ready to execute.
        
        Args:
            task_id: The task ID.
            
        Returns:
            True if the task is ready to execute, False otherwise.
        """
        task = self.get_task(task_id)
        
        if not task:
            return False
        
        if task.status != TaskStatus.PENDING:
            return False
        
        for dep in self.get_dependencies(task_id):
            if dep.status != TaskStatus.SUCCEEDED:
                return False
        
        return True
    
    def is_workflow_complete(self) -> bool:
        """
        Check if the workflow is complete.
        
        Returns:
            True if the workflow is complete, False otherwise.
        """
        return all(
            task.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED)
            for task in self.tasks.values()
        )
    
    def is_workflow_successful(self) -> bool:
        """
        Check if the workflow is successful.
        
        Returns:
            True if the workflow is successful, False otherwise.
        """
        return all(
            task.status in (TaskStatus.SUCCEEDED, TaskStatus.SKIPPED)
            for task in self.tasks.values()
        )
    
    def get_next_tasks(self) -> List[Task]:
        """
        Get the next tasks to execute.
        
        Returns:
            The next tasks to execute.
        """
        return [
            task for task_id, task in self.tasks.items()
            if self.is_task_ready(task_id)
        ]
    
    def execute_task(self, task_id: str, inputs: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a task.
        
        Args:
            task_id: The task ID.
            inputs: The task inputs.
            
        Returns:
            The execution result.
        """
        task = self.get_task(task_id)
        
        if not task:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Task {task_id} not found",
            )
        
        if task.status != TaskStatus.PENDING:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Task {task_id} is not pending",
            )
        
        # Create the execution context
        context = ExecutionContext(
            workflow_id=self.id,
            execution_id=self.execution_id,
            task_id=task_id,
            inputs=inputs,
        )
        
        # Execute the task
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.datetime.now()
        
        try:
            result = task.execute(context)
            
            if result.status == ExecutionStatus.SUCCEEDED:
                task.status = TaskStatus.SUCCEEDED
                task.result = result.result
            elif result.status == ExecutionStatus.FAILED:
                task.status = TaskStatus.FAILED
                task.error = result.error
            elif result.status == ExecutionStatus.CANCELLED:
                task.status = TaskStatus.CANCELLED
            
            task.end_time = datetime.datetime.now()
            
            return result
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = datetime.datetime.now()
            
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                start_time=task.start_time,
                end_time=task.end_time,
            )


class WorkflowEngine:
    """
    Workflow engine.
    
    This class is responsible for executing workflows.
    """
    
    def __init__(self, storage_manager: Optional[StorageManager] = None):
        """
        Initialize the workflow engine.
        
        Args:
            storage_manager: The storage manager.
        """
        self.storage_manager = storage_manager or get_storage_manager()
        self.workflows: Dict[str, Workflow] = {}
        self.lock = threading.RLock()
    
    def register_workflow(self, workflow: Workflow) -> None:
        """
        Register a workflow.
        
        Args:
            workflow: The workflow to register.
        """
        with self.lock:
            self.workflows[workflow.id] = workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """
        Get a workflow by ID.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            The workflow, or None if the workflow is not found.
        """
        with self.lock:
            return self.workflows.get(workflow_id)
    
    def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_id: The workflow ID.
            inputs: The workflow inputs.
            
        Returns:
            The workflow outputs.
        """
        workflow = self.get_workflow(workflow_id)
        
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Initialize the workflow
        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = datetime.datetime.now()
        
        # Execute the workflow
        outputs = {}
        
        while True:
            # Get the next tasks to execute
            next_tasks = workflow.get_next_tasks()
            
            if not next_tasks:
                # No more tasks to execute
                break
            
            # Execute the tasks
            for task in next_tasks:
                # Get the task inputs
                task_inputs = {}
                
                for dep in workflow.get_dependencies(task.id):
                    if dep.status == TaskStatus.SUCCEEDED and dep.result is not None:
                        task_inputs[dep.id] = dep.result
                
                # Add the workflow inputs
                task_inputs.update(inputs)
                
                # Execute the task
                result = workflow.execute_task(task.id, task_inputs)
                
                if result.status == ExecutionStatus.SUCCEEDED and result.result is not None:
                    outputs[task.id] = result.result
        
        # Update the workflow status
        if workflow.is_workflow_complete():
            if workflow.is_workflow_successful():
                workflow.status = WorkflowStatus.SUCCEEDED
            else:
                workflow.status = WorkflowStatus.FAILED
        
        workflow.end_time = datetime.datetime.now()
        
        return outputs
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a workflow.
        
        Args:
            workflow_id: The workflow ID.
            
        Returns:
            True if the workflow was cancelled, False otherwise.
        """
        workflow = self.get_workflow(workflow_id)
        
        if not workflow:
            return False
        
        # Cancel the workflow
        workflow.status = WorkflowStatus.CANCELLED
        
        # Cancel all pending tasks
        for task in workflow.tasks.values():
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
        
        return True


# Global workflow engine instance
_workflow_engine = None
_workflow_engine_lock = threading.RLock()


def get_workflow_engine() -> WorkflowEngine:
    """
    Get the global workflow engine instance.
    
    Returns:
        The global workflow engine instance.
    """
    global _workflow_engine
    
    with _workflow_engine_lock:
        if _workflow_engine is None:
            _workflow_engine = WorkflowEngine()
        
        return _workflow_engine
