"""
Worker system for Feluda.

This module provides a worker system for distributed processing.
"""

import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.distributed.broker import BrokerManager, Message, get_broker_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class TaskStatus(str, enum.Enum):
    """Enum for task status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """
    Task.
    
    This class represents a task in the worker system.
    """
    
    id: str = Field(..., description="The task ID")
    type: str = Field(..., description="The task type")
    payload: Dict[str, Any] = Field(..., description="The task payload")
    status: TaskStatus = Field(TaskStatus.PENDING, description="The task status")
    worker_id: Optional[str] = Field(None, description="The worker ID")
    created_at: float = Field(..., description="The task creation timestamp")
    started_at: Optional[float] = Field(None, description="The task start timestamp")
    completed_at: Optional[float] = Field(None, description="The task completion timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="The task result")
    error: Optional[str] = Field(None, description="The task error")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        
        Returns:
            A dictionary representation of the task.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """
        Create a task from a dictionary.
        
        Args:
            data: The dictionary to create the task from.
            
        Returns:
            A task.
        """
        return cls(**data)
    
    @classmethod
    def create(cls, task_type: str, payload: Dict[str, Any]) -> "Task":
        """
        Create a new task.
        
        Args:
            task_type: The task type.
            payload: The task payload.
            
        Returns:
            A new task.
        """
        return cls(
            id=str(uuid.uuid4()),
            type=task_type,
            payload=payload,
            created_at=time.time(),
        )


class TaskResult(BaseModel):
    """
    Task result.
    
    This class represents the result of a task execution.
    """
    
    task_id: str = Field(..., description="The task ID")
    status: TaskStatus = Field(..., description="The task status")
    result: Optional[Dict[str, Any]] = Field(None, description="The task result")
    error: Optional[str] = Field(None, description="The task error")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task result to a dictionary.
        
        Returns:
            A dictionary representation of the task result.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """
        Create a task result from a dictionary.
        
        Args:
            data: The dictionary to create the task result from.
            
        Returns:
            A task result.
        """
        return cls(**data)


class Worker(BaseModel):
    """
    Worker.
    
    This class represents a worker in the worker system.
    """
    
    id: str = Field(..., description="The worker ID")
    name: str = Field(..., description="The worker name")
    capabilities: List[str] = Field(default_factory=list, description="The worker capabilities")
    status: str = Field("idle", description="The worker status")
    task_id: Optional[str] = Field(None, description="The current task ID")
    last_heartbeat: float = Field(..., description="The last heartbeat timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the worker to a dictionary.
        
        Returns:
            A dictionary representation of the worker.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Worker":
        """
        Create a worker from a dictionary.
        
        Args:
            data: The dictionary to create the worker from.
            
        Returns:
            A worker.
        """
        return cls(**data)
    
    @classmethod
    def create(cls, name: str, capabilities: List[str]) -> "Worker":
        """
        Create a new worker.
        
        Args:
            name: The worker name.
            capabilities: The worker capabilities.
            
        Returns:
            A new worker.
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            capabilities=capabilities,
            last_heartbeat=time.time(),
        )


class TaskManager:
    """
    Task manager.
    
    This class is responsible for managing tasks.
    """
    
    def __init__(self, broker_manager: Optional[BrokerManager] = None):
        """
        Initialize the task manager.
        
        Args:
            broker_manager: The broker manager.
        """
        self.broker_manager = broker_manager or get_broker_manager()
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.RLock()
        self.task_handlers: Dict[str, Callable[[Task], TaskResult]] = {}
        
        # Subscribe to task topics
        self.broker_manager.subscribe("task.submit", self._handle_task_submit)
        self.broker_manager.subscribe("task.result", self._handle_task_result)
        self.broker_manager.subscribe("task.cancel", self._handle_task_cancel)
    
    def register_task_handler(self, task_type: str, handler: Callable[[Task], TaskResult]) -> None:
        """
        Register a task handler.
        
        Args:
            task_type: The task type.
            handler: The task handler.
        """
        with self.lock:
            self.task_handlers[task_type] = handler
    
    def submit_task(self, task_type: str, payload: Dict[str, Any]) -> Task:
        """
        Submit a task.
        
        Args:
            task_type: The task type.
            payload: The task payload.
            
        Returns:
            The submitted task.
        """
        with self.lock:
            # Create a task
            task = Task.create(task_type, payload)
            
            # Store the task
            self.tasks[task.id] = task
            
            # Publish a task submit message
            self.broker_manager.publish(
                topic="task.submit",
                payload=task.to_dict(),
            )
            
            return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task, or None if the task is not found.
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: The task ID.
            
        Returns:
            True if the task was cancelled, False otherwise.
        """
        with self.lock:
            task = self.tasks.get(task_id)
            
            if not task:
                return False
            
            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False
            
            # Publish a task cancel message
            self.broker_manager.publish(
                topic="task.cancel",
                payload={"task_id": task_id},
            )
            
            return True
    
    def _handle_task_submit(self, message: Message) -> None:
        """
        Handle a task submit message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the task
            task_dict = message.payload
            task = Task.from_dict(task_dict)
            
            with self.lock:
                # Store the task
                self.tasks[task.id] = task
                
                # Check if we have a handler for this task type
                if task.type in self.task_handlers:
                    # Execute the task
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                    
                    try:
                        # Call the handler
                        result = self.task_handlers[task.type](task)
                        
                        # Update the task
                        task.status = result.status
                        task.result = result.result
                        task.error = result.error
                        task.completed_at = time.time()
                        
                        # Publish a task result message
                        self.broker_manager.publish(
                            topic="task.result",
                            payload=result.to_dict(),
                        )
                    
                    except Exception as e:
                        # Update the task
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.completed_at = time.time()
                        
                        # Publish a task result message
                        self.broker_manager.publish(
                            topic="task.result",
                            payload=TaskResult(
                                task_id=task.id,
                                status=TaskStatus.FAILED,
                                error=str(e),
                            ).to_dict(),
                        )
        
        except Exception as e:
            log.error(f"Error handling task submit message: {e}")
    
    def _handle_task_result(self, message: Message) -> None:
        """
        Handle a task result message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the task result
            result_dict = message.payload
            result = TaskResult.from_dict(result_dict)
            
            with self.lock:
                # Get the task
                task = self.tasks.get(result.task_id)
                
                if not task:
                    return
                
                # Update the task
                task.status = result.status
                task.result = result.result
                task.error = result.error
                task.completed_at = time.time()
        
        except Exception as e:
            log.error(f"Error handling task result message: {e}")
    
    def _handle_task_cancel(self, message: Message) -> None:
        """
        Handle a task cancel message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the task ID
            task_id = message.payload.get("task_id")
            
            if not task_id:
                return
            
            with self.lock:
                # Get the task
                task = self.tasks.get(task_id)
                
                if not task:
                    return
                
                # Update the task
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = time.time()
        
        except Exception as e:
            log.error(f"Error handling task cancel message: {e}")


class WorkerManager:
    """
    Worker manager.
    
    This class is responsible for managing workers.
    """
    
    def __init__(self, broker_manager: Optional[BrokerManager] = None):
        """
        Initialize the worker manager.
        
        Args:
            broker_manager: The broker manager.
        """
        self.broker_manager = broker_manager or get_broker_manager()
        self.workers: Dict[str, Worker] = {}
        self.lock = threading.RLock()
        
        # Subscribe to worker topics
        self.broker_manager.subscribe("worker.register", self._handle_worker_register)
        self.broker_manager.subscribe("worker.heartbeat", self._handle_worker_heartbeat)
        self.broker_manager.subscribe("worker.task", self._handle_worker_task)
    
    def register_worker(self, name: str, capabilities: List[str]) -> Worker:
        """
        Register a worker.
        
        Args:
            name: The worker name.
            capabilities: The worker capabilities.
            
        Returns:
            The registered worker.
        """
        with self.lock:
            # Create a worker
            worker = Worker.create(name, capabilities)
            
            # Store the worker
            self.workers[worker.id] = worker
            
            # Publish a worker register message
            self.broker_manager.publish(
                topic="worker.register",
                payload=worker.to_dict(),
            )
            
            return worker
    
    def get_worker(self, worker_id: str) -> Optional[Worker]:
        """
        Get a worker by ID.
        
        Args:
            worker_id: The worker ID.
            
        Returns:
            The worker, or None if the worker is not found.
        """
        with self.lock:
            return self.workers.get(worker_id)
    
    def get_workers(self) -> List[Worker]:
        """
        Get all workers.
        
        Returns:
            A list of workers.
        """
        with self.lock:
            return list(self.workers.values())
    
    def send_heartbeat(self, worker_id: str) -> bool:
        """
        Send a heartbeat for a worker.
        
        Args:
            worker_id: The worker ID.
            
        Returns:
            True if the heartbeat was sent, False otherwise.
        """
        with self.lock:
            worker = self.workers.get(worker_id)
            
            if not worker:
                return False
            
            # Update the worker
            worker.last_heartbeat = time.time()
            
            # Publish a worker heartbeat message
            self.broker_manager.publish(
                topic="worker.heartbeat",
                payload={"worker_id": worker_id},
            )
            
            return True
    
    def assign_task(self, worker_id: str, task_id: str) -> bool:
        """
        Assign a task to a worker.
        
        Args:
            worker_id: The worker ID.
            task_id: The task ID.
            
        Returns:
            True if the task was assigned, False otherwise.
        """
        with self.lock:
            worker = self.workers.get(worker_id)
            
            if not worker:
                return False
            
            if worker.status != "idle":
                return False
            
            # Update the worker
            worker.status = "busy"
            worker.task_id = task_id
            
            # Publish a worker task message
            self.broker_manager.publish(
                topic="worker.task",
                payload={
                    "worker_id": worker_id,
                    "task_id": task_id,
                },
            )
            
            return True
    
    def complete_task(self, worker_id: str) -> bool:
        """
        Complete a task for a worker.
        
        Args:
            worker_id: The worker ID.
            
        Returns:
            True if the task was completed, False otherwise.
        """
        with self.lock:
            worker = self.workers.get(worker_id)
            
            if not worker:
                return False
            
            if worker.status != "busy":
                return False
            
            # Update the worker
            worker.status = "idle"
            worker.task_id = None
            
            return True
    
    def _handle_worker_register(self, message: Message) -> None:
        """
        Handle a worker register message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the worker
            worker_dict = message.payload
            worker = Worker.from_dict(worker_dict)
            
            with self.lock:
                # Store the worker
                self.workers[worker.id] = worker
        
        except Exception as e:
            log.error(f"Error handling worker register message: {e}")
    
    def _handle_worker_heartbeat(self, message: Message) -> None:
        """
        Handle a worker heartbeat message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the worker ID
            worker_id = message.payload.get("worker_id")
            
            if not worker_id:
                return
            
            with self.lock:
                # Get the worker
                worker = self.workers.get(worker_id)
                
                if not worker:
                    return
                
                # Update the worker
                worker.last_heartbeat = time.time()
        
        except Exception as e:
            log.error(f"Error handling worker heartbeat message: {e}")
    
    def _handle_worker_task(self, message: Message) -> None:
        """
        Handle a worker task message.
        
        Args:
            message: The message.
        """
        try:
            # Parse the worker ID and task ID
            worker_id = message.payload.get("worker_id")
            task_id = message.payload.get("task_id")
            
            if not worker_id or not task_id:
                return
            
            with self.lock:
                # Get the worker
                worker = self.workers.get(worker_id)
                
                if not worker:
                    return
                
                # Update the worker
                worker.status = "busy"
                worker.task_id = task_id
        
        except Exception as e:
            log.error(f"Error handling worker task message: {e}")


# Global task manager instance
_task_manager = None
_task_manager_lock = threading.RLock()


def get_task_manager() -> TaskManager:
    """
    Get the global task manager instance.
    
    Returns:
        The global task manager instance.
    """
    global _task_manager
    
    with _task_manager_lock:
        if _task_manager is None:
            _task_manager = TaskManager()
        
        return _task_manager


# Global worker manager instance
_worker_manager = None
_worker_manager_lock = threading.RLock()


def get_worker_manager() -> WorkerManager:
    """
    Get the global worker manager instance.
    
    Returns:
        The global worker manager instance.
    """
    global _worker_manager
    
    with _worker_manager_lock:
        if _worker_manager is None:
            _worker_manager = WorkerManager()
        
        return _worker_manager
