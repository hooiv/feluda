"""
Distributed scheduler for Feluda.

This module provides a scheduler for distributed computing.
"""

import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.distributed.backend import DistributedBackend, get_distributed_backend
from feluda.distributed.cluster import ClusterManager, get_cluster_manager
from feluda.distributed.worker import Task, TaskManager, TaskStatus, get_task_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class SchedulerPolicy(str, enum.Enum):
    """Enum for scheduler policies."""
    
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    LOCALITY_AWARE = "locality_aware"


class DistributedTask(BaseModel):
    """
    Distributed task.
    
    This class represents a task in the distributed scheduler.
    """
    
    id: str = Field(..., description="The task ID")
    function: str = Field(..., description="The function name")
    args: List[Any] = Field(default_factory=list, description="The function arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="The function keyword arguments")
    status: TaskStatus = Field(TaskStatus.PENDING, description="The task status")
    worker_id: Optional[str] = Field(None, description="The worker ID")
    node_id: Optional[str] = Field(None, description="The node ID")
    priority: int = Field(0, description="The task priority")
    dependencies: List[str] = Field(default_factory=list, description="The task dependencies")
    result: Optional[Any] = Field(None, description="The task result")
    error: Optional[str] = Field(None, description="The task error")
    created_at: float = Field(..., description="The creation timestamp")
    started_at: Optional[float] = Field(None, description="The start timestamp")
    completed_at: Optional[float] = Field(None, description="The completion timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        
        Returns:
            A dictionary representation of the task.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedTask":
        """
        Create a task from a dictionary.
        
        Args:
            data: The dictionary to create the task from.
            
        Returns:
            A task.
        """
        return cls(**data)
    
    @classmethod
    def create(
        cls,
        function: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        priority: int = 0,
        dependencies: List[str] = None,
    ) -> "DistributedTask":
        """
        Create a new task.
        
        Args:
            function: The function name.
            args: The function arguments.
            kwargs: The function keyword arguments.
            priority: The task priority.
            dependencies: The task dependencies.
            
        Returns:
            A new task.
        """
        return cls(
            id=str(uuid.uuid4()),
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=dependencies or [],
            created_at=time.time(),
        )


class DistributedScheduler:
    """
    Distributed scheduler.
    
    This class is responsible for scheduling tasks in a distributed environment.
    """
    
    def __init__(
        self,
        task_manager: Optional[TaskManager] = None,
        cluster_manager: Optional[ClusterManager] = None,
        backend: Optional[DistributedBackend] = None,
        policy: SchedulerPolicy = SchedulerPolicy.ROUND_ROBIN,
    ):
        """
        Initialize the distributed scheduler.
        
        Args:
            task_manager: The task manager.
            cluster_manager: The cluster manager.
            backend: The distributed backend.
            policy: The scheduler policy.
        """
        self.task_manager = task_manager or get_task_manager()
        self.cluster_manager = cluster_manager or get_cluster_manager()
        self.backend = backend or get_distributed_backend()
        self.policy = policy
        self.tasks: Dict[str, DistributedTask] = {}
        self.pending_tasks: List[str] = []
        self.running_tasks: Dict[str, str] = {}  # task_id -> worker_id
        self.completed_tasks: Set[str] = set()
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        
        # Initialize the backend
        self.backend.initialize({})
    
    def submit(
        self,
        function: str,
        *args: Any,
        **kwargs: Any,
        priority: int = 0,
        dependencies: List[str] = None,
    ) -> DistributedTask:
        """
        Submit a task.
        
        Args:
            function: The function name.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            priority: The task priority.
            dependencies: The task dependencies.
            
        Returns:
            The submitted task.
        """
        with self.lock:
            # Create a task
            task = DistributedTask.create(
                function=function,
                args=list(args),
                kwargs=kwargs,
                priority=priority,
                dependencies=dependencies,
            )
            
            # Store the task
            self.tasks[task.id] = task
            
            # Add the task to the pending tasks
            self._add_pending_task(task.id)
            
            return task
    
    def _add_pending_task(self, task_id: str) -> None:
        """
        Add a task to the pending tasks.
        
        Args:
            task_id: The task ID.
        """
        # Check if the task is already in the pending tasks
        if task_id in self.pending_tasks:
            return
        
        # Get the task
        task = self.tasks.get(task_id)
        
        if not task:
            return
        
        # Check if the task is ready to run
        if not self._is_task_ready(task_id):
            return
        
        # Add the task to the pending tasks
        self.pending_tasks.append(task_id)
        
        # Sort the pending tasks by priority
        self.pending_tasks.sort(
            key=lambda tid: self.tasks[tid].priority,
            reverse=True,
        )
    
    def _is_task_ready(self, task_id: str) -> bool:
        """
        Check if a task is ready to run.
        
        Args:
            task_id: The task ID.
            
        Returns:
            True if the task is ready to run, False otherwise.
        """
        # Get the task
        task = self.tasks.get(task_id)
        
        if not task:
            return False
        
        # Check if the task is already running or completed
        if task.status != TaskStatus.PENDING:
            return False
        
        # Check if all dependencies are completed
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            
            if not dep_task or dep_task.status != TaskStatus.SUCCEEDED:
                return False
        
        return True
    
    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task, or None if the task is not found.
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_tasks(self) -> List[DistributedTask]:
        """
        Get all tasks.
        
        Returns:
            A list of tasks.
        """
        with self.lock:
            return list(self.tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: The task ID.
            
        Returns:
            True if the task was cancelled, False otherwise.
        """
        with self.lock:
            # Get the task
            task = self.tasks.get(task_id)
            
            if not task:
                return False
            
            # Check if the task is already completed
            if task.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False
            
            # Cancel the task
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            
            # Remove the task from the pending tasks
            if task_id in self.pending_tasks:
                self.pending_tasks.remove(task_id)
            
            # Remove the task from the running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            # Add the task to the completed tasks
            self.completed_tasks.add(task_id)
            
            return True
    
    def start(self) -> None:
        """
        Start the scheduler.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the scheduler.
        """
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            if self.thread:
                self.thread.join()
                self.thread = None
    
    def _run(self) -> None:
        """
        Run the scheduler.
        """
        while self.running:
            try:
                # Schedule tasks
                self._schedule_tasks()
                
                # Check running tasks
                self._check_running_tasks()
                
                # Sleep for a while
                time.sleep(1)
            
            except Exception as e:
                log.error(f"Error in distributed scheduler: {e}")
                time.sleep(1)
    
    def _schedule_tasks(self) -> None:
        """
        Schedule tasks.
        """
        with self.lock:
            # Get available workers
            workers = self.task_manager.get_workers()
            
            if not workers:
                return
            
            # Get pending tasks
            pending_tasks = self.pending_tasks.copy()
            
            if not pending_tasks:
                return
            
            # Schedule tasks
            for task_id in pending_tasks:
                # Get the task
                task = self.tasks.get(task_id)
                
                if not task:
                    continue
                
                # Get an available worker
                worker = self._get_worker(workers, task)
                
                if not worker:
                    continue
                
                # Assign the task to the worker
                self._assign_task(task, worker)
    
    def _get_worker(self, workers: List[Any], task: DistributedTask) -> Optional[Any]:
        """
        Get a worker for a task.
        
        Args:
            workers: The available workers.
            task: The task.
            
        Returns:
            A worker, or None if no worker is available.
        """
        # Filter out busy workers
        available_workers = [w for w in workers if w.status == "idle"]
        
        if not available_workers:
            return None
        
        # Select a worker based on the policy
        if self.policy == SchedulerPolicy.ROUND_ROBIN:
            # Use the first available worker
            return available_workers[0]
        elif self.policy == SchedulerPolicy.LEAST_LOADED:
            # Use the worker with the least load
            return min(available_workers, key=lambda w: len(w.tasks))
        elif self.policy == SchedulerPolicy.RANDOM:
            # Use a random worker
            import random
            return random.choice(available_workers)
        elif self.policy == SchedulerPolicy.LOCALITY_AWARE:
            # Use a worker on the same node as the task's dependencies
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                
                if dep_task and dep_task.node_id:
                    for worker in available_workers:
                        if worker.node_id == dep_task.node_id:
                            return worker
            
            # Fall back to the first available worker
            return available_workers[0]
        
        # Default to the first available worker
        return available_workers[0]
    
    def _assign_task(self, task: DistributedTask, worker: Any) -> None:
        """
        Assign a task to a worker.
        
        Args:
            task: The task.
            worker: The worker.
        """
        # Update the task
        task.status = TaskStatus.RUNNING
        task.worker_id = worker.id
        task.node_id = worker.node_id
        task.started_at = time.time()
        
        # Remove the task from the pending tasks
        if task.id in self.pending_tasks:
            self.pending_tasks.remove(task.id)
        
        # Add the task to the running tasks
        self.running_tasks[task.id] = worker.id
        
        # Submit the task to the worker
        self.task_manager.submit_task(
            task_type=task.function,
            payload={
                "args": task.args,
                "kwargs": task.kwargs,
            },
        )
    
    def _check_running_tasks(self) -> None:
        """
        Check running tasks.
        """
        with self.lock:
            # Get running tasks
            running_tasks = self.running_tasks.copy()
            
            if not running_tasks:
                return
            
            # Check each task
            for task_id, worker_id in running_tasks.items():
                # Get the task
                task = self.tasks.get(task_id)
                
                if not task:
                    continue
                
                # Get the worker
                worker = self.task_manager.get_worker(worker_id)
                
                if not worker:
                    continue
                
                # Check if the task is completed
                if worker.task_id != task_id:
                    # The worker is no longer working on this task
                    # Check if the task is completed
                    task_result = self.task_manager.get_task_result(task_id)
                    
                    if task_result:
                        # Update the task
                        task.status = task_result.status
                        task.result = task_result.result
                        task.error = task_result.error
                        task.completed_at = time.time()
                        
                        # Remove the task from the running tasks
                        del self.running_tasks[task_id]
                        
                        # Add the task to the completed tasks
                        self.completed_tasks.add(task_id)
                        
                        # Check if any pending tasks depend on this task
                        self._check_dependencies()


# Global distributed scheduler instance
_distributed_scheduler = None
_distributed_scheduler_lock = threading.RLock()


def get_distributed_scheduler() -> DistributedScheduler:
    """
    Get the global distributed scheduler instance.
    
    Returns:
        The global distributed scheduler instance.
    """
    global _distributed_scheduler
    
    with _distributed_scheduler_lock:
        if _distributed_scheduler is None:
            _distributed_scheduler = DistributedScheduler()
            _distributed_scheduler.start()
        
        return _distributed_scheduler
