"""
Workflow executor for Feluda.

This module provides executors for the workflow engine.
"""

import abc
import concurrent.futures
import enum
import logging
import multiprocessing
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from feluda.observability import get_logger
from feluda.workflow.engine import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    Task,
)

log = get_logger(__name__)


class ExecutorType(str, enum.Enum):
    """Enum for executor types."""
    
    LOCAL = "local"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    DISTRIBUTED = "distributed"


class Executor(abc.ABC):
    """
    Base class for executors.
    
    This class defines the interface for executors.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def execute(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute a task.
        
        Args:
            task: The task to execute.
            context: The execution context.
            
        Returns:
            The execution result.
        """
        pass
    
    @abc.abstractmethod
    def execute_function(
        self,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function.
        
        Args:
            function: The function to execute.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            
        Returns:
            The function result.
        """
        pass
    
    @abc.abstractmethod
    def execute_map(
        self,
        function: Callable,
        items: List[Any],
    ) -> List[Any]:
        """
        Execute a function on a list of items.
        
        Args:
            function: The function to execute.
            items: The items to process.
            
        Returns:
            The function results.
        """
        pass


class LocalExecutor(Executor):
    """
    Local executor.
    
    This executor executes tasks locally in the current thread.
    """
    
    def execute(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute a task.
        
        Args:
            task: The task to execute.
            context: The execution context.
            
        Returns:
            The execution result.
        """
        try:
            return task.execute(context)
        
        except Exception as e:
            log.error(f"Error executing task {task.id}: {e}")
            
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
            )
    
    def execute_function(
        self,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function.
        
        Args:
            function: The function to execute.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            
        Returns:
            The function result.
        """
        return function(*args, **kwargs)
    
    def execute_map(
        self,
        function: Callable,
        items: List[Any],
    ) -> List[Any]:
        """
        Execute a function on a list of items.
        
        Args:
            function: The function to execute.
            items: The items to process.
            
        Returns:
            The function results.
        """
        return [function(item) for item in items]


class ThreadPoolExecutor(Executor):
    """
    Thread pool executor.
    
    This executor executes tasks in a thread pool.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize a thread pool executor.
        
        Args:
            max_workers: The maximum number of worker threads.
        """
        self.max_workers = max_workers
    
    def execute(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute a task.
        
        Args:
            task: The task to execute.
            context: The execution context.
            
        Returns:
            The execution result.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.submit(task.execute, context)
            
            try:
                return future.result()
            
            except Exception as e:
                log.error(f"Error executing task {task.id}: {e}")
                
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                )
    
    def execute_function(
        self,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function.
        
        Args:
            function: The function to execute.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            
        Returns:
            The function result.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.submit(function, *args, **kwargs)
            return future.result()
    
    def execute_map(
        self,
        function: Callable,
        items: List[Any],
    ) -> List[Any]:
        """
        Execute a function on a list of items.
        
        Args:
            function: The function to execute.
            items: The items to process.
            
        Returns:
            The function results.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(function, items))


class ProcessPoolExecutor(Executor):
    """
    Process pool executor.
    
    This executor executes tasks in a process pool.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize a process pool executor.
        
        Args:
            max_workers: The maximum number of worker processes.
        """
        self.max_workers = max_workers
    
    def execute(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute a task.
        
        Args:
            task: The task to execute.
            context: The execution context.
            
        Returns:
            The execution result.
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.submit(task.execute, context)
            
            try:
                return future.result()
            
            except Exception as e:
                log.error(f"Error executing task {task.id}: {e}")
                
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                )
    
    def execute_function(
        self,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function.
        
        Args:
            function: The function to execute.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            
        Returns:
            The function result.
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.submit(function, *args, **kwargs)
            return future.result()
    
    def execute_map(
        self,
        function: Callable,
        items: List[Any],
    ) -> List[Any]:
        """
        Execute a function on a list of items.
        
        Args:
            function: The function to execute.
            items: The items to process.
            
        Returns:
            The function results.
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(function, items))


class DistributedExecutor(Executor):
    """
    Distributed executor.
    
    This executor executes tasks in a distributed environment.
    """
    
    def __init__(self, backend: str = "dask"):
        """
        Initialize a distributed executor.
        
        Args:
            backend: The distributed backend to use.
        """
        self.backend = backend
        
        if backend == "dask":
            try:
                import dask.distributed
                self.client = dask.distributed.Client()
            except ImportError:
                log.error("Dask is not installed. Please install dask[distributed].")
                raise
        else:
            raise ValueError(f"Unsupported distributed backend: {backend}")
    
    def execute(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute a task.
        
        Args:
            task: The task to execute.
            context: The execution context.
            
        Returns:
            The execution result.
        """
        if self.backend == "dask":
            future = self.client.submit(task.execute, context)
            
            try:
                return future.result()
            
            except Exception as e:
                log.error(f"Error executing task {task.id}: {e}")
                
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                )
        
        raise ValueError(f"Unsupported distributed backend: {self.backend}")
    
    def execute_function(
        self,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function.
        
        Args:
            function: The function to execute.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            
        Returns:
            The function result.
        """
        if self.backend == "dask":
            future = self.client.submit(function, *args, **kwargs)
            return future.result()
        
        raise ValueError(f"Unsupported distributed backend: {self.backend}")
    
    def execute_map(
        self,
        function: Callable,
        items: List[Any],
    ) -> List[Any]:
        """
        Execute a function on a list of items.
        
        Args:
            function: The function to execute.
            items: The items to process.
            
        Returns:
            The function results.
        """
        if self.backend == "dask":
            futures = self.client.map(function, items)
            return self.client.gather(futures)
        
        raise ValueError(f"Unsupported distributed backend: {self.backend}")


# Dictionary of executors
_executors: Dict[ExecutorType, Executor] = {
    ExecutorType.LOCAL: LocalExecutor(),
    ExecutorType.THREAD_POOL: ThreadPoolExecutor(),
    ExecutorType.PROCESS_POOL: ProcessPoolExecutor(),
}


def get_executor(executor_type: ExecutorType = ExecutorType.LOCAL) -> Executor:
    """
    Get an executor.
    
    Args:
        executor_type: The executor type.
        
    Returns:
        An executor.
    """
    if executor_type not in _executors:
        if executor_type == ExecutorType.DISTRIBUTED:
            _executors[executor_type] = DistributedExecutor()
        else:
            raise ValueError(f"Unsupported executor type: {executor_type}")
    
    return _executors[executor_type]
