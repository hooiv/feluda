"""
Distributed computing backends for Feluda.

This module provides backends for distributed computing.
"""

import abc
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from feluda.config import get_config
from feluda.distributed.cluster import ClusterManager, get_cluster_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class DistributedBackend(abc.ABC):
    """
    Base class for distributed computing backends.
    
    This class defines the interface for distributed computing backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the backend.
        
        Args:
            config: The backend configuration.
        """
        pass
    
    @abc.abstractmethod
    def shutdown(self) -> None:
        """
        Shut down the backend.
        """
        pass
    
    @abc.abstractmethod
    def submit(
        self,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Submit a function for execution.
        
        Args:
            function: The function to execute.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            
        Returns:
            A future representing the function execution.
        """
        pass
    
    @abc.abstractmethod
    def map(
        self,
        function: Callable,
        *iterables: Any,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Apply a function to each element of the iterables.
        
        Args:
            function: The function to apply.
            *iterables: The iterables to process.
            **kwargs: Additional keyword arguments.
            
        Returns:
            A list of results.
        """
        pass
    
    @abc.abstractmethod
    def get_resources(self) -> Dict[str, Any]:
        """
        Get the available resources.
        
        Returns:
            A dictionary of resources.
        """
        pass


class DaskBackend(DistributedBackend):
    """
    Dask distributed computing backend.
    
    This class implements a distributed computing backend using Dask.
    """
    
    def __init__(self):
        """
        Initialize the Dask backend.
        """
        self.client = None
        self.lock = threading.RLock()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the backend.
        
        Args:
            config: The backend configuration.
        """
        with self.lock:
            if self.client:
                return
            
            try:
                import dask.distributed
                
                # Get the scheduler address
                scheduler_address = config.get("scheduler_address")
                
                if scheduler_address:
                    # Connect to an existing scheduler
                    self.client = dask.distributed.Client(scheduler_address)
                else:
                    # Create a local cluster
                    n_workers = config.get("n_workers")
                    threads_per_worker = config.get("threads_per_worker")
                    memory_limit = config.get("memory_limit")
                    
                    cluster = dask.distributed.LocalCluster(
                        n_workers=n_workers,
                        threads_per_worker=threads_per_worker,
                        memory_limit=memory_limit,
                    )
                    
                    self.client = dask.distributed.Client(cluster)
            
            except ImportError:
                log.error("Dask is not installed. Please install dask[distributed].")
                raise
    
    def shutdown(self) -> None:
        """
        Shut down the backend.
        """
        with self.lock:
            if not self.client:
                return
            
            self.client.close()
            self.client = None
    
    def submit(
        self,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Submit a function for execution.
        
        Args:
            function: The function to execute.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            
        Returns:
            A future representing the function execution.
        """
        with self.lock:
            if not self.client:
                raise RuntimeError("Dask backend is not initialized")
            
            return self.client.submit(function, *args, **kwargs)
    
    def map(
        self,
        function: Callable,
        *iterables: Any,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Apply a function to each element of the iterables.
        
        Args:
            function: The function to apply.
            *iterables: The iterables to process.
            **kwargs: Additional keyword arguments.
            
        Returns:
            A list of results.
        """
        with self.lock:
            if not self.client:
                raise RuntimeError("Dask backend is not initialized")
            
            futures = self.client.map(function, *iterables, **kwargs)
            return self.client.gather(futures)
    
    def get_resources(self) -> Dict[str, Any]:
        """
        Get the available resources.
        
        Returns:
            A dictionary of resources.
        """
        with self.lock:
            if not self.client:
                raise RuntimeError("Dask backend is not initialized")
            
            return self.client.resources()


class RayBackend(DistributedBackend):
    """
    Ray distributed computing backend.
    
    This class implements a distributed computing backend using Ray.
    """
    
    def __init__(self):
        """
        Initialize the Ray backend.
        """
        self.initialized = False
        self.lock = threading.RLock()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the backend.
        
        Args:
            config: The backend configuration.
        """
        with self.lock:
            if self.initialized:
                return
            
            try:
                import ray
                
                # Get the address
                address = config.get("address")
                
                if address:
                    # Connect to an existing Ray cluster
                    ray.init(address=address)
                else:
                    # Create a local Ray cluster
                    num_cpus = config.get("num_cpus")
                    num_gpus = config.get("num_gpus")
                    
                    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
                
                self.initialized = True
            
            except ImportError:
                log.error("Ray is not installed. Please install ray.")
                raise
    
    def shutdown(self) -> None:
        """
        Shut down the backend.
        """
        with self.lock:
            if not self.initialized:
                return
            
            import ray
            ray.shutdown()
            self.initialized = False
    
    def submit(
        self,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Submit a function for execution.
        
        Args:
            function: The function to execute.
            *args: The function arguments.
            **kwargs: The function keyword arguments.
            
        Returns:
            A future representing the function execution.
        """
        with self.lock:
            if not self.initialized:
                raise RuntimeError("Ray backend is not initialized")
            
            import ray
            
            # Create a remote function
            remote_function = ray.remote(function)
            
            # Submit the function
            return remote_function.remote(*args, **kwargs)
    
    def map(
        self,
        function: Callable,
        *iterables: Any,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Apply a function to each element of the iterables.
        
        Args:
            function: The function to apply.
            *iterables: The iterables to process.
            **kwargs: Additional keyword arguments.
            
        Returns:
            A list of results.
        """
        with self.lock:
            if not self.initialized:
                raise RuntimeError("Ray backend is not initialized")
            
            import ray
            
            # Create a remote function
            remote_function = ray.remote(function)
            
            # Submit the function for each element
            futures = []
            
            for args in zip(*iterables):
                futures.append(remote_function.remote(*args, **kwargs))
            
            # Get the results
            return ray.get(futures)
    
    def get_resources(self) -> Dict[str, Any]:
        """
        Get the available resources.
        
        Returns:
            A dictionary of resources.
        """
        with self.lock:
            if not self.initialized:
                raise RuntimeError("Ray backend is not initialized")
            
            import ray
            return ray.available_resources()


# Dictionary of distributed backends
_distributed_backends: Dict[str, DistributedBackend] = {}
_distributed_backends_lock = threading.RLock()


def register_distributed_backend(name: str, backend: DistributedBackend) -> None:
    """
    Register a distributed backend.
    
    Args:
        name: The backend name.
        backend: The backend.
    """
    with _distributed_backends_lock:
        _distributed_backends[name] = backend


def get_distributed_backend(name: str = "dask") -> DistributedBackend:
    """
    Get a distributed backend.
    
    Args:
        name: The backend name.
        
    Returns:
        The distributed backend.
    """
    with _distributed_backends_lock:
        if name not in _distributed_backends:
            if name == "dask":
                _distributed_backends[name] = DaskBackend()
            elif name == "ray":
                _distributed_backends[name] = RayBackend()
            else:
                raise ValueError(f"Unknown distributed backend: {name}")
        
        return _distributed_backends[name]


# Register default backends
register_distributed_backend("dask", DaskBackend())
register_distributed_backend("ray", RayBackend())
