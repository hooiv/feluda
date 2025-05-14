"""
Workflow operators for Feluda.

This module provides operators for the workflow engine.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from feluda.observability import get_logger
from feluda.workflow.engine import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    Task,
    TaskDefinition,
)

log = get_logger(__name__)


class OperatorTask(Task):
    """
    Operator task.
    
    This class represents a task that executes an operator.
    """
    
    def __init__(self, definition: TaskDefinition, operator: Optional[Callable] = None):
        """
        Initialize an operator task.
        
        Args:
            definition: The task definition.
            operator: The operator to execute.
        """
        super().__init__(definition)
        self.operator = operator
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute the task.
        
        Args:
            context: The execution context.
            
        Returns:
            The execution result.
        """
        if not self.operator:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error="No operator specified",
            )
        
        try:
            # Execute the operator
            result = self.operator(context)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCEEDED,
                result=result,
            )
        
        except Exception as e:
            log.error(f"Error executing operator task {self.id}: {e}")
            
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
            )


class SequentialOperator:
    """
    Sequential operator.
    
    This operator executes a sequence of tasks in order.
    """
    
    def __init__(self, tasks: List[Callable]):
        """
        Initialize a sequential operator.
        
        Args:
            tasks: The tasks to execute.
        """
        self.tasks = tasks
    
    def __call__(self, context: ExecutionContext) -> Any:
        """
        Execute the operator.
        
        Args:
            context: The execution context.
            
        Returns:
            The result of the last task.
        """
        result = None
        
        for task in self.tasks:
            result = task(context)
            
            # Update the context with the result
            context.outputs[task.__name__] = result
        
        return result


class ParallelOperator:
    """
    Parallel operator.
    
    This operator executes a set of tasks in parallel.
    """
    
    def __init__(self, tasks: Dict[str, Callable]):
        """
        Initialize a parallel operator.
        
        Args:
            tasks: The tasks to execute.
        """
        self.tasks = tasks
    
    def __call__(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute the operator.
        
        Args:
            context: The execution context.
            
        Returns:
            A dictionary mapping task names to results.
        """
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the tasks
            futures = {
                name: executor.submit(task, context)
                for name, task in self.tasks.items()
            }
            
            # Wait for the tasks to complete
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    log.error(f"Error executing parallel task {name}: {e}")
                    results[name] = None
        
        return results


class BranchOperator:
    """
    Branch operator.
    
    This operator executes one of several tasks based on a condition.
    """
    
    def __init__(self, condition: Callable[[ExecutionContext], str], branches: Dict[str, Callable]):
        """
        Initialize a branch operator.
        
        Args:
            condition: The condition to evaluate.
            branches: The branches to execute.
        """
        self.condition = condition
        self.branches = branches
    
    def __call__(self, context: ExecutionContext) -> Any:
        """
        Execute the operator.
        
        Args:
            context: The execution context.
            
        Returns:
            The result of the selected branch.
        """
        # Evaluate the condition
        branch = self.condition(context)
        
        # Execute the selected branch
        if branch in self.branches:
            return self.branches[branch](context)
        
        # No branch selected
        return None


class ConditionalOperator:
    """
    Conditional operator.
    
    This operator executes a task if a condition is met.
    """
    
    def __init__(self, condition: Callable[[ExecutionContext], bool], task: Callable, else_task: Optional[Callable] = None):
        """
        Initialize a conditional operator.
        
        Args:
            condition: The condition to evaluate.
            task: The task to execute if the condition is met.
            else_task: The task to execute if the condition is not met.
        """
        self.condition = condition
        self.task = task
        self.else_task = else_task
    
    def __call__(self, context: ExecutionContext) -> Any:
        """
        Execute the operator.
        
        Args:
            context: The execution context.
            
        Returns:
            The result of the task.
        """
        # Evaluate the condition
        if self.condition(context):
            return self.task(context)
        elif self.else_task:
            return self.else_task(context)
        
        # No task executed
        return None


class MapOperator:
    """
    Map operator.
    
    This operator applies a function to each item in a collection.
    """
    
    def __init__(self, function: Callable[[Any], Any], collection_key: str):
        """
        Initialize a map operator.
        
        Args:
            function: The function to apply.
            collection_key: The key of the collection in the context inputs.
        """
        self.function = function
        self.collection_key = collection_key
    
    def __call__(self, context: ExecutionContext) -> List[Any]:
        """
        Execute the operator.
        
        Args:
            context: The execution context.
            
        Returns:
            A list of results.
        """
        # Get the collection
        collection = context.inputs.get(self.collection_key, [])
        
        if not collection:
            return []
        
        # Apply the function to each item
        return [self.function(item) for item in collection]


class ReduceOperator:
    """
    Reduce operator.
    
    This operator reduces a collection to a single value.
    """
    
    def __init__(self, function: Callable[[Any, Any], Any], collection_key: str, initial_value: Optional[Any] = None):
        """
        Initialize a reduce operator.
        
        Args:
            function: The function to apply.
            collection_key: The key of the collection in the context inputs.
            initial_value: The initial value for the reduction.
        """
        self.function = function
        self.collection_key = collection_key
        self.initial_value = initial_value
    
    def __call__(self, context: ExecutionContext) -> Any:
        """
        Execute the operator.
        
        Args:
            context: The execution context.
            
        Returns:
            The reduced value.
        """
        # Get the collection
        collection = context.inputs.get(self.collection_key, [])
        
        if not collection:
            return self.initial_value
        
        # Reduce the collection
        if self.initial_value is not None:
            result = self.initial_value
            
            for item in collection:
                result = self.function(result, item)
            
            return result
        else:
            result = collection[0]
            
            for item in collection[1:]:
                result = self.function(result, item)
            
            return result


class JoinOperator:
    """
    Join operator.
    
    This operator joins the results of multiple tasks.
    """
    
    def __init__(self, task_keys: List[str], join_function: Optional[Callable[[Dict[str, Any]], Any]] = None):
        """
        Initialize a join operator.
        
        Args:
            task_keys: The keys of the tasks to join.
            join_function: The function to join the results.
        """
        self.task_keys = task_keys
        self.join_function = join_function
    
    def __call__(self, context: ExecutionContext) -> Any:
        """
        Execute the operator.
        
        Args:
            context: The execution context.
            
        Returns:
            The joined result.
        """
        # Get the task results
        results = {
            key: context.inputs.get(key)
            for key in self.task_keys
            if key in context.inputs
        }
        
        # Join the results
        if self.join_function:
            return self.join_function(results)
        
        return results
