"""
Workflow scheduler for Feluda.

This module provides a scheduler for the workflow engine.
"""

import abc
import datetime
import enum
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.observability import get_logger
from feluda.workflow.engine import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    Task,
    TaskStatus,
    Workflow,
    WorkflowEngine,
    WorkflowStatus,
    get_workflow_engine,
)

log = get_logger(__name__)


class SchedulerPolicy(str, enum.Enum):
    """Enum for scheduler policies."""
    
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    PRIORITY = "priority"  # Priority-based
    DEADLINE = "deadline"  # Deadline-based


class ScheduledWorkflow(BaseModel):
    """
    Scheduled workflow.
    
    This class represents a scheduled workflow.
    """
    
    id: str = Field(..., description="The scheduled workflow ID")
    workflow_id: str = Field(..., description="The workflow ID")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="The workflow inputs")
    priority: int = Field(0, description="The workflow priority")
    deadline: Optional[datetime.datetime] = Field(None, description="The workflow deadline")
    schedule_time: datetime.datetime = Field(..., description="The schedule time")
    start_time: Optional[datetime.datetime] = Field(None, description="The start time")
    end_time: Optional[datetime.datetime] = Field(None, description="The end time")
    status: WorkflowStatus = Field(WorkflowStatus.PENDING, description="The workflow status")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the scheduled workflow to a dictionary.
        
        Returns:
            A dictionary representation of the scheduled workflow.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledWorkflow":
        """
        Create a scheduled workflow from a dictionary.
        
        Args:
            data: The dictionary to create the scheduled workflow from.
            
        Returns:
            A scheduled workflow.
        """
        return cls(**data)


class Scheduler:
    """
    Workflow scheduler.
    
    This class is responsible for scheduling and executing workflows.
    """
    
    def __init__(
        self,
        workflow_engine: Optional[WorkflowEngine] = None,
        policy: SchedulerPolicy = SchedulerPolicy.FIFO,
        max_concurrent_workflows: int = 10,
    ):
        """
        Initialize the scheduler.
        
        Args:
            workflow_engine: The workflow engine.
            policy: The scheduler policy.
            max_concurrent_workflows: The maximum number of concurrent workflows.
        """
        self.workflow_engine = workflow_engine or get_workflow_engine()
        self.policy = policy
        self.max_concurrent_workflows = max_concurrent_workflows
        self.scheduled_workflows: Dict[str, ScheduledWorkflow] = {}
        self.running_workflows: Set[str] = set()
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
    
    def schedule_workflow(
        self,
        workflow_id: str,
        inputs: Dict[str, Any],
        priority: int = 0,
        deadline: Optional[datetime.datetime] = None,
        schedule_time: Optional[datetime.datetime] = None,
    ) -> str:
        """
        Schedule a workflow.
        
        Args:
            workflow_id: The workflow ID.
            inputs: The workflow inputs.
            priority: The workflow priority.
            deadline: The workflow deadline.
            schedule_time: The schedule time. If None, the workflow is scheduled immediately.
            
        Returns:
            The scheduled workflow ID.
        """
        with self.lock:
            # Create the scheduled workflow
            scheduled_workflow = ScheduledWorkflow(
                id=f"{workflow_id}:{time.time()}",
                workflow_id=workflow_id,
                inputs=inputs,
                priority=priority,
                deadline=deadline,
                schedule_time=schedule_time or datetime.datetime.now(),
                status=WorkflowStatus.PENDING,
            )
            
            # Store the scheduled workflow
            self.scheduled_workflows[scheduled_workflow.id] = scheduled_workflow
            
            return scheduled_workflow.id
    
    def cancel_workflow(self, scheduled_workflow_id: str) -> bool:
        """
        Cancel a scheduled workflow.
        
        Args:
            scheduled_workflow_id: The scheduled workflow ID.
            
        Returns:
            True if the workflow was cancelled, False otherwise.
        """
        with self.lock:
            # Get the scheduled workflow
            scheduled_workflow = self.scheduled_workflows.get(scheduled_workflow_id)
            
            if not scheduled_workflow:
                return False
            
            # Check if the workflow is running
            if scheduled_workflow_id in self.running_workflows:
                # Cancel the workflow
                self.workflow_engine.cancel_workflow(scheduled_workflow.workflow_id)
            
            # Update the scheduled workflow status
            scheduled_workflow.status = WorkflowStatus.CANCELLED
            scheduled_workflow.end_time = datetime.datetime.now()
            
            return True
    
    def get_scheduled_workflow(self, scheduled_workflow_id: str) -> Optional[ScheduledWorkflow]:
        """
        Get a scheduled workflow.
        
        Args:
            scheduled_workflow_id: The scheduled workflow ID.
            
        Returns:
            The scheduled workflow, or None if the workflow is not found.
        """
        with self.lock:
            return self.scheduled_workflows.get(scheduled_workflow_id)
    
    def get_scheduled_workflows(self) -> Dict[str, ScheduledWorkflow]:
        """
        Get all scheduled workflows.
        
        Returns:
            A dictionary mapping scheduled workflow IDs to scheduled workflows.
        """
        with self.lock:
            return self.scheduled_workflows.copy()
    
    def start(self) -> None:
        """
        Start the scheduler.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
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
                # Get the next workflows to execute
                next_workflows = self._get_next_workflows()
                
                # Execute the workflows
                for scheduled_workflow_id in next_workflows:
                    self._execute_workflow(scheduled_workflow_id)
                
                # Sleep for a while
                time.sleep(1)
            
            except Exception as e:
                log.error(f"Error in scheduler: {e}")
                time.sleep(1)
    
    def _get_next_workflows(self) -> List[str]:
        """
        Get the next workflows to execute.
        
        Returns:
            A list of scheduled workflow IDs.
        """
        with self.lock:
            # Check if we can execute more workflows
            if len(self.running_workflows) >= self.max_concurrent_workflows:
                return []
            
            # Get the pending workflows
            pending_workflows = [
                scheduled_workflow_id
                for scheduled_workflow_id, scheduled_workflow in self.scheduled_workflows.items()
                if scheduled_workflow.status == WorkflowStatus.PENDING
                and scheduled_workflow.schedule_time <= datetime.datetime.now()
            ]
            
            # Sort the workflows based on the policy
            if self.policy == SchedulerPolicy.FIFO:
                # Sort by schedule time (oldest first)
                pending_workflows.sort(
                    key=lambda scheduled_workflow_id: self.scheduled_workflows[scheduled_workflow_id].schedule_time
                )
            elif self.policy == SchedulerPolicy.LIFO:
                # Sort by schedule time (newest first)
                pending_workflows.sort(
                    key=lambda scheduled_workflow_id: self.scheduled_workflows[scheduled_workflow_id].schedule_time,
                    reverse=True,
                )
            elif self.policy == SchedulerPolicy.PRIORITY:
                # Sort by priority (highest first)
                pending_workflows.sort(
                    key=lambda scheduled_workflow_id: self.scheduled_workflows[scheduled_workflow_id].priority,
                    reverse=True,
                )
            elif self.policy == SchedulerPolicy.DEADLINE:
                # Sort by deadline (earliest first)
                pending_workflows.sort(
                    key=lambda scheduled_workflow_id: (
                        self.scheduled_workflows[scheduled_workflow_id].deadline or datetime.datetime.max
                    )
                )
            
            # Limit the number of workflows to execute
            max_workflows = self.max_concurrent_workflows - len(self.running_workflows)
            return pending_workflows[:max_workflows]
    
    def _execute_workflow(self, scheduled_workflow_id: str) -> None:
        """
        Execute a workflow.
        
        Args:
            scheduled_workflow_id: The scheduled workflow ID.
        """
        with self.lock:
            # Get the scheduled workflow
            scheduled_workflow = self.scheduled_workflows.get(scheduled_workflow_id)
            
            if not scheduled_workflow:
                return
            
            # Update the scheduled workflow status
            scheduled_workflow.status = WorkflowStatus.RUNNING
            scheduled_workflow.start_time = datetime.datetime.now()
            
            # Add the workflow to the running workflows
            self.running_workflows.add(scheduled_workflow_id)
        
        # Execute the workflow in a separate thread
        threading.Thread(
            target=self._execute_workflow_thread,
            args=(scheduled_workflow_id,),
            daemon=True,
        ).start()
    
    def _execute_workflow_thread(self, scheduled_workflow_id: str) -> None:
        """
        Execute a workflow in a separate thread.
        
        Args:
            scheduled_workflow_id: The scheduled workflow ID.
        """
        try:
            # Get the scheduled workflow
            scheduled_workflow = self.get_scheduled_workflow(scheduled_workflow_id)
            
            if not scheduled_workflow:
                return
            
            # Execute the workflow
            outputs = self.workflow_engine.execute_workflow(
                scheduled_workflow.workflow_id,
                scheduled_workflow.inputs,
            )
            
            # Update the scheduled workflow status
            with self.lock:
                scheduled_workflow = self.scheduled_workflows.get(scheduled_workflow_id)
                
                if scheduled_workflow:
                    scheduled_workflow.status = WorkflowStatus.SUCCEEDED
                    scheduled_workflow.end_time = datetime.datetime.now()
        
        except Exception as e:
            log.error(f"Error executing workflow {scheduled_workflow_id}: {e}")
            
            # Update the scheduled workflow status
            with self.lock:
                scheduled_workflow = self.scheduled_workflows.get(scheduled_workflow_id)
                
                if scheduled_workflow:
                    scheduled_workflow.status = WorkflowStatus.FAILED
                    scheduled_workflow.end_time = datetime.datetime.now()
        
        finally:
            # Remove the workflow from the running workflows
            with self.lock:
                self.running_workflows.discard(scheduled_workflow_id)


# Global scheduler instance
_scheduler = None
_scheduler_lock = threading.RLock()


def get_scheduler() -> Scheduler:
    """
    Get the global scheduler instance.
    
    Returns:
        The global scheduler instance.
    """
    global _scheduler
    
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = Scheduler()
            _scheduler.start()
        
        return _scheduler
