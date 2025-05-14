"""
Swarm module for Feluda.

This module provides agent swarm functionality for the Feluda framework.
"""

import abc
import enum
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.agents.agent import Agent, AgentAction, AgentGoal, AgentObservation, AgentState, get_agent_manager
from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class SwarmState(str, enum.Enum):
    """Enum for swarm states."""
    
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class SwarmMessage(BaseModel):
    """
    Swarm message.
    
    This class represents a message that can be sent between agents in a swarm.
    """
    
    id: str = Field(..., description="The message ID")
    sender_id: str = Field(..., description="The sender agent ID")
    recipient_id: Optional[str] = Field(None, description="The recipient agent ID")
    topic: str = Field(..., description="The message topic")
    content: Dict[str, Any] = Field(..., description="The message content")
    timestamp: float = Field(..., description="The message timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            A dictionary representation of the message.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmMessage":
        """
        Create a message from a dictionary.
        
        Args:
            data: The dictionary to create the message from.
            
        Returns:
            A message.
        """
        return cls(**data)


class SwarmTask(BaseModel):
    """
    Swarm task.
    
    This class represents a task that can be assigned to agents in a swarm.
    """
    
    id: str = Field(..., description="The task ID")
    name: str = Field(..., description="The task name")
    description: Optional[str] = Field(None, description="The task description")
    status: str = Field(..., description="The task status")
    priority: int = Field(0, description="The task priority")
    deadline: Optional[float] = Field(None, description="The task deadline")
    assigned_to: Optional[str] = Field(None, description="The agent ID the task is assigned to")
    dependencies: List[str] = Field(default_factory=list, description="The task dependencies")
    created_at: float = Field(..., description="The task creation timestamp")
    updated_at: float = Field(..., description="The task update timestamp")
    completed_at: Optional[float] = Field(None, description="The task completion timestamp")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="The task parameters")
    results: Optional[Dict[str, Any]] = Field(None, description="The task results")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        
        Returns:
            A dictionary representation of the task.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmTask":
        """
        Create a task from a dictionary.
        
        Args:
            data: The dictionary to create the task from.
            
        Returns:
            A task.
        """
        return cls(**data)


class AgentSwarm:
    """
    Agent swarm.
    
    This class represents a swarm of agents that can collaborate to achieve goals.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = None):
        """
        Initialize an agent swarm.
        
        Args:
            id: The swarm ID.
            name: The swarm name.
            config: The swarm configuration.
        """
        self.id = id
        self.name = name
        self.config = config or {}
        self.state = SwarmState.IDLE
        self.agent_ids: List[str] = []
        self.messages: List[SwarmMessage] = []
        self.tasks: Dict[str, SwarmTask] = {}
        self.lock = threading.RLock()
        self.thread = None
        self.running = False
    
    def add_agent(self, agent_id: str) -> bool:
        """
        Add an agent to the swarm.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            True if the agent was added, False otherwise.
        """
        with self.lock:
            # Check if the agent exists
            agent_manager = get_agent_manager()
            agent = agent_manager.get_agent(agent_id)
            
            if not agent:
                return False
            
            # Add the agent
            if agent_id not in self.agent_ids:
                self.agent_ids.append(agent_id)
            
            return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the swarm.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            True if the agent was removed, False otherwise.
        """
        with self.lock:
            if agent_id in self.agent_ids:
                self.agent_ids.remove(agent_id)
                return True
            
            return False
    
    def get_agents(self) -> List[str]:
        """
        Get the agents in the swarm.
        
        Returns:
            A list of agent IDs.
        """
        with self.lock:
            return self.agent_ids.copy()
    
    def start(self) -> None:
        """
        Start the swarm.
        """
        with self.lock:
            if self.state != SwarmState.IDLE and self.state != SwarmState.STOPPED:
                return
            
            self.state = SwarmState.RUNNING
            self.running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
            
            # Start all agents
            agent_manager = get_agent_manager()
            
            for agent_id in self.agent_ids:
                agent_manager.start_agent(agent_id)
    
    def stop(self) -> None:
        """
        Stop the swarm.
        """
        with self.lock:
            if self.state != SwarmState.RUNNING and self.state != SwarmState.PAUSED:
                return
            
            self.state = SwarmState.STOPPED
            self.running = False
            
            if self.thread:
                self.thread.join()
                self.thread = None
            
            # Stop all agents
            agent_manager = get_agent_manager()
            
            for agent_id in self.agent_ids:
                agent_manager.stop_agent(agent_id)
    
    def pause(self) -> None:
        """
        Pause the swarm.
        """
        with self.lock:
            if self.state != SwarmState.RUNNING:
                return
            
            self.state = SwarmState.PAUSED
            
            # Pause all agents
            agent_manager = get_agent_manager()
            
            for agent_id in self.agent_ids:
                agent_manager.pause_agent(agent_id)
    
    def resume(self) -> None:
        """
        Resume the swarm.
        """
        with self.lock:
            if self.state != SwarmState.PAUSED:
                return
            
            self.state = SwarmState.RUNNING
            
            # Resume all agents
            agent_manager = get_agent_manager()
            
            for agent_id in self.agent_ids:
                agent_manager.resume_agent(agent_id)
    
    def send_message(self, sender_id: str, topic: str, content: Dict[str, Any], recipient_id: Optional[str] = None) -> SwarmMessage:
        """
        Send a message.
        
        Args:
            sender_id: The sender agent ID.
            topic: The message topic.
            content: The message content.
            recipient_id: The recipient agent ID. If None, the message is broadcast to all agents.
            
        Returns:
            The sent message.
        """
        with self.lock:
            # Create the message
            message = SwarmMessage(
                id=str(uuid.uuid4()),
                sender_id=sender_id,
                recipient_id=recipient_id,
                topic=topic,
                content=content,
                timestamp=time.time(),
            )
            
            # Store the message
            self.messages.append(message)
            
            return message
    
    def get_messages(
        self,
        agent_id: Optional[str] = None,
        topic: Optional[str] = None,
        since: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[SwarmMessage]:
        """
        Get messages.
        
        Args:
            agent_id: The agent ID. If provided, only messages sent to or from this agent are returned.
            topic: The message topic. If provided, only messages with this topic are returned.
            since: The timestamp to get messages since. If provided, only messages after this timestamp are returned.
            limit: The maximum number of messages to return. If None, all matching messages are returned.
            
        Returns:
            A list of messages.
        """
        with self.lock:
            # Filter messages
            filtered_messages = []
            
            for message in self.messages:
                if agent_id and message.sender_id != agent_id and message.recipient_id != agent_id and message.recipient_id is not None:
                    continue
                
                if topic and message.topic != topic:
                    continue
                
                if since and message.timestamp < since:
                    continue
                
                filtered_messages.append(message)
            
            # Sort messages by timestamp
            filtered_messages.sort(key=lambda m: m.timestamp)
            
            # Limit the number of messages
            if limit is not None:
                filtered_messages = filtered_messages[-limit:]
            
            return filtered_messages
    
    def create_task(
        self,
        name: str,
        description: Optional[str] = None,
        priority: int = 0,
        deadline: Optional[float] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SwarmTask:
        """
        Create a task.
        
        Args:
            name: The task name.
            description: The task description.
            priority: The task priority.
            deadline: The task deadline.
            dependencies: The task dependencies.
            parameters: The task parameters.
            
        Returns:
            The created task.
        """
        with self.lock:
            # Create the task
            now = time.time()
            
            task = SwarmTask(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                status="pending",
                priority=priority,
                deadline=deadline,
                dependencies=dependencies or [],
                created_at=now,
                updated_at=now,
                parameters=parameters or {},
            )
            
            # Store the task
            self.tasks[task.id] = task
            
            return task
    
    def get_task(self, task_id: str) -> Optional[SwarmTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: The task ID.
            
        Returns:
            The task, or None if the task is not found.
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_tasks(
        self,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
        priority_min: Optional[int] = None,
        priority_max: Optional[int] = None,
    ) -> List[SwarmTask]:
        """
        Get tasks.
        
        Args:
            status: The task status. If provided, only tasks with this status are returned.
            agent_id: The agent ID. If provided, only tasks assigned to this agent are returned.
            priority_min: The minimum task priority. If provided, only tasks with a priority greater than or equal to this are returned.
            priority_max: The maximum task priority. If provided, only tasks with a priority less than or equal to this are returned.
            
        Returns:
            A list of tasks.
        """
        with self.lock:
            # Filter tasks
            filtered_tasks = []
            
            for task in self.tasks.values():
                if status and task.status != status:
                    continue
                
                if agent_id and task.assigned_to != agent_id:
                    continue
                
                if priority_min is not None and task.priority < priority_min:
                    continue
                
                if priority_max is not None and task.priority > priority_max:
                    continue
                
                filtered_tasks.append(task)
            
            # Sort tasks by priority (highest first) and creation time
            filtered_tasks.sort(key=lambda t: (-t.priority, t.created_at))
            
            return filtered_tasks
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """
        Assign a task to an agent.
        
        Args:
            task_id: The task ID.
            agent_id: The agent ID.
            
        Returns:
            True if the task was assigned, False otherwise.
        """
        with self.lock:
            # Get the task
            task = self.get_task(task_id)
            
            if not task:
                return False
            
            # Check if the agent exists
            if agent_id not in self.agent_ids:
                return False
            
            # Assign the task
            task.assigned_to = agent_id
            task.status = "assigned"
            task.updated_at = time.time()
            
            return True
    
    def update_task_status(self, task_id: str, status: str, results: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a task.
        
        Args:
            task_id: The task ID.
            status: The new task status.
            results: The task results. Only used if the status is "completed".
            
        Returns:
            True if the task status was updated, False otherwise.
        """
        with self.lock:
            # Get the task
            task = self.get_task(task_id)
            
            if not task:
                return False
            
            # Update the task status
            task.status = status
            task.updated_at = time.time()
            
            if status == "completed":
                task.completed_at = time.time()
                task.results = results
            
            return True
    
    def _run(self) -> None:
        """
        Run the swarm.
        """
        try:
            while self.running:
                with self.lock:
                    if self.state == SwarmState.PAUSED:
                        time.sleep(0.1)
                        continue
                    
                    if self.state != SwarmState.RUNNING:
                        break
                
                # Run the swarm
                try:
                    self._process_tasks()
                    self._process_messages()
                except Exception as e:
                    log.error(f"Error running swarm {self.name}: {e}")
                    with self.lock:
                        self.state = SwarmState.ERROR
                    
                    break
                
                # Sleep for a short time
                time.sleep(0.1)
        
        except Exception as e:
            log.error(f"Error in swarm {self.name} thread: {e}")
            with self.lock:
                self.state = SwarmState.ERROR
    
    def _process_tasks(self) -> None:
        """
        Process tasks.
        """
        with self.lock:
            # Get pending tasks
            pending_tasks = self.get_tasks(status="pending")
            
            # Assign tasks to agents
            for task in pending_tasks:
                # Check if the task has dependencies
                if task.dependencies:
                    # Check if all dependencies are completed
                    all_completed = True
                    
                    for dep_id in task.dependencies:
                        dep_task = self.get_task(dep_id)
                        
                        if not dep_task or dep_task.status != "completed":
                            all_completed = False
                            break
                    
                    if not all_completed:
                        continue
                
                # Find an available agent
                agent_manager = get_agent_manager()
                
                for agent_id in self.agent_ids:
                    agent = agent_manager.get_agent(agent_id)
                    
                    if not agent:
                        continue
                    
                    # Check if the agent is running
                    if agent.state != AgentState.RUNNING:
                        continue
                    
                    # Check if the agent has any assigned tasks
                    assigned_tasks = self.get_tasks(status="assigned", agent_id=agent_id)
                    
                    if assigned_tasks:
                        continue
                    
                    # Assign the task to the agent
                    self.assign_task(task.id, agent_id)
                    
                    # Notify the agent
                    self.send_message(
                        sender_id=self.id,
                        recipient_id=agent_id,
                        topic="task_assigned",
                        content={
                            "task_id": task.id,
                            "task_name": task.name,
                            "task_description": task.description,
                            "task_parameters": task.parameters,
                        },
                    )
                    
                    break
    
    def _process_messages(self) -> None:
        """
        Process messages.
        """
        with self.lock:
            # Get recent messages
            recent_messages = self.get_messages(since=time.time() - 1.0)
            
            # Process each message
            for message in recent_messages:
                # Check if the message is for the swarm
                if message.recipient_id == self.id:
                    # Process the message
                    if message.topic == "task_completed":
                        # Update the task status
                        task_id = message.content.get("task_id")
                        results = message.content.get("results")
                        
                        if task_id:
                            self.update_task_status(task_id, "completed", results)


class SwarmManager:
    """
    Swarm manager.
    
    This class is responsible for managing agent swarms.
    """
    
    def __init__(self):
        """
        Initialize the swarm manager.
        """
        self.swarms: Dict[str, AgentSwarm] = {}
        self.lock = threading.RLock()
    
    def create_swarm(self, name: str, config: Optional[Dict[str, Any]] = None) -> AgentSwarm:
        """
        Create a swarm.
        
        Args:
            name: The swarm name.
            config: The swarm configuration.
            
        Returns:
            The created swarm.
        """
        with self.lock:
            swarm = AgentSwarm(
                id=str(uuid.uuid4()),
                name=name,
                config=config or {},
            )
            
            self.swarms[swarm.id] = swarm
            
            return swarm
    
    def get_swarm(self, swarm_id: str) -> Optional[AgentSwarm]:
        """
        Get a swarm by ID.
        
        Args:
            swarm_id: The swarm ID.
            
        Returns:
            The swarm, or None if the swarm is not found.
        """
        with self.lock:
            return self.swarms.get(swarm_id)
    
    def get_swarms(self) -> Dict[str, AgentSwarm]:
        """
        Get all swarms.
        
        Returns:
            A dictionary mapping swarm IDs to swarms.
        """
        with self.lock:
            return self.swarms.copy()
    
    def delete_swarm(self, swarm_id: str) -> bool:
        """
        Delete a swarm.
        
        Args:
            swarm_id: The swarm ID.
            
        Returns:
            True if the swarm was deleted, False otherwise.
        """
        with self.lock:
            swarm = self.get_swarm(swarm_id)
            
            if not swarm:
                return False
            
            # Stop the swarm
            swarm.stop()
            
            # Delete the swarm
            del self.swarms[swarm_id]
            
            return True
    
    def start_swarm(self, swarm_id: str) -> bool:
        """
        Start a swarm.
        
        Args:
            swarm_id: The swarm ID.
            
        Returns:
            True if the swarm was started, False otherwise.
        """
        with self.lock:
            swarm = self.get_swarm(swarm_id)
            
            if not swarm:
                return False
            
            swarm.start()
            return True
    
    def stop_swarm(self, swarm_id: str) -> bool:
        """
        Stop a swarm.
        
        Args:
            swarm_id: The swarm ID.
            
        Returns:
            True if the swarm was stopped, False otherwise.
        """
        with self.lock:
            swarm = self.get_swarm(swarm_id)
            
            if not swarm:
                return False
            
            swarm.stop()
            return True


# Global swarm manager instance
_swarm_manager = None
_swarm_manager_lock = threading.RLock()


def get_swarm_manager() -> SwarmManager:
    """
    Get the global swarm manager instance.
    
    Returns:
        The global swarm manager instance.
    """
    global _swarm_manager
    
    with _swarm_manager_lock:
        if _swarm_manager is None:
            _swarm_manager = SwarmManager()
        
        return _swarm_manager
