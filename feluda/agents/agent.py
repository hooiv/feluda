"""
Agent module for Feluda.

This module provides autonomous agents for the Feluda framework.
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

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class AgentState(str, enum.Enum):
    """Enum for agent states."""
    
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class AgentAction(BaseModel):
    """
    Agent action.
    
    This class represents an action that an agent can perform.
    """
    
    id: str = Field(..., description="The action ID")
    name: str = Field(..., description="The action name")
    description: Optional[str] = Field(None, description="The action description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="The action parameters")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the action to a dictionary.
        
        Returns:
            A dictionary representation of the action.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentAction":
        """
        Create an action from a dictionary.
        
        Args:
            data: The dictionary to create the action from.
            
        Returns:
            An action.
        """
        return cls(**data)


class AgentObservation(BaseModel):
    """
    Agent observation.
    
    This class represents an observation that an agent can make.
    """
    
    id: str = Field(..., description="The observation ID")
    timestamp: float = Field(..., description="The observation timestamp")
    data: Dict[str, Any] = Field(..., description="The observation data")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the observation to a dictionary.
        
        Returns:
            A dictionary representation of the observation.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentObservation":
        """
        Create an observation from a dictionary.
        
        Args:
            data: The dictionary to create the observation from.
            
        Returns:
            An observation.
        """
        return cls(**data)


class AgentGoal(BaseModel):
    """
    Agent goal.
    
    This class represents a goal that an agent can pursue.
    """
    
    id: str = Field(..., description="The goal ID")
    name: str = Field(..., description="The goal name")
    description: Optional[str] = Field(None, description="The goal description")
    criteria: Dict[str, Any] = Field(..., description="The goal criteria")
    priority: int = Field(0, description="The goal priority")
    deadline: Optional[float] = Field(None, description="The goal deadline")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the goal to a dictionary.
        
        Returns:
            A dictionary representation of the goal.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentGoal":
        """
        Create a goal from a dictionary.
        
        Args:
            data: The dictionary to create the goal from.
            
        Returns:
            A goal.
        """
        return cls(**data)


class Agent(abc.ABC):
    """
    Base class for agents.
    
    This class defines the interface for agents.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = None):
        """
        Initialize an agent.
        
        Args:
            id: The agent ID.
            name: The agent name.
            config: The agent configuration.
        """
        self.id = id
        self.name = name
        self.config = config or {}
        self.state = AgentState.IDLE
        self.observations: List[AgentObservation] = []
        self.actions: List[AgentAction] = []
        self.goals: List[AgentGoal] = []
        self.lock = threading.RLock()
        self.thread = None
        self.running = False
    
    def start(self) -> None:
        """
        Start the agent.
        """
        with self.lock:
            if self.state != AgentState.IDLE and self.state != AgentState.STOPPED:
                return
            
            self.state = AgentState.RUNNING
            self.running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the agent.
        """
        with self.lock:
            if self.state != AgentState.RUNNING and self.state != AgentState.PAUSED:
                return
            
            self.state = AgentState.STOPPED
            self.running = False
            
            if self.thread:
                self.thread.join()
                self.thread = None
    
    def pause(self) -> None:
        """
        Pause the agent.
        """
        with self.lock:
            if self.state != AgentState.RUNNING:
                return
            
            self.state = AgentState.PAUSED
    
    def resume(self) -> None:
        """
        Resume the agent.
        """
        with self.lock:
            if self.state != AgentState.PAUSED:
                return
            
            self.state = AgentState.RUNNING
    
    def add_observation(self, data: Dict[str, Any]) -> AgentObservation:
        """
        Add an observation.
        
        Args:
            data: The observation data.
            
        Returns:
            The created observation.
        """
        with self.lock:
            observation = AgentObservation(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                data=data,
            )
            
            self.observations.append(observation)
            
            return observation
    
    def get_observations(self, limit: Optional[int] = None) -> List[AgentObservation]:
        """
        Get observations.
        
        Args:
            limit: The maximum number of observations to return. If None, return all observations.
            
        Returns:
            A list of observations.
        """
        with self.lock:
            if limit is None:
                return self.observations.copy()
            
            return self.observations[-limit:].copy()
    
    def add_action(self, name: str, description: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> AgentAction:
        """
        Add an action.
        
        Args:
            name: The action name.
            description: The action description.
            parameters: The action parameters.
            
        Returns:
            The created action.
        """
        with self.lock:
            action = AgentAction(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                parameters=parameters or {},
            )
            
            self.actions.append(action)
            
            return action
    
    def get_actions(self) -> List[AgentAction]:
        """
        Get actions.
        
        Returns:
            A list of actions.
        """
        with self.lock:
            return self.actions.copy()
    
    def add_goal(
        self,
        name: str,
        criteria: Dict[str, Any],
        description: Optional[str] = None,
        priority: int = 0,
        deadline: Optional[float] = None,
    ) -> AgentGoal:
        """
        Add a goal.
        
        Args:
            name: The goal name.
            criteria: The goal criteria.
            description: The goal description.
            priority: The goal priority.
            deadline: The goal deadline.
            
        Returns:
            The created goal.
        """
        with self.lock:
            goal = AgentGoal(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                criteria=criteria,
                priority=priority,
                deadline=deadline,
            )
            
            self.goals.append(goal)
            
            return goal
    
    def get_goals(self) -> List[AgentGoal]:
        """
        Get goals.
        
        Returns:
            A list of goals.
        """
        with self.lock:
            return self.goals.copy()
    
    def _run(self) -> None:
        """
        Run the agent.
        """
        try:
            while self.running:
                with self.lock:
                    if self.state == AgentState.PAUSED:
                        time.sleep(0.1)
                        continue
                    
                    if self.state != AgentState.RUNNING:
                        break
                
                # Run the agent
                try:
                    self.run()
                except Exception as e:
                    log.error(f"Error running agent {self.name}: {e}")
                    with self.lock:
                        self.state = AgentState.ERROR
                    
                    break
                
                # Sleep for a short time
                time.sleep(0.1)
        
        except Exception as e:
            log.error(f"Error in agent {self.name} thread: {e}")
            with self.lock:
                self.state = AgentState.ERROR
    
    @abc.abstractmethod
    def run(self) -> None:
        """
        Run the agent.
        
        This method is called repeatedly while the agent is running.
        """
        pass


class RuleBasedAgent(Agent):
    """
    Rule-based agent.
    
    This class implements a rule-based agent.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = None):
        """
        Initialize a rule-based agent.
        
        Args:
            id: The agent ID.
            name: The agent name.
            config: The agent configuration.
        """
        super().__init__(id, name, config)
        self.rules: List[Dict[str, Any]] = []
    
    def add_rule(self, condition: Dict[str, Any], action: Dict[str, Any]) -> None:
        """
        Add a rule.
        
        Args:
            condition: The rule condition.
            action: The rule action.
        """
        with self.lock:
            self.rules.append({
                "condition": condition,
                "action": action,
            })
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """
        Get rules.
        
        Returns:
            A list of rules.
        """
        with self.lock:
            return self.rules.copy()
    
    def run(self) -> None:
        """
        Run the agent.
        """
        with self.lock:
            # Get the latest observation
            if not self.observations:
                return
            
            observation = self.observations[-1]
            
            # Check each rule
            for rule in self.rules:
                # Check if the condition matches
                if self._match_condition(rule["condition"], observation.data):
                    # Perform the action
                    self._perform_action(rule["action"])
    
    def _match_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """
        Check if a condition matches data.
        
        Args:
            condition: The condition to check.
            data: The data to check against.
            
        Returns:
            True if the condition matches, False otherwise.
        """
        for key, value in condition.items():
            if key not in data:
                return False
            
            if isinstance(value, dict):
                # Nested condition
                if not isinstance(data[key], dict):
                    return False
                
                if not self._match_condition(value, data[key]):
                    return False
            elif isinstance(value, list):
                # List condition
                if not isinstance(data[key], list):
                    return False
                
                if not all(item in data[key] for item in value):
                    return False
            else:
                # Simple condition
                if data[key] != value:
                    return False
        
        return True
    
    def _perform_action(self, action: Dict[str, Any]) -> None:
        """
        Perform an action.
        
        Args:
            action: The action to perform.
        """
        # Add the action
        self.add_action(
            name=action.get("name", "unknown"),
            description=action.get("description"),
            parameters=action.get("parameters", {}),
        )
        
        # Perform the action
        # This is a placeholder implementation
        log.info(f"Agent {self.name} performed action: {action}")


class LearningAgent(Agent):
    """
    Learning agent.
    
    This class implements a learning agent.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = None):
        """
        Initialize a learning agent.
        
        Args:
            id: The agent ID.
            name: The agent name.
            config: The agent configuration.
        """
        super().__init__(id, name, config)
        self.model = None
        self.learning_rate = config.get("learning_rate", 0.1)
        self.discount_factor = config.get("discount_factor", 0.9)
        self.exploration_rate = config.get("exploration_rate", 0.1)
    
    def run(self) -> None:
        """
        Run the agent.
        """
        # This is a placeholder implementation
        # In a real implementation, this would use a machine learning model
        
        with self.lock:
            # Get the latest observation
            if not self.observations:
                return
            
            observation = self.observations[-1]
            
            # Choose an action
            if self.model is None or random.random() < self.exploration_rate:
                # Explore: choose a random action
                action_name = random.choice(["action1", "action2", "action3"])
                action_parameters = {}
            else:
                # Exploit: choose the best action according to the model
                action_name = "action1"  # Placeholder
                action_parameters = {}
            
            # Perform the action
            self.add_action(
                name=action_name,
                parameters=action_parameters,
            )
            
            # Update the model
            # This is a placeholder implementation
            
            log.info(f"Agent {self.name} performed action: {action_name}")


class AgentManager:
    """
    Agent manager.
    
    This class is responsible for managing agents.
    """
    
    def __init__(self):
        """
        Initialize the agent manager.
        """
        self.agents: Dict[str, Agent] = {}
        self.lock = threading.RLock()
    
    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent.
        
        Args:
            agent: The agent to register.
        """
        with self.lock:
            self.agents[agent.id] = agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            The agent, or None if the agent is not found.
        """
        with self.lock:
            return self.agents.get(agent_id)
    
    def get_agents(self) -> Dict[str, Agent]:
        """
        Get all agents.
        
        Returns:
            A dictionary mapping agent IDs to agents.
        """
        with self.lock:
            return self.agents.copy()
    
    def create_rule_based_agent(self, name: str, config: Optional[Dict[str, Any]] = None) -> RuleBasedAgent:
        """
        Create a rule-based agent.
        
        Args:
            name: The agent name.
            config: The agent configuration.
            
        Returns:
            The created agent.
        """
        with self.lock:
            agent = RuleBasedAgent(
                id=str(uuid.uuid4()),
                name=name,
                config=config or {},
            )
            
            self.register_agent(agent)
            
            return agent
    
    def create_learning_agent(self, name: str, config: Optional[Dict[str, Any]] = None) -> LearningAgent:
        """
        Create a learning agent.
        
        Args:
            name: The agent name.
            config: The agent configuration.
            
        Returns:
            The created agent.
        """
        with self.lock:
            agent = LearningAgent(
                id=str(uuid.uuid4()),
                name=name,
                config=config or {},
            )
            
            self.register_agent(agent)
            
            return agent
    
    def start_agent(self, agent_id: str) -> bool:
        """
        Start an agent.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            True if the agent was started, False otherwise.
        """
        with self.lock:
            agent = self.get_agent(agent_id)
            
            if not agent:
                return False
            
            agent.start()
            return True
    
    def stop_agent(self, agent_id: str) -> bool:
        """
        Stop an agent.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            True if the agent was stopped, False otherwise.
        """
        with self.lock:
            agent = self.get_agent(agent_id)
            
            if not agent:
                return False
            
            agent.stop()
            return True
    
    def pause_agent(self, agent_id: str) -> bool:
        """
        Pause an agent.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            True if the agent was paused, False otherwise.
        """
        with self.lock:
            agent = self.get_agent(agent_id)
            
            if not agent:
                return False
            
            agent.pause()
            return True
    
    def resume_agent(self, agent_id: str) -> bool:
        """
        Resume an agent.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            True if the agent was resumed, False otherwise.
        """
        with self.lock:
            agent = self.get_agent(agent_id)
            
            if not agent:
                return False
            
            agent.resume()
            return True
    
    def add_observation(self, agent_id: str, data: Dict[str, Any]) -> Optional[AgentObservation]:
        """
        Add an observation to an agent.
        
        Args:
            agent_id: The agent ID.
            data: The observation data.
            
        Returns:
            The created observation, or None if the agent is not found.
        """
        with self.lock:
            agent = self.get_agent(agent_id)
            
            if not agent:
                return None
            
            return agent.add_observation(data)
    
    def add_goal(
        self,
        agent_id: str,
        name: str,
        criteria: Dict[str, Any],
        description: Optional[str] = None,
        priority: int = 0,
        deadline: Optional[float] = None,
    ) -> Optional[AgentGoal]:
        """
        Add a goal to an agent.
        
        Args:
            agent_id: The agent ID.
            name: The goal name.
            criteria: The goal criteria.
            description: The goal description.
            priority: The goal priority.
            deadline: The goal deadline.
            
        Returns:
            The created goal, or None if the agent is not found.
        """
        with self.lock:
            agent = self.get_agent(agent_id)
            
            if not agent:
                return None
            
            return agent.add_goal(
                name=name,
                criteria=criteria,
                description=description,
                priority=priority,
                deadline=deadline,
            )


# Global agent manager instance
_agent_manager = None
_agent_manager_lock = threading.RLock()


def get_agent_manager() -> AgentManager:
    """
    Get the global agent manager instance.
    
    Returns:
        The global agent manager instance.
    """
    global _agent_manager
    
    with _agent_manager_lock:
        if _agent_manager is None:
            _agent_manager = AgentManager()
        
        return _agent_manager
