"""
AI Agent Swarm Module

This module provides integration with AI agent swarms for collaborative development and QA.
"""

import json
import logging
import os
import tempfile
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import requests

from feluda.observability import get_logger

log = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class AgentRole(str, Enum):
    """Enum for agent roles."""
    
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    QA = "qa"
    SECURITY = "security"
    INTEGRATOR = "integrator"
    COORDINATOR = "coordinator"


class AgentMessage:
    """
    Message exchanged between agents.
    
    This class represents a message exchanged between agents in a swarm.
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an AgentMessage.
        
        Args:
            sender: The ID of the sender agent.
            receiver: The ID of the receiver agent.
            content: The content of the message.
            message_type: The type of the message.
            metadata: Additional metadata for the message.
        """
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            A dictionary representation of the message.
        """
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """
        Create a message from a dictionary.
        
        Args:
            data: The dictionary representation of the message.
            
        Returns:
            The created message.
        """
        message = cls(
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            message_type=data["message_type"],
            metadata=data["metadata"],
        )
        message.timestamp = data["timestamp"]
        return message


class Agent:
    """
    Base class for agents.
    
    This class defines the interface for agents in a swarm.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[str],
        **kwargs: Any,
    ):
        """
        Initialize an agent.
        
        Args:
            agent_id: The ID of the agent.
            role: The role of the agent.
            capabilities: The capabilities of the agent.
            **kwargs: Additional agent-specific parameters.
        """
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.memory: Dict[str, Any] = {}
    
    def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message.
        
        Args:
            message: The message to receive.
        """
        if message.receiver == self.agent_id:
            self.inbox.append(message)
    
    def send_message(self, receiver: str, content: str, message_type: str = "text", metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Send a message.
        
        Args:
            receiver: The ID of the receiver agent.
            content: The content of the message.
            message_type: The type of the message.
            metadata: Additional metadata for the message.
        """
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            message_type=message_type,
            metadata=metadata,
        )
        self.outbox.append(message)
    
    def process_messages(self) -> None:
        """
        Process incoming messages.
        
        This method should be implemented by subclasses to process incoming messages.
        """
        pass
    
    def act(self) -> None:
        """
        Perform an action.
        
        This method should be implemented by subclasses to perform actions.
        """
        pass
    
    def update_memory(self, key: str, value: Any) -> None:
        """
        Update the agent's memory.
        
        Args:
            key: The key to update.
            value: The value to set.
        """
        self.memory[key] = value
    
    def get_memory(self, key: str) -> Any:
        """
        Get a value from the agent's memory.
        
        Args:
            key: The key to get.
            
        Returns:
            The value associated with the key, or None if the key is not found.
        """
        return self.memory.get(key)


class LLMAgent(Agent):
    """
    Agent powered by a large language model.
    
    This class implements an agent powered by a large language model.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[str],
        model: str,
        api_key: str,
        api_url: str,
        system_prompt: str,
        **kwargs: Any,
    ):
        """
        Initialize an LLM agent.
        
        Args:
            agent_id: The ID of the agent.
            role: The role of the agent.
            capabilities: The capabilities of the agent.
            model: The name of the language model to use.
            api_key: The API key for the language model service.
            api_url: The URL of the language model service API.
            system_prompt: The system prompt for the language model.
            **kwargs: Additional agent-specific parameters.
        """
        super().__init__(agent_id, role, capabilities, **kwargs)
        self.model = model
        self.api_key = api_key
        self.api_url = api_url
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
    
    def process_messages(self) -> None:
        """Process incoming messages."""
        for message in self.inbox:
            self.conversation_history.append({
                "role": "user" if message.sender != self.agent_id else "assistant",
                "content": message.content,
            })
        
        self.inbox = []
    
    def act(self) -> None:
        """Perform an action using the language model."""
        if len(self.conversation_history) <= 1:
            # No messages to process
            return
        
        try:
            # Call the language model API
            response = self._call_llm_api()
            
            # Process the response
            if response:
                # Add the response to the conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                })
                
                # Send the response to the coordinator
                self.send_message(
                    receiver="coordinator",
                    content=response,
                    message_type="text",
                    metadata={"model": self.model},
                )
        except Exception as e:
            log.error(f"Error calling LLM API: {e}")
    
    def _call_llm_api(self) -> str:
        """
        Call the language model API.
        
        Returns:
            The response from the language model.
            
        Raises:
            Exception: If the API call fails.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        data = {
            "model": self.model,
            "messages": self.conversation_history,
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")


class AgentSwarm:
    """
    Swarm of agents working together.
    
    This class manages a swarm of agents working together on a task.
    """
    
    def __init__(self):
        """Initialize an agent swarm."""
        self.agents: Dict[str, Agent] = {}
        self.message_queue: List[AgentMessage] = []
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the swarm.
        
        Args:
            agent: The agent to add.
        """
        self.agents[agent.agent_id] = agent
    
    def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the swarm.
        
        Args:
            agent_id: The ID of the agent to remove.
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: The ID of the agent to get.
            
        Returns:
            The agent with the specified ID, or None if not found.
        """
        return self.agents.get(agent_id)
    
    def dispatch_messages(self) -> None:
        """Dispatch messages from agents' outboxes to the message queue."""
        for agent in self.agents.values():
            for message in agent.outbox:
                self.message_queue.append(message)
            agent.outbox = []
    
    def deliver_messages(self) -> None:
        """Deliver messages from the message queue to agents' inboxes."""
        for message in self.message_queue:
            if message.receiver in self.agents:
                self.agents[message.receiver].receive_message(message)
        
        self.message_queue = []
    
    def run_step(self) -> None:
        """Run a single step of the swarm."""
        # Process incoming messages
        for agent in self.agents.values():
            agent.process_messages()
        
        # Let agents act
        for agent in self.agents.values():
            agent.act()
        
        # Dispatch and deliver messages
        self.dispatch_messages()
        self.deliver_messages()
    
    def run(self, steps: int) -> None:
        """
        Run the swarm for a specified number of steps.
        
        Args:
            steps: The number of steps to run.
        """
        for _ in range(steps):
            self.run_step()


def create_development_swarm(
    task: str,
    code_context: str,
    api_key: str,
    api_url: str,
) -> AgentSwarm:
    """
    Create a swarm of agents for collaborative development.
    
    Args:
        task: The development task to perform.
        code_context: The context of the code to work on.
        api_key: The API key for the language model service.
        api_url: The URL of the language model service API.
        
    Returns:
        The created agent swarm.
    """
    swarm = AgentSwarm()
    
    # Create the architect agent
    architect = LLMAgent(
        agent_id="architect",
        role=AgentRole.ARCHITECT,
        capabilities=["design", "architecture"],
        model="gpt-4",
        api_key=api_key,
        api_url=api_url,
        system_prompt=(
            "You are an expert software architect. Your task is to design a solution for the given task. "
            "Consider the code context provided and propose a high-level design."
        ),
    )
    swarm.add_agent(architect)
    
    # Create the developer agent
    developer = LLMAgent(
        agent_id="developer",
        role=AgentRole.DEVELOPER,
        capabilities=["coding", "implementation"],
        model="gpt-4",
        api_key=api_key,
        api_url=api_url,
        system_prompt=(
            "You are an expert software developer. Your task is to implement the solution based on the architect's design. "
            "Write clean, efficient, and well-documented code."
        ),
    )
    swarm.add_agent(developer)
    
    # Create the reviewer agent
    reviewer = LLMAgent(
        agent_id="reviewer",
        role=AgentRole.REVIEWER,
        capabilities=["code review", "quality assurance"],
        model="gpt-4",
        api_key=api_key,
        api_url=api_url,
        system_prompt=(
            "You are an expert code reviewer. Your task is to review the code written by the developer. "
            "Look for bugs, inefficiencies, and areas for improvement."
        ),
    )
    swarm.add_agent(reviewer)
    
    # Create the tester agent
    tester = LLMAgent(
        agent_id="tester",
        role=AgentRole.TESTER,
        capabilities=["testing", "quality assurance"],
        model="gpt-4",
        api_key=api_key,
        api_url=api_url,
        system_prompt=(
            "You are an expert software tester. Your task is to write tests for the code written by the developer. "
            "Ensure that the code works as expected and handles edge cases."
        ),
    )
    swarm.add_agent(tester)
    
    # Create the coordinator agent
    coordinator = LLMAgent(
        agent_id="coordinator",
        role=AgentRole.COORDINATOR,
        capabilities=["coordination", "integration"],
        model="gpt-4",
        api_key=api_key,
        api_url=api_url,
        system_prompt=(
            "You are the coordinator of a team of AI agents working on a software development task. "
            "Your task is to coordinate the work of the other agents and ensure that the task is completed successfully."
        ),
    )
    swarm.add_agent(coordinator)
    
    # Initialize the task
    coordinator.send_message(
        receiver="architect",
        content=f"Task: {task}\n\nCode Context:\n{code_context}",
        message_type="task",
    )
    
    return swarm
