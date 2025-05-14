"""
Learning module for Feluda.

This module provides learning capabilities for agents.
"""

import abc
import enum
import json
import logging
import os
import pickle
import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class LearningAlgorithm(str, enum.Enum):
    """Enum for learning algorithms."""
    
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    DQN = "dqn"
    PPO = "ppo"
    A2C = "a2c"
    DDPG = "ddpg"
    SAC = "sac"


class Experience(BaseModel):
    """
    Experience.
    
    This class represents an experience that an agent can learn from.
    """
    
    id: str = Field(..., description="The experience ID")
    state: Dict[str, Any] = Field(..., description="The state")
    action: Dict[str, Any] = Field(..., description="The action")
    reward: float = Field(..., description="The reward")
    next_state: Dict[str, Any] = Field(..., description="The next state")
    done: bool = Field(..., description="Whether the episode is done")
    timestamp: float = Field(..., description="The experience timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experience to a dictionary.
        
        Returns:
            A dictionary representation of the experience.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """
        Create an experience from a dictionary.
        
        Args:
            data: The dictionary to create the experience from.
            
        Returns:
            An experience.
        """
        return cls(**data)


class ExperienceBuffer:
    """
    Experience buffer.
    
    This class stores experiences for learning.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize an experience buffer.
        
        Args:
            capacity: The buffer capacity.
        """
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
        self.lock = threading.RLock()
    
    def add(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience: The experience to add.
        """
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample experiences from the buffer.
        
        Args:
            batch_size: The batch size.
            
        Returns:
            A list of experiences.
        """
        with self.lock:
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def clear(self) -> None:
        """
        Clear the buffer.
        """
        with self.lock:
            self.buffer = []
            self.position = 0
    
    def __len__(self) -> int:
        """
        Get the buffer length.
        
        Returns:
            The buffer length.
        """
        with self.lock:
            return len(self.buffer)


class LearningModel(abc.ABC):
    """
    Base class for learning models.
    
    This class defines the interface for learning models.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a learning model.
        
        Args:
            config: The model configuration.
        """
        self.config = config or {}
    
    @abc.abstractmethod
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict an action for a state.
        
        Args:
            state: The state.
            
        Returns:
            The predicted action.
        """
        pass
    
    @abc.abstractmethod
    def update(self, experiences: List[Experience]) -> None:
        """
        Update the model with experiences.
        
        Args:
            experiences: The experiences to update with.
        """
        pass
    
    @abc.abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: The path to save the model to.
        """
        pass
    
    @abc.abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: The path to load the model from.
        """
        pass


class QLearningModel(LearningModel):
    """
    Q-learning model.
    
    This class implements a Q-learning model.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a Q-learning model.
        
        Args:
            config: The model configuration.
        """
        super().__init__(config)
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = config.get("learning_rate", 0.1)
        self.discount_factor = config.get("discount_factor", 0.9)
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.lock = threading.RLock()
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """
        Convert a state to a key.
        
        Args:
            state: The state.
            
        Returns:
            The key.
        """
        return json.dumps(state, sort_keys=True)
    
    def _action_to_key(self, action: Dict[str, Any]) -> str:
        """
        Convert an action to a key.
        
        Args:
            action: The action.
            
        Returns:
            The key.
        """
        return json.dumps(action, sort_keys=True)
    
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict an action for a state.
        
        Args:
            state: The state.
            
        Returns:
            The predicted action.
        """
        with self.lock:
            # Convert the state to a key
            state_key = self._state_to_key(state)
            
            # Check if the state is in the Q-table
            if state_key not in self.q_table:
                # Return a random action
                return {}
            
            # Get the Q-values for the state
            q_values = self.q_table[state_key]
            
            # Check if there are any Q-values
            if not q_values:
                # Return a random action
                return {}
            
            # Explore or exploit
            if random.random() < self.exploration_rate:
                # Explore: choose a random action
                action_key = random.choice(list(q_values.keys()))
            else:
                # Exploit: choose the best action
                action_key = max(q_values, key=q_values.get)
            
            # Convert the action key to an action
            return json.loads(action_key)
    
    def update(self, experiences: List[Experience]) -> None:
        """
        Update the model with experiences.
        
        Args:
            experiences: The experiences to update with.
        """
        with self.lock:
            for experience in experiences:
                # Convert the state and action to keys
                state_key = self._state_to_key(experience.state)
                action_key = self._action_to_key(experience.action)
                next_state_key = self._state_to_key(experience.next_state)
                
                # Initialize the Q-values for the state if needed
                if state_key not in self.q_table:
                    self.q_table[state_key] = {}
                
                # Initialize the Q-value for the state-action pair if needed
                if action_key not in self.q_table[state_key]:
                    self.q_table[state_key][action_key] = 0.0
                
                # Get the current Q-value
                q_value = self.q_table[state_key][action_key]
                
                # Get the maximum Q-value for the next state
                max_q_value = 0.0
                
                if next_state_key in self.q_table and self.q_table[next_state_key]:
                    max_q_value = max(self.q_table[next_state_key].values())
                
                # Update the Q-value
                if experience.done:
                    # Terminal state
                    target = experience.reward
                else:
                    # Non-terminal state
                    target = experience.reward + self.discount_factor * max_q_value
                
                self.q_table[state_key][action_key] += self.learning_rate * (target - q_value)
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: The path to save the model to.
        """
        with self.lock:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            with open(path, "wb") as f:
                pickle.dump(self.q_table, f)
    
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: The path to load the model from.
        """
        with self.lock:
            # Load the model
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)


class SarsaModel(LearningModel):
    """
    SARSA model.
    
    This class implements a SARSA model.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a SARSA model.
        
        Args:
            config: The model configuration.
        """
        super().__init__(config)
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = config.get("learning_rate", 0.1)
        self.discount_factor = config.get("discount_factor", 0.9)
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.lock = threading.RLock()
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """
        Convert a state to a key.
        
        Args:
            state: The state.
            
        Returns:
            The key.
        """
        return json.dumps(state, sort_keys=True)
    
    def _action_to_key(self, action: Dict[str, Any]) -> str:
        """
        Convert an action to a key.
        
        Args:
            action: The action.
            
        Returns:
            The key.
        """
        return json.dumps(action, sort_keys=True)
    
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict an action for a state.
        
        Args:
            state: The state.
            
        Returns:
            The predicted action.
        """
        with self.lock:
            # Convert the state to a key
            state_key = self._state_to_key(state)
            
            # Check if the state is in the Q-table
            if state_key not in self.q_table:
                # Return a random action
                return {}
            
            # Get the Q-values for the state
            q_values = self.q_table[state_key]
            
            # Check if there are any Q-values
            if not q_values:
                # Return a random action
                return {}
            
            # Explore or exploit
            if random.random() < self.exploration_rate:
                # Explore: choose a random action
                action_key = random.choice(list(q_values.keys()))
            else:
                # Exploit: choose the best action
                action_key = max(q_values, key=q_values.get)
            
            # Convert the action key to an action
            return json.loads(action_key)
    
    def update(self, experiences: List[Experience]) -> None:
        """
        Update the model with experiences.
        
        Args:
            experiences: The experiences to update with.
        """
        with self.lock:
            for i, experience in enumerate(experiences):
                # Convert the state and action to keys
                state_key = self._state_to_key(experience.state)
                action_key = self._action_to_key(experience.action)
                next_state_key = self._state_to_key(experience.next_state)
                
                # Initialize the Q-values for the state if needed
                if state_key not in self.q_table:
                    self.q_table[state_key] = {}
                
                # Initialize the Q-value for the state-action pair if needed
                if action_key not in self.q_table[state_key]:
                    self.q_table[state_key][action_key] = 0.0
                
                # Get the current Q-value
                q_value = self.q_table[state_key][action_key]
                
                # Get the next action
                next_action = {}
                
                if i + 1 < len(experiences):
                    next_action = experiences[i + 1].action
                
                # Convert the next action to a key
                next_action_key = self._action_to_key(next_action)
                
                # Get the Q-value for the next state-action pair
                next_q_value = 0.0
                
                if next_state_key in self.q_table and next_action_key in self.q_table[next_state_key]:
                    next_q_value = self.q_table[next_state_key][next_action_key]
                
                # Update the Q-value
                if experience.done:
                    # Terminal state
                    target = experience.reward
                else:
                    # Non-terminal state
                    target = experience.reward + self.discount_factor * next_q_value
                
                self.q_table[state_key][action_key] += self.learning_rate * (target - q_value)
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: The path to save the model to.
        """
        with self.lock:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            with open(path, "wb") as f:
                pickle.dump(self.q_table, f)
    
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: The path to load the model from.
        """
        with self.lock:
            # Load the model
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)


class LearningManager:
    """
    Learning manager.
    
    This class is responsible for managing learning models.
    """
    
    def __init__(self):
        """
        Initialize the learning manager.
        """
        self.models: Dict[str, LearningModel] = {}
        self.buffers: Dict[str, ExperienceBuffer] = {}
        self.lock = threading.RLock()
    
    def create_model(self, model_id: str, algorithm: LearningAlgorithm, config: Optional[Dict[str, Any]] = None) -> LearningModel:
        """
        Create a learning model.
        
        Args:
            model_id: The model ID.
            algorithm: The learning algorithm.
            config: The model configuration.
            
        Returns:
            The created model.
            
        Raises:
            ValueError: If the algorithm is not supported.
        """
        with self.lock:
            # Create the model
            if algorithm == LearningAlgorithm.Q_LEARNING:
                model = QLearningModel(config)
            elif algorithm == LearningAlgorithm.SARSA:
                model = SarsaModel(config)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Store the model
            self.models[model_id] = model
            
            # Create a buffer for the model
            buffer_capacity = config.get("buffer_capacity", 10000) if config else 10000
            self.buffers[model_id] = ExperienceBuffer(capacity=buffer_capacity)
            
            return model
    
    def get_model(self, model_id: str) -> Optional[LearningModel]:
        """
        Get a model by ID.
        
        Args:
            model_id: The model ID.
            
        Returns:
            The model, or None if the model is not found.
        """
        with self.lock:
            return self.models.get(model_id)
    
    def get_buffer(self, model_id: str) -> Optional[ExperienceBuffer]:
        """
        Get a buffer by model ID.
        
        Args:
            model_id: The model ID.
            
        Returns:
            The buffer, or None if the buffer is not found.
        """
        with self.lock:
            return self.buffers.get(model_id)
    
    def add_experience(self, model_id: str, experience: Experience) -> bool:
        """
        Add an experience to a buffer.
        
        Args:
            model_id: The model ID.
            experience: The experience to add.
            
        Returns:
            True if the experience was added, False otherwise.
        """
        with self.lock:
            # Get the buffer
            buffer = self.get_buffer(model_id)
            
            if not buffer:
                return False
            
            # Add the experience
            buffer.add(experience)
            
            return True
    
    def train(self, model_id: str, batch_size: int = 32) -> bool:
        """
        Train a model.
        
        Args:
            model_id: The model ID.
            batch_size: The batch size.
            
        Returns:
            True if the model was trained, False otherwise.
        """
        with self.lock:
            # Get the model and buffer
            model = self.get_model(model_id)
            buffer = self.get_buffer(model_id)
            
            if not model or not buffer:
                return False
            
            # Check if there are enough experiences
            if len(buffer) < batch_size:
                return False
            
            # Sample experiences
            experiences = buffer.sample(batch_size)
            
            # Update the model
            model.update(experiences)
            
            return True
    
    def save_model(self, model_id: str, path: str) -> bool:
        """
        Save a model.
        
        Args:
            model_id: The model ID.
            path: The path to save the model to.
            
        Returns:
            True if the model was saved, False otherwise.
        """
        with self.lock:
            # Get the model
            model = self.get_model(model_id)
            
            if not model:
                return False
            
            # Save the model
            try:
                model.save(path)
                return True
            except Exception as e:
                log.error(f"Error saving model {model_id}: {e}")
                return False
    
    def load_model(self, model_id: str, path: str) -> bool:
        """
        Load a model.
        
        Args:
            model_id: The model ID.
            path: The path to load the model from.
            
        Returns:
            True if the model was loaded, False otherwise.
        """
        with self.lock:
            # Get the model
            model = self.get_model(model_id)
            
            if not model:
                return False
            
            # Load the model
            try:
                model.load(path)
                return True
            except Exception as e:
                log.error(f"Error loading model {model_id}: {e}")
                return False


# Global learning manager instance
_learning_manager = None
_learning_manager_lock = threading.RLock()


def get_learning_manager() -> LearningManager:
    """
    Get the global learning manager instance.
    
    Returns:
        The global learning manager instance.
    """
    global _learning_manager
    
    with _learning_manager_lock:
        if _learning_manager is None:
            _learning_manager = LearningManager()
        
        return _learning_manager
