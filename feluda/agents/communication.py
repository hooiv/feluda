"""
Communication module for Feluda.

This module provides communication capabilities for agents.
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

from feluda.agents.agent import Agent, get_agent_manager
from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class MessageType(str, enum.Enum):
    """Enum for message types."""
    
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    DATA = "data"
    ERROR = "error"


class Message(BaseModel):
    """
    Message.
    
    This class represents a message that can be sent between agents.
    """
    
    id: str = Field(..., description="The message ID")
    type: MessageType = Field(..., description="The message type")
    sender_id: str = Field(..., description="The sender ID")
    recipient_id: str = Field(..., description="The recipient ID")
    subject: str = Field(..., description="The message subject")
    content: Dict[str, Any] = Field(..., description="The message content")
    timestamp: float = Field(..., description="The message timestamp")
    correlation_id: Optional[str] = Field(None, description="The correlation ID for related messages")
    expires_at: Optional[float] = Field(None, description="The expiration timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            A dictionary representation of the message.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create a message from a dictionary.
        
        Args:
            data: The dictionary to create the message from.
            
        Returns:
            A message.
        """
        return cls(**data)


class Channel(abc.ABC):
    """
    Base class for communication channels.
    
    This class defines the interface for communication channels.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = None):
        """
        Initialize a channel.
        
        Args:
            id: The channel ID.
            name: The channel name.
            config: The channel configuration.
        """
        self.id = id
        self.name = name
        self.config = config or {}
        self.subscribers: Set[str] = set()
        self.lock = threading.RLock()
    
    def subscribe(self, agent_id: str) -> bool:
        """
        Subscribe an agent to the channel.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            True if the agent was subscribed, False otherwise.
        """
        with self.lock:
            # Check if the agent exists
            agent_manager = get_agent_manager()
            agent = agent_manager.get_agent(agent_id)
            
            if not agent:
                return False
            
            # Subscribe the agent
            self.subscribers.add(agent_id)
            
            return True
    
    def unsubscribe(self, agent_id: str) -> bool:
        """
        Unsubscribe an agent from the channel.
        
        Args:
            agent_id: The agent ID.
            
        Returns:
            True if the agent was unsubscribed, False otherwise.
        """
        with self.lock:
            if agent_id in self.subscribers:
                self.subscribers.remove(agent_id)
                return True
            
            return False
    
    def get_subscribers(self) -> List[str]:
        """
        Get the subscribers to the channel.
        
        Returns:
            A list of agent IDs.
        """
        with self.lock:
            return list(self.subscribers)
    
    @abc.abstractmethod
    def send(self, message: Message) -> bool:
        """
        Send a message.
        
        Args:
            message: The message to send.
            
        Returns:
            True if the message was sent, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message.
        
        Args:
            agent_id: The agent ID.
            timeout: The timeout in seconds. If None, the method blocks indefinitely.
            
        Returns:
            The received message, or None if no message was received.
        """
        pass


class DirectChannel(Channel):
    """
    Direct channel.
    
    This class implements a direct communication channel between agents.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = None):
        """
        Initialize a direct channel.
        
        Args:
            id: The channel ID.
            name: The channel name.
            config: The channel configuration.
        """
        super().__init__(id, name, config)
        self.messages: Dict[str, List[Message]] = {}
    
    def send(self, message: Message) -> bool:
        """
        Send a message.
        
        Args:
            message: The message to send.
            
        Returns:
            True if the message was sent, False otherwise.
        """
        with self.lock:
            # Check if the recipient is subscribed
            if message.recipient_id not in self.subscribers:
                return False
            
            # Add the message to the recipient's queue
            if message.recipient_id not in self.messages:
                self.messages[message.recipient_id] = []
            
            self.messages[message.recipient_id].append(message)
            
            return True
    
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message.
        
        Args:
            agent_id: The agent ID.
            timeout: The timeout in seconds. If None, the method blocks indefinitely.
            
        Returns:
            The received message, or None if no message was received.
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                # Check if the agent has any messages
                if agent_id in self.messages and self.messages[agent_id]:
                    # Get the oldest message
                    message = self.messages[agent_id].pop(0)
                    
                    # Remove the agent's queue if it's empty
                    if not self.messages[agent_id]:
                        del self.messages[agent_id]
                    
                    return message
            
            # Check if the timeout has expired
            if timeout is not None and time.time() - start_time > timeout:
                return None
            
            # Sleep for a short time
            time.sleep(0.1)


class BroadcastChannel(Channel):
    """
    Broadcast channel.
    
    This class implements a broadcast communication channel between agents.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = None):
        """
        Initialize a broadcast channel.
        
        Args:
            id: The channel ID.
            name: The channel name.
            config: The channel configuration.
        """
        super().__init__(id, name, config)
        self.messages: List[Message] = []
        self.last_received: Dict[str, int] = {}
    
    def send(self, message: Message) -> bool:
        """
        Send a message.
        
        Args:
            message: The message to send.
            
        Returns:
            True if the message was sent, False otherwise.
        """
        with self.lock:
            # Add the message to the channel
            self.messages.append(message)
            
            return True
    
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message.
        
        Args:
            agent_id: The agent ID.
            timeout: The timeout in seconds. If None, the method blocks indefinitely.
            
        Returns:
            The received message, or None if no message was received.
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                # Check if the agent is subscribed
                if agent_id not in self.subscribers:
                    return None
                
                # Get the index of the last message received by the agent
                last_index = self.last_received.get(agent_id, -1)
                
                # Check if there are any new messages
                if last_index + 1 < len(self.messages):
                    # Get the next message
                    message = self.messages[last_index + 1]
                    
                    # Update the last received index
                    self.last_received[agent_id] = last_index + 1
                    
                    return message
            
            # Check if the timeout has expired
            if timeout is not None and time.time() - start_time > timeout:
                return None
            
            # Sleep for a short time
            time.sleep(0.1)


class TopicChannel(Channel):
    """
    Topic channel.
    
    This class implements a topic-based communication channel between agents.
    """
    
    def __init__(self, id: str, name: str, config: Dict[str, Any] = None):
        """
        Initialize a topic channel.
        
        Args:
            id: The channel ID.
            name: The channel name.
            config: The channel configuration.
        """
        super().__init__(id, name, config)
        self.topics: Dict[str, List[Message]] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.last_received: Dict[str, Dict[str, int]] = {}
    
    def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        """
        Subscribe an agent to a topic.
        
        Args:
            agent_id: The agent ID.
            topic: The topic.
            
        Returns:
            True if the agent was subscribed, False otherwise.
        """
        with self.lock:
            # Check if the agent is subscribed to the channel
            if agent_id not in self.subscribers:
                return False
            
            # Subscribe the agent to the topic
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            
            self.subscriptions[topic].add(agent_id)
            
            return True
    
    def unsubscribe_from_topic(self, agent_id: str, topic: str) -> bool:
        """
        Unsubscribe an agent from a topic.
        
        Args:
            agent_id: The agent ID.
            topic: The topic.
            
        Returns:
            True if the agent was unsubscribed, False otherwise.
        """
        with self.lock:
            # Check if the topic exists
            if topic not in self.subscriptions:
                return False
            
            # Unsubscribe the agent from the topic
            if agent_id in self.subscriptions[topic]:
                self.subscriptions[topic].remove(agent_id)
                
                # Remove the topic if it has no subscribers
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
                
                return True
            
            return False
    
    def get_topics(self) -> List[str]:
        """
        Get the topics.
        
        Returns:
            A list of topics.
        """
        with self.lock:
            return list(self.topics.keys())
    
    def get_topic_subscribers(self, topic: str) -> List[str]:
        """
        Get the subscribers to a topic.
        
        Args:
            topic: The topic.
            
        Returns:
            A list of agent IDs.
        """
        with self.lock:
            if topic not in self.subscriptions:
                return []
            
            return list(self.subscriptions[topic])
    
    def send(self, message: Message) -> bool:
        """
        Send a message.
        
        Args:
            message: The message to send.
            
        Returns:
            True if the message was sent, False otherwise.
        """
        with self.lock:
            # Get the topic from the message subject
            topic = message.subject
            
            # Add the message to the topic
            if topic not in self.topics:
                self.topics[topic] = []
            
            self.topics[topic].append(message)
            
            return True
    
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message.
        
        Args:
            agent_id: The agent ID.
            timeout: The timeout in seconds. If None, the method blocks indefinitely.
            
        Returns:
            The received message, or None if no message was received.
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                # Check if the agent is subscribed to the channel
                if agent_id not in self.subscribers:
                    return None
                
                # Check if the agent is subscribed to any topics
                subscribed_topics = [
                    topic
                    for topic, subscribers in self.subscriptions.items()
                    if agent_id in subscribers
                ]
                
                if not subscribed_topics:
                    return None
                
                # Initialize the last received indices if needed
                if agent_id not in self.last_received:
                    self.last_received[agent_id] = {}
                
                # Check each topic for new messages
                for topic in subscribed_topics:
                    # Skip topics with no messages
                    if topic not in self.topics:
                        continue
                    
                    # Get the index of the last message received by the agent for this topic
                    last_index = self.last_received[agent_id].get(topic, -1)
                    
                    # Check if there are any new messages
                    if last_index + 1 < len(self.topics[topic]):
                        # Get the next message
                        message = self.topics[topic][last_index + 1]
                        
                        # Update the last received index
                        self.last_received[agent_id][topic] = last_index + 1
                        
                        return message
            
            # Check if the timeout has expired
            if timeout is not None and time.time() - start_time > timeout:
                return None
            
            # Sleep for a short time
            time.sleep(0.1)


class CommunicationManager:
    """
    Communication manager.
    
    This class is responsible for managing communication channels.
    """
    
    def __init__(self):
        """
        Initialize the communication manager.
        """
        self.channels: Dict[str, Channel] = {}
        self.lock = threading.RLock()
    
    def create_direct_channel(self, name: str, config: Optional[Dict[str, Any]] = None) -> DirectChannel:
        """
        Create a direct channel.
        
        Args:
            name: The channel name.
            config: The channel configuration.
            
        Returns:
            The created channel.
        """
        with self.lock:
            channel = DirectChannel(
                id=str(uuid.uuid4()),
                name=name,
                config=config or {},
            )
            
            self.channels[channel.id] = channel
            
            return channel
    
    def create_broadcast_channel(self, name: str, config: Optional[Dict[str, Any]] = None) -> BroadcastChannel:
        """
        Create a broadcast channel.
        
        Args:
            name: The channel name.
            config: The channel configuration.
            
        Returns:
            The created channel.
        """
        with self.lock:
            channel = BroadcastChannel(
                id=str(uuid.uuid4()),
                name=name,
                config=config or {},
            )
            
            self.channels[channel.id] = channel
            
            return channel
    
    def create_topic_channel(self, name: str, config: Optional[Dict[str, Any]] = None) -> TopicChannel:
        """
        Create a topic channel.
        
        Args:
            name: The channel name.
            config: The channel configuration.
            
        Returns:
            The created channel.
        """
        with self.lock:
            channel = TopicChannel(
                id=str(uuid.uuid4()),
                name=name,
                config=config or {},
            )
            
            self.channels[channel.id] = channel
            
            return channel
    
    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """
        Get a channel by ID.
        
        Args:
            channel_id: The channel ID.
            
        Returns:
            The channel, or None if the channel is not found.
        """
        with self.lock:
            return self.channels.get(channel_id)
    
    def get_channels(self) -> Dict[str, Channel]:
        """
        Get all channels.
        
        Returns:
            A dictionary mapping channel IDs to channels.
        """
        with self.lock:
            return self.channels.copy()
    
    def delete_channel(self, channel_id: str) -> bool:
        """
        Delete a channel.
        
        Args:
            channel_id: The channel ID.
            
        Returns:
            True if the channel was deleted, False otherwise.
        """
        with self.lock:
            if channel_id in self.channels:
                del self.channels[channel_id]
                return True
            
            return False


# Global communication manager instance
_communication_manager = None
_communication_manager_lock = threading.RLock()


def get_communication_manager() -> CommunicationManager:
    """
    Get the global communication manager instance.
    
    Returns:
        The global communication manager instance.
    """
    global _communication_manager
    
    with _communication_manager_lock:
        if _communication_manager is None:
            _communication_manager = CommunicationManager()
        
        return _communication_manager
