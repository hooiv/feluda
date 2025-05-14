"""
Message broker for Feluda.

This module provides a message broker for distributed processing.
"""

import abc
import enum
import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import pika
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class Message(BaseModel):
    """
    Message.
    
    This class represents a message in the broker.
    """
    
    id: str = Field(..., description="The message ID")
    topic: str = Field(..., description="The message topic")
    payload: Dict[str, Any] = Field(..., description="The message payload")
    timestamp: float = Field(..., description="The message timestamp")
    headers: Dict[str, str] = Field(default_factory=dict, description="The message headers")
    
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
    
    @classmethod
    def create(cls, topic: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> "Message":
        """
        Create a new message.
        
        Args:
            topic: The message topic.
            payload: The message payload.
            headers: The message headers.
            
        Returns:
            A new message.
        """
        return cls(
            id=str(uuid.uuid4()),
            topic=topic,
            payload=payload,
            timestamp=time.time(),
            headers=headers or {},
        )


MessageHandler = Callable[[Message], None]


class BrokerBackend(abc.ABC):
    """
    Base class for broker backends.
    
    This class defines the interface for broker backends.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    @abc.abstractmethod
    def publish(self, message: Message) -> None:
        """
        Publish a message.
        
        Args:
            message: The message to publish.
        """
        pass
    
    @abc.abstractmethod
    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to.
            handler: The message handler.
        """
        pass
    
    @abc.abstractmethod
    def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: The topic to unsubscribe from.
            handler: The message handler.
        """
        pass
    
    @abc.abstractmethod
    def start(self) -> None:
        """
        Start the broker backend.
        """
        pass
    
    @abc.abstractmethod
    def stop(self) -> None:
        """
        Stop the broker backend.
        """
        pass


class MemoryBrokerBackend(BrokerBackend):
    """
    Memory broker backend.
    
    This class implements a broker backend that stores messages in memory.
    """
    
    def __init__(self):
        """
        Initialize a memory broker backend.
        """
        self.subscribers: Dict[str, List[MessageHandler]] = {}
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        self.queue: queue.Queue = queue.Queue()
    
    def publish(self, message: Message) -> None:
        """
        Publish a message.
        
        Args:
            message: The message to publish.
        """
        with self.lock:
            self.queue.put(message)
    
    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to.
            handler: The message handler.
        """
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            
            if handler not in self.subscribers[topic]:
                self.subscribers[topic].append(handler)
    
    def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: The topic to unsubscribe from.
            handler: The message handler.
        """
        with self.lock:
            if topic in self.subscribers and handler in self.subscribers[topic]:
                self.subscribers[topic].remove(handler)
                
                if not self.subscribers[topic]:
                    del self.subscribers[topic]
    
    def start(self) -> None:
        """
        Start the broker backend.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._process_queue)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the broker backend.
        """
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            if self.thread:
                self.thread.join()
                self.thread = None
    
    def _process_queue(self) -> None:
        """
        Process the message queue.
        """
        while self.running:
            try:
                # Get a message from the queue
                message = self.queue.get(timeout=1.0)
                
                # Process the message
                self._process_message(message)
                
                # Mark the message as processed
                self.queue.task_done()
            
            except queue.Empty:
                # No messages in the queue
                pass
            
            except Exception as e:
                log.error(f"Error processing message queue: {e}")
    
    def _process_message(self, message: Message) -> None:
        """
        Process a message.
        
        Args:
            message: The message to process.
        """
        with self.lock:
            # Get the subscribers for the topic
            subscribers = self.subscribers.get(message.topic, [])
            
            # Call the handlers
            for handler in subscribers:
                try:
                    handler(message)
                
                except Exception as e:
                    log.error(f"Error calling message handler: {e}")


class RabbitMQBrokerBackend(BrokerBackend):
    """
    RabbitMQ broker backend.
    
    This class implements a broker backend that uses RabbitMQ.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
        exchange: str = "feluda",
    ):
        """
        Initialize a RabbitMQ broker backend.
        
        Args:
            host: The RabbitMQ host.
            port: The RabbitMQ port.
            username: The RabbitMQ username.
            password: The RabbitMQ password.
            virtual_host: The RabbitMQ virtual host.
            exchange: The RabbitMQ exchange.
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.exchange = exchange
        self.connection = None
        self.channel = None
        self.subscribers: Dict[str, List[MessageHandler]] = {}
        self.consumer_tags: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
        self.running = False
    
    def _connect(self) -> None:
        """
        Connect to RabbitMQ.
        """
        if self.connection and self.connection.is_open:
            return
        
        # Create a connection
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=credentials,
        )
        
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        # Declare the exchange
        self.channel.exchange_declare(
            exchange=self.exchange,
            exchange_type="topic",
            durable=True,
        )
    
    def publish(self, message: Message) -> None:
        """
        Publish a message.
        
        Args:
            message: The message to publish.
        """
        with self.lock:
            if not self.running:
                raise RuntimeError("Broker backend is not running")
            
            # Connect to RabbitMQ
            self._connect()
            
            # Publish the message
            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key=message.topic,
                body=json.dumps(message.to_dict()),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    headers=message.headers,
                ),
            )
    
    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to.
            handler: The message handler.
        """
        with self.lock:
            if not self.running:
                raise RuntimeError("Broker backend is not running")
            
            # Connect to RabbitMQ
            self._connect()
            
            # Add the handler to the subscribers
            if topic not in self.subscribers:
                self.subscribers[topic] = []
                self.consumer_tags[topic] = []
            
            if handler not in self.subscribers[topic]:
                self.subscribers[topic].append(handler)
                
                # Declare a queue
                result = self.channel.queue_declare(queue="", exclusive=True)
                queue_name = result.method.queue
                
                # Bind the queue to the exchange
                self.channel.queue_bind(
                    exchange=self.exchange,
                    queue=queue_name,
                    routing_key=topic,
                )
                
                # Create a consumer
                consumer_tag = self.channel.basic_consume(
                    queue=queue_name,
                    on_message_callback=lambda ch, method, properties, body: self._on_message(
                        ch, method, properties, body, handler
                    ),
                    auto_ack=True,
                )
                
                self.consumer_tags[topic].append(consumer_tag)
    
    def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: The topic to unsubscribe from.
            handler: The message handler.
        """
        with self.lock:
            if not self.running:
                raise RuntimeError("Broker backend is not running")
            
            # Connect to RabbitMQ
            self._connect()
            
            # Remove the handler from the subscribers
            if topic in self.subscribers and handler in self.subscribers[topic]:
                index = self.subscribers[topic].index(handler)
                self.subscribers[topic].remove(handler)
                
                # Cancel the consumer
                if topic in self.consumer_tags and index < len(self.consumer_tags[topic]):
                    consumer_tag = self.consumer_tags[topic][index]
                    self.channel.basic_cancel(consumer_tag=consumer_tag)
                    self.consumer_tags[topic].remove(consumer_tag)
                
                if not self.subscribers[topic]:
                    del self.subscribers[topic]
                    del self.consumer_tags[topic]
    
    def start(self) -> None:
        """
        Start the broker backend.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            
            # Connect to RabbitMQ
            self._connect()
    
    def stop(self) -> None:
        """
        Stop the broker backend.
        """
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            # Close the connection
            if self.connection and self.connection.is_open:
                self.connection.close()
                self.connection = None
                self.channel = None
    
    def _on_message(
        self,
        channel: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
        handler: MessageHandler,
    ) -> None:
        """
        Handle a message.
        
        Args:
            channel: The channel.
            method: The method.
            properties: The properties.
            body: The message body.
            handler: The message handler.
        """
        try:
            # Parse the message
            message_dict = json.loads(body.decode())
            message = Message.from_dict(message_dict)
            
            # Call the handler
            handler(message)
        
        except Exception as e:
            log.error(f"Error handling message: {e}")


class BrokerManager:
    """
    Broker manager.
    
    This class is responsible for managing the message broker.
    """
    
    def __init__(self, backend: Optional[BrokerBackend] = None):
        """
        Initialize the broker manager.
        
        Args:
            backend: The broker backend.
        """
        config = get_config()
        
        if backend:
            self.backend = backend
        elif config.queue_url and config.queue_url.startswith("amqp://"):
            # Parse the RabbitMQ URL
            import urllib.parse
            
            url = urllib.parse.urlparse(config.queue_url)
            
            host = url.hostname or "localhost"
            port = url.port or 5672
            username = url.username or "guest"
            password = url.password or "guest"
            virtual_host = url.path or "/"
            
            if virtual_host.startswith("/"):
                virtual_host = virtual_host[1:]
            
            self.backend = RabbitMQBrokerBackend(
                host=host,
                port=port,
                username=username,
                password=password,
                virtual_host=virtual_host,
                exchange=config.queue_name,
            )
        else:
            self.backend = MemoryBrokerBackend()
        
        self.lock = threading.RLock()
    
    def publish(self, topic: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> None:
        """
        Publish a message.
        
        Args:
            topic: The message topic.
            payload: The message payload.
            headers: The message headers.
        """
        with self.lock:
            # Create a message
            message = Message.create(topic, payload, headers)
            
            # Publish the message
            self.backend.publish(message)
    
    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to.
            handler: The message handler.
        """
        with self.lock:
            self.backend.subscribe(topic, handler)
    
    def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: The topic to unsubscribe from.
            handler: The message handler.
        """
        with self.lock:
            self.backend.unsubscribe(topic, handler)
    
    def start(self) -> None:
        """
        Start the broker manager.
        """
        with self.lock:
            self.backend.start()
    
    def stop(self) -> None:
        """
        Stop the broker manager.
        """
        with self.lock:
            self.backend.stop()


# Global broker manager instance
_broker_manager = None
_broker_manager_lock = threading.RLock()


def get_broker_manager() -> BrokerManager:
    """
    Get the global broker manager instance.
    
    Returns:
        The global broker manager instance.
    """
    global _broker_manager
    
    with _broker_manager_lock:
        if _broker_manager is None:
            _broker_manager = BrokerManager()
            _broker_manager.start()
        
        return _broker_manager
