"""
Connectors module for Feluda.

This module provides connectors for the data pipeline.
"""

import abc
import enum
import json
import logging
import os
import threading
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, TypeVar, Union

import boto3
import pandas as pd
import sqlalchemy
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


class Connector(abc.ABC):
    """
    Base class for connectors.
    
    This class defines the interface for connectors.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a connector.
        
        Args:
            name: The connector name.
            config: The connector configuration.
        """
        self.name = name
        self.config = config
    
    @abc.abstractmethod
    def read(self, **kwargs) -> Any:
        """
        Read data from the connector.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            The data.
        """
        pass
    
    @abc.abstractmethod
    def write(self, data: Any, **kwargs) -> None:
        """
        Write data to the connector.
        
        Args:
            data: The data to write.
            **kwargs: Additional arguments.
        """
        pass
    
    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the connector.
        """
        pass


class FileConnector(Connector):
    """
    File connector.
    
    This class implements a connector that reads from and writes to files.
    """
    
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Read data from a file.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            The data as a pandas DataFrame.
        """
        # Get the file path
        file_path = kwargs.get("file_path") or self.config.get("file_path")
        
        if not file_path:
            raise ValueError("File path not specified")
        
        # Get the file format
        file_format = kwargs.get("file_format") or self.config.get("file_format", "csv")
        
        # Read the file
        if file_format == "csv":
            return pd.read_csv(file_path, **kwargs.get("read_options", {}))
        elif file_format == "json":
            return pd.read_json(file_path, **kwargs.get("read_options", {}))
        elif file_format == "parquet":
            return pd.read_parquet(file_path, **kwargs.get("read_options", {}))
        elif file_format == "excel":
            return pd.read_excel(file_path, **kwargs.get("read_options", {}))
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def write(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Write data to a file.
        
        Args:
            data: The data to write.
            **kwargs: Additional arguments.
        """
        # Get the file path
        file_path = kwargs.get("file_path") or self.config.get("file_path")
        
        if not file_path:
            raise ValueError("File path not specified")
        
        # Get the file format
        file_format = kwargs.get("file_format") or self.config.get("file_format", "csv")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the file
        if file_format == "csv":
            data.to_csv(file_path, **kwargs.get("write_options", {}))
        elif file_format == "json":
            data.to_json(file_path, **kwargs.get("write_options", {}))
        elif file_format == "parquet":
            data.to_parquet(file_path, **kwargs.get("write_options", {}))
        elif file_format == "excel":
            data.to_excel(file_path, **kwargs.get("write_options", {}))
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def close(self) -> None:
        """
        Close the connector.
        """
        pass


class SQLConnector(Connector):
    """
    SQL connector.
    
    This class implements a connector that reads from and writes to SQL databases.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a SQL connector.
        
        Args:
            name: The connector name.
            config: The connector configuration.
        """
        super().__init__(name, config)
        self.engine = None
    
    def _get_engine(self) -> sqlalchemy.engine.Engine:
        """
        Get the SQLAlchemy engine.
        
        Returns:
            The SQLAlchemy engine.
        """
        if not self.engine:
            # Get the connection string
            connection_string = self.config.get("connection_string")
            
            if not connection_string:
                # Build the connection string from the configuration
                dialect = self.config.get("dialect", "sqlite")
                username = self.config.get("username")
                password = self.config.get("password")
                host = self.config.get("host")
                port = self.config.get("port")
                database = self.config.get("database")
                
                if dialect == "sqlite":
                    connection_string = f"sqlite:///{database}"
                else:
                    connection_string = f"{dialect}://"
                    
                    if username and password:
                        connection_string += f"{username}:{password}@"
                    
                    if host:
                        connection_string += host
                        
                        if port:
                            connection_string += f":{port}"
                    
                    if database:
                        connection_string += f"/{database}"
            
            # Create the engine
            self.engine = sqlalchemy.create_engine(connection_string)
        
        return self.engine
    
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Read data from a SQL database.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            The data as a pandas DataFrame.
        """
        # Get the query
        query = kwargs.get("query") or self.config.get("query")
        
        if not query:
            # Get the table name
            table_name = kwargs.get("table_name") or self.config.get("table_name")
            
            if not table_name:
                raise ValueError("Neither query nor table name specified")
            
            query = f"SELECT * FROM {table_name}"
        
        # Get the engine
        engine = self._get_engine()
        
        # Read the data
        return pd.read_sql(query, engine, **kwargs.get("read_options", {}))
    
    def write(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Write data to a SQL database.
        
        Args:
            data: The data to write.
            **kwargs: Additional arguments.
        """
        # Get the table name
        table_name = kwargs.get("table_name") or self.config.get("table_name")
        
        if not table_name:
            raise ValueError("Table name not specified")
        
        # Get the engine
        engine = self._get_engine()
        
        # Write the data
        data.to_sql(table_name, engine, **kwargs.get("write_options", {}))
    
    def close(self) -> None:
        """
        Close the connector.
        """
        if self.engine:
            self.engine.dispose()
            self.engine = None


class S3Connector(Connector):
    """
    S3 connector.
    
    This class implements a connector that reads from and writes to S3.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize an S3 connector.
        
        Args:
            name: The connector name.
            config: The connector configuration.
        """
        super().__init__(name, config)
        self.s3_client = None
    
    def _get_s3_client(self) -> boto3.client:
        """
        Get the S3 client.
        
        Returns:
            The S3 client.
        """
        if not self.s3_client:
            # Get the AWS credentials
            aws_access_key_id = self.config.get("aws_access_key_id")
            aws_secret_access_key = self.config.get("aws_secret_access_key")
            region_name = self.config.get("region_name")
            
            # Create the S3 client
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
            )
        
        return self.s3_client
    
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Read data from S3.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            The data as a pandas DataFrame.
        """
        # Get the bucket and key
        bucket = kwargs.get("bucket") or self.config.get("bucket")
        key = kwargs.get("key") or self.config.get("key")
        
        if not bucket or not key:
            raise ValueError("Bucket and key not specified")
        
        # Get the file format
        file_format = kwargs.get("file_format") or self.config.get("file_format", "csv")
        
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # Download the file to a temporary location
        import tempfile
        
        with tempfile.NamedTemporaryFile() as temp_file:
            s3_client.download_file(bucket, key, temp_file.name)
            
            # Read the file
            if file_format == "csv":
                return pd.read_csv(temp_file.name, **kwargs.get("read_options", {}))
            elif file_format == "json":
                return pd.read_json(temp_file.name, **kwargs.get("read_options", {}))
            elif file_format == "parquet":
                return pd.read_parquet(temp_file.name, **kwargs.get("read_options", {}))
            elif file_format == "excel":
                return pd.read_excel(temp_file.name, **kwargs.get("read_options", {}))
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
    
    def write(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Write data to S3.
        
        Args:
            data: The data to write.
            **kwargs: Additional arguments.
        """
        # Get the bucket and key
        bucket = kwargs.get("bucket") or self.config.get("bucket")
        key = kwargs.get("key") or self.config.get("key")
        
        if not bucket or not key:
            raise ValueError("Bucket and key not specified")
        
        # Get the file format
        file_format = kwargs.get("file_format") or self.config.get("file_format", "csv")
        
        # Get the S3 client
        s3_client = self._get_s3_client()
        
        # Upload the file from a temporary location
        import tempfile
        
        with tempfile.NamedTemporaryFile() as temp_file:
            # Write the file
            if file_format == "csv":
                data.to_csv(temp_file.name, **kwargs.get("write_options", {}))
            elif file_format == "json":
                data.to_json(temp_file.name, **kwargs.get("write_options", {}))
            elif file_format == "parquet":
                data.to_parquet(temp_file.name, **kwargs.get("write_options", {}))
            elif file_format == "excel":
                data.to_excel(temp_file.name, **kwargs.get("write_options", {}))
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Upload the file
            s3_client.upload_file(temp_file.name, bucket, key)
    
    def close(self) -> None:
        """
        Close the connector.
        """
        self.s3_client = None


class KafkaConnector(Connector):
    """
    Kafka connector.
    
    This class implements a connector that reads from and writes to Kafka.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a Kafka connector.
        
        Args:
            name: The connector name.
            config: The connector configuration.
        """
        super().__init__(name, config)
        self.producer = None
        self.consumer = None
    
    def _get_producer(self) -> KafkaProducer:
        """
        Get the Kafka producer.
        
        Returns:
            The Kafka producer.
        """
        if not self.producer:
            # Get the Kafka configuration
            bootstrap_servers = self.config.get("bootstrap_servers", "localhost:9092")
            
            # Create the producer
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                **self.config.get("producer_config", {}),
            )
        
        return self.producer
    
    def _get_consumer(self) -> KafkaConsumer:
        """
        Get the Kafka consumer.
        
        Returns:
            The Kafka consumer.
        """
        if not self.consumer:
            # Get the Kafka configuration
            bootstrap_servers = self.config.get("bootstrap_servers", "localhost:9092")
            topic = self.config.get("topic")
            
            if not topic:
                raise ValueError("Topic not specified")
            
            # Create the consumer
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                **self.config.get("consumer_config", {}),
            )
        
        return self.consumer
    
    def read(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Read data from Kafka.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            An iterator of messages.
        """
        # Get the consumer
        consumer = self._get_consumer()
        
        # Read messages
        for message in consumer:
            yield message.value
    
    def write(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> None:
        """
        Write data to Kafka.
        
        Args:
            data: The data to write.
            **kwargs: Additional arguments.
        """
        # Get the topic
        topic = kwargs.get("topic") or self.config.get("topic")
        
        if not topic:
            raise ValueError("Topic not specified")
        
        # Get the producer
        producer = self._get_producer()
        
        # Write the data
        if isinstance(data, list):
            for item in data:
                producer.send(topic, item)
        else:
            producer.send(topic, data)
        
        # Flush the producer
        producer.flush()
    
    def close(self) -> None:
        """
        Close the connector.
        """
        if self.producer:
            self.producer.close()
            self.producer = None
        
        if self.consumer:
            self.consumer.close()
            self.consumer = None


class ConnectorManager:
    """
    Connector manager.
    
    This class is responsible for managing connectors.
    """
    
    def __init__(self):
        """
        Initialize the connector manager.
        """
        self.connectors: Dict[str, Connector] = {}
        self.lock = threading.RLock()
    
    def register_connector(self, connector: Connector) -> None:
        """
        Register a connector.
        
        Args:
            connector: The connector to register.
        """
        with self.lock:
            self.connectors[connector.name] = connector
    
    def get_connector(self, name: str) -> Optional[Connector]:
        """
        Get a connector by name.
        
        Args:
            name: The connector name.
            
        Returns:
            The connector, or None if the connector is not found.
        """
        with self.lock:
            return self.connectors.get(name)
    
    def get_connectors(self) -> Dict[str, Connector]:
        """
        Get all connectors.
        
        Returns:
            A dictionary mapping connector names to connectors.
        """
        with self.lock:
            return self.connectors.copy()
    
    def create_file_connector(self, name: str, config: Dict[str, Any]) -> FileConnector:
        """
        Create a file connector.
        
        Args:
            name: The connector name.
            config: The connector configuration.
            
        Returns:
            The file connector.
        """
        with self.lock:
            connector = FileConnector(name, config)
            self.register_connector(connector)
            return connector
    
    def create_sql_connector(self, name: str, config: Dict[str, Any]) -> SQLConnector:
        """
        Create a SQL connector.
        
        Args:
            name: The connector name.
            config: The connector configuration.
            
        Returns:
            The SQL connector.
        """
        with self.lock:
            connector = SQLConnector(name, config)
            self.register_connector(connector)
            return connector
    
    def create_s3_connector(self, name: str, config: Dict[str, Any]) -> S3Connector:
        """
        Create an S3 connector.
        
        Args:
            name: The connector name.
            config: The connector configuration.
            
        Returns:
            The S3 connector.
        """
        with self.lock:
            connector = S3Connector(name, config)
            self.register_connector(connector)
            return connector
    
    def create_kafka_connector(self, name: str, config: Dict[str, Any]) -> KafkaConnector:
        """
        Create a Kafka connector.
        
        Args:
            name: The connector name.
            config: The connector configuration.
            
        Returns:
            The Kafka connector.
        """
        with self.lock:
            connector = KafkaConnector(name, config)
            self.register_connector(connector)
            return connector
    
    def close_connector(self, name: str) -> bool:
        """
        Close a connector.
        
        Args:
            name: The connector name.
            
        Returns:
            True if the connector was closed, False otherwise.
        """
        with self.lock:
            connector = self.get_connector(name)
            
            if not connector:
                return False
            
            connector.close()
            del self.connectors[name]
            
            return True
    
    def close_all_connectors(self) -> None:
        """
        Close all connectors.
        """
        with self.lock:
            for name in list(self.connectors.keys()):
                self.close_connector(name)


# Global connector manager instance
_connector_manager = None
_connector_manager_lock = threading.RLock()


def get_connector_manager() -> ConnectorManager:
    """
    Get the global connector manager instance.
    
    Returns:
        The global connector manager instance.
    """
    global _connector_manager
    
    with _connector_manager_lock:
        if _connector_manager is None:
            _connector_manager = ConnectorManager()
        
        return _connector_manager
