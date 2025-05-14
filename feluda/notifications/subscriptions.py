"""
Notification subscriptions for Feluda.

This module provides subscriptions for notifications.
"""

import abc
import enum
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.notifications.notifications import NotificationPriority
from feluda.observability import get_logger

log = get_logger(__name__)


class Subscription(BaseModel):
    """
    Subscription.
    
    This class represents a subscription to notifications.
    """
    
    id: str = Field(..., description="The subscription ID")
    user_id: str = Field(..., description="The user ID")
    channel: str = Field(..., description="The notification channel")
    recipient: str = Field(..., description="The notification recipient")
    event_type: str = Field(..., description="The event type")
    min_priority: NotificationPriority = Field(NotificationPriority.LOW, description="The minimum notification priority")
    filters: Dict[str, Any] = Field(default_factory=dict, description="The subscription filters")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the subscription to a dictionary.
        
        Returns:
            A dictionary representation of the subscription.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subscription":
        """
        Create a subscription from a dictionary.
        
        Args:
            data: The dictionary to create the subscription from.
            
        Returns:
            A subscription.
        """
        return cls(**data)


class SubscriptionManager:
    """
    Subscription manager.
    
    This class is responsible for managing notification subscriptions.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the subscription manager.
        
        Args:
            db_path: The path to the SQLite database.
        """
        self.db_path = db_path or get_config().subscription_db or "notifications/subscriptions.db"
        self.conn = None
        self.lock = threading.RLock()
        
        # Create the database if it doesn't exist
        self._create_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database.
        
        Returns:
            A connection to the SQLite database.
        """
        if not self.conn:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to the database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        
        return self.conn
    
    def _create_database(self) -> None:
        """
        Create the SQLite database.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Create the subscriptions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    channel TEXT,
                    recipient TEXT,
                    event_type TEXT,
                    min_priority TEXT,
                    filters TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions (user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_event_type ON subscriptions (event_type)")
            
            # Commit the changes
            conn.commit()
    
    def create_subscription(
        self,
        user_id: str,
        channel: str,
        recipient: str,
        event_type: str,
        min_priority: NotificationPriority = NotificationPriority.LOW,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Subscription:
        """
        Create a subscription.
        
        Args:
            user_id: The user ID.
            channel: The notification channel.
            recipient: The notification recipient.
            event_type: The event type.
            min_priority: The minimum notification priority.
            filters: The subscription filters.
            
        Returns:
            The created subscription.
        """
        with self.lock:
            # Create the subscription
            now = time.time()
            
            subscription = Subscription(
                id=str(uuid.uuid4()),
                user_id=user_id,
                channel=channel,
                recipient=recipient,
                event_type=event_type,
                min_priority=min_priority,
                filters=filters or {},
                created_at=now,
                updated_at=now,
            )
            
            # Save the subscription
            self._save_subscription(subscription)
            
            return subscription
    
    def _save_subscription(self, subscription: Subscription) -> None:
        """
        Save a subscription to the database.
        
        Args:
            subscription: The subscription to save.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Save the subscription
            conn.execute(
                """
                INSERT OR REPLACE INTO subscriptions (
                    id, user_id, channel, recipient, event_type, min_priority, filters,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subscription.id,
                    subscription.user_id,
                    subscription.channel,
                    subscription.recipient,
                    subscription.event_type,
                    subscription.min_priority,
                    json.dumps(subscription.filters),
                    subscription.created_at,
                    subscription.updated_at,
                ),
            )
            
            # Commit the changes
            conn.commit()
    
    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """
        Get a subscription by ID.
        
        Args:
            subscription_id: The subscription ID.
            
        Returns:
            The subscription, or None if the subscription is not found.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get the subscription
            cursor = conn.execute(
                "SELECT * FROM subscriptions WHERE id = ?",
                (subscription_id,),
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Create the subscription
            return Subscription(
                id=row["id"],
                user_id=row["user_id"],
                channel=row["channel"],
                recipient=row["recipient"],
                event_type=row["event_type"],
                min_priority=row["min_priority"],
                filters=json.loads(row["filters"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
    
    def get_subscriptions(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[Subscription]:
        """
        Get subscriptions.
        
        Args:
            user_id: The user ID.
            event_type: The event type.
            
        Returns:
            A list of subscriptions.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Build the query
            query = "SELECT * FROM subscriptions"
            params = []
            
            conditions = []
            
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            if event_type:
                conditions.append("event_type = ?")
                params.append(event_type)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Execute the query
            cursor = conn.execute(query, params)
            
            # Create the subscriptions
            return [
                Subscription(
                    id=row["id"],
                    user_id=row["user_id"],
                    channel=row["channel"],
                    recipient=row["recipient"],
                    event_type=row["event_type"],
                    min_priority=row["min_priority"],
                    filters=json.loads(row["filters"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in cursor.fetchall()
            ]
    
    def update_subscription(
        self,
        subscription_id: str,
        channel: Optional[str] = None,
        recipient: Optional[str] = None,
        min_priority: Optional[NotificationPriority] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Subscription]:
        """
        Update a subscription.
        
        Args:
            subscription_id: The subscription ID.
            channel: The notification channel.
            recipient: The notification recipient.
            min_priority: The minimum notification priority.
            filters: The subscription filters.
            
        Returns:
            The updated subscription, or None if the subscription is not found.
        """
        with self.lock:
            # Get the subscription
            subscription = self.get_subscription(subscription_id)
            
            if not subscription:
                return None
            
            # Update the subscription
            if channel is not None:
                subscription.channel = channel
            
            if recipient is not None:
                subscription.recipient = recipient
            
            if min_priority is not None:
                subscription.min_priority = min_priority
            
            if filters is not None:
                subscription.filters = filters
            
            subscription.updated_at = time.time()
            
            # Save the subscription
            self._save_subscription(subscription)
            
            return subscription
    
    def delete_subscription(self, subscription_id: str) -> bool:
        """
        Delete a subscription.
        
        Args:
            subscription_id: The subscription ID.
            
        Returns:
            True if the subscription was deleted, False otherwise.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Delete the subscription
            cursor = conn.execute(
                "DELETE FROM subscriptions WHERE id = ?",
                (subscription_id,),
            )
            
            # Commit the changes
            conn.commit()
            
            return cursor.rowcount > 0
    
    def get_matching_subscriptions(
        self,
        event_type: str,
        priority: NotificationPriority,
        event_data: Dict[str, Any],
    ) -> List[Subscription]:
        """
        Get subscriptions that match an event.
        
        Args:
            event_type: The event type.
            priority: The event priority.
            event_data: The event data.
            
        Returns:
            A list of matching subscriptions.
        """
        with self.lock:
            # Get subscriptions for the event type
            subscriptions = self.get_subscriptions(event_type=event_type)
            
            # Filter subscriptions by priority
            priority_values = {
                NotificationPriority.LOW: 0,
                NotificationPriority.MEDIUM: 1,
                NotificationPriority.HIGH: 2,
                NotificationPriority.CRITICAL: 3,
            }
            
            priority_value = priority_values.get(priority, 0)
            
            subscriptions = [
                subscription for subscription in subscriptions
                if priority_value >= priority_values.get(subscription.min_priority, 0)
            ]
            
            # Filter subscriptions by filters
            matching_subscriptions = []
            
            for subscription in subscriptions:
                # Check if the subscription filters match the event data
                if self._match_filters(subscription.filters, event_data):
                    matching_subscriptions.append(subscription)
            
            return matching_subscriptions
    
    def _match_filters(self, filters: Dict[str, Any], event_data: Dict[str, Any]) -> bool:
        """
        Check if filters match event data.
        
        Args:
            filters: The filters.
            event_data: The event data.
            
        Returns:
            True if the filters match the event data, False otherwise.
        """
        for key, value in filters.items():
            # Check if the key exists in the event data
            if key not in event_data:
                return False
            
            # Check if the value matches
            if isinstance(value, list):
                # Check if the event data value is in the list
                if event_data[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Check if the event data value matches the dictionary
                if not isinstance(event_data[key], dict):
                    return False
                
                if not self._match_filters(value, event_data[key]):
                    return False
            else:
                # Check if the event data value equals the filter value
                if event_data[key] != value:
                    return False
        
        return True


# Global subscription manager instance
_subscription_manager = None
_subscription_manager_lock = threading.RLock()


def get_subscription_manager() -> SubscriptionManager:
    """
    Get the global subscription manager instance.
    
    Returns:
        The global subscription manager instance.
    """
    global _subscription_manager
    
    with _subscription_manager_lock:
        if _subscription_manager is None:
            _subscription_manager = SubscriptionManager()
        
        return _subscription_manager
