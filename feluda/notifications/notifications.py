"""
Notifications for Feluda.

This module provides notifications for alerts and events.
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
from feluda.notifications.channels import NotificationChannel, get_notification_channel
from feluda.notifications.templates import TemplateManager, get_template_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class NotificationPriority(str, enum.Enum):
    """Enum for notification priorities."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(str, enum.Enum):
    """Enum for notification statuses."""
    
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"
    READ = "read"


class Notification(BaseModel):
    """
    Notification.
    
    This class represents a notification.
    """
    
    id: str = Field(..., description="The notification ID")
    title: str = Field(..., description="The notification title")
    message: str = Field(..., description="The notification message")
    priority: NotificationPriority = Field(..., description="The notification priority")
    status: NotificationStatus = Field(..., description="The notification status")
    recipient: str = Field(..., description="The notification recipient")
    channel: str = Field(..., description="The notification channel")
    template: Optional[str] = Field(None, description="The notification template")
    data: Dict[str, Any] = Field(default_factory=dict, description="The notification data")
    created_at: float = Field(..., description="The creation timestamp")
    updated_at: float = Field(..., description="The update timestamp")
    sent_at: Optional[float] = Field(None, description="The sent timestamp")
    delivered_at: Optional[float] = Field(None, description="The delivered timestamp")
    read_at: Optional[float] = Field(None, description="The read timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the notification to a dictionary.
        
        Returns:
            A dictionary representation of the notification.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Notification":
        """
        Create a notification from a dictionary.
        
        Args:
            data: The dictionary to create the notification from.
            
        Returns:
            A notification.
        """
        return cls(**data)


class NotificationManager:
    """
    Notification manager.
    
    This class is responsible for managing notifications.
    """
    
    def __init__(
        self,
        template_manager: Optional[TemplateManager] = None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize the notification manager.
        
        Args:
            template_manager: The template manager.
            db_path: The path to the SQLite database.
        """
        self.template_manager = template_manager or get_template_manager()
        self.db_path = db_path or get_config().notification_db or "notifications/notifications.db"
        self.conn = None
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        
        # Create the database if it doesn't exist
        self._create_database()
        
        # Start the notification sender
        self.start()
    
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
            
            # Create the notifications table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    message TEXT,
                    priority TEXT,
                    status TEXT,
                    recipient TEXT,
                    channel TEXT,
                    template TEXT,
                    data TEXT,
                    created_at REAL,
                    updated_at REAL,
                    sent_at REAL,
                    delivered_at REAL,
                    read_at REAL
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_recipient ON notifications (recipient)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications (created_at)")
            
            # Commit the changes
            conn.commit()
    
    def create_notification(
        self,
        title: str,
        message: str,
        recipient: str,
        channel: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        template: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """
        Create a notification.
        
        Args:
            title: The notification title.
            message: The notification message.
            recipient: The notification recipient.
            channel: The notification channel.
            priority: The notification priority.
            template: The notification template.
            data: The notification data.
            
        Returns:
            The created notification.
        """
        with self.lock:
            # Create the notification
            now = time.time()
            
            notification = Notification(
                id=str(uuid.uuid4()),
                title=title,
                message=message,
                priority=priority,
                status=NotificationStatus.PENDING,
                recipient=recipient,
                channel=channel,
                template=template,
                data=data or {},
                created_at=now,
                updated_at=now,
            )
            
            # Save the notification
            self._save_notification(notification)
            
            return notification
    
    def _save_notification(self, notification: Notification) -> None:
        """
        Save a notification to the database.
        
        Args:
            notification: The notification to save.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Save the notification
            conn.execute(
                """
                INSERT OR REPLACE INTO notifications (
                    id, title, message, priority, status, recipient, channel, template, data,
                    created_at, updated_at, sent_at, delivered_at, read_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notification.id,
                    notification.title,
                    notification.message,
                    notification.priority,
                    notification.status,
                    notification.recipient,
                    notification.channel,
                    notification.template,
                    json.dumps(notification.data),
                    notification.created_at,
                    notification.updated_at,
                    notification.sent_at,
                    notification.delivered_at,
                    notification.read_at,
                ),
            )
            
            # Commit the changes
            conn.commit()
    
    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """
        Get a notification by ID.
        
        Args:
            notification_id: The notification ID.
            
        Returns:
            The notification, or None if the notification is not found.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get the notification
            cursor = conn.execute(
                "SELECT * FROM notifications WHERE id = ?",
                (notification_id,),
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Create the notification
            return Notification(
                id=row["id"],
                title=row["title"],
                message=row["message"],
                priority=row["priority"],
                status=row["status"],
                recipient=row["recipient"],
                channel=row["channel"],
                template=row["template"],
                data=json.loads(row["data"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                sent_at=row["sent_at"],
                delivered_at=row["delivered_at"],
                read_at=row["read_at"],
            )
    
    def get_notifications(
        self,
        recipient: Optional[str] = None,
        status: Optional[NotificationStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Notification]:
        """
        Get notifications.
        
        Args:
            recipient: The notification recipient.
            status: The notification status.
            limit: The maximum number of notifications to return.
            offset: The number of notifications to skip.
            
        Returns:
            A list of notifications.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Build the query
            query = "SELECT * FROM notifications"
            params = []
            
            conditions = []
            
            if recipient:
                conditions.append("recipient = ?")
                params.append(recipient)
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute the query
            cursor = conn.execute(query, params)
            
            # Create the notifications
            return [
                Notification(
                    id=row["id"],
                    title=row["title"],
                    message=row["message"],
                    priority=row["priority"],
                    status=row["status"],
                    recipient=row["recipient"],
                    channel=row["channel"],
                    template=row["template"],
                    data=json.loads(row["data"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    sent_at=row["sent_at"],
                    delivered_at=row["delivered_at"],
                    read_at=row["read_at"],
                )
                for row in cursor.fetchall()
            ]
    
    def update_notification_status(
        self,
        notification_id: str,
        status: NotificationStatus,
    ) -> bool:
        """
        Update the status of a notification.
        
        Args:
            notification_id: The notification ID.
            status: The new notification status.
            
        Returns:
            True if the notification status was updated, False otherwise.
        """
        with self.lock:
            # Get the notification
            notification = self.get_notification(notification_id)
            
            if not notification:
                return False
            
            # Update the notification status
            notification.status = status
            notification.updated_at = time.time()
            
            # Update the timestamp
            if status == NotificationStatus.SENT:
                notification.sent_at = time.time()
            elif status == NotificationStatus.DELIVERED:
                notification.delivered_at = time.time()
            elif status == NotificationStatus.READ:
                notification.read_at = time.time()
            
            # Save the notification
            self._save_notification(notification)
            
            return True
    
    def mark_as_read(self, notification_id: str) -> bool:
        """
        Mark a notification as read.
        
        Args:
            notification_id: The notification ID.
            
        Returns:
            True if the notification was marked as read, False otherwise.
        """
        return self.update_notification_status(notification_id, NotificationStatus.READ)
    
    def delete_notification(self, notification_id: str) -> bool:
        """
        Delete a notification.
        
        Args:
            notification_id: The notification ID.
            
        Returns:
            True if the notification was deleted, False otherwise.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Delete the notification
            cursor = conn.execute(
                "DELETE FROM notifications WHERE id = ?",
                (notification_id,),
            )
            
            # Commit the changes
            conn.commit()
            
            return cursor.rowcount > 0
    
    def start(self) -> None:
        """
        Start the notification sender.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the notification sender.
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
        Run the notification sender.
        """
        while self.running:
            try:
                # Get pending notifications
                notifications = self.get_notifications(status=NotificationStatus.PENDING, limit=10)
                
                # Send each notification
                for notification in notifications:
                    self._send_notification(notification)
                
                # Sleep for a while
                time.sleep(1)
            
            except Exception as e:
                log.error(f"Error in notification sender: {e}")
                time.sleep(1)
    
    def _send_notification(self, notification: Notification) -> None:
        """
        Send a notification.
        
        Args:
            notification: The notification to send.
        """
        try:
            # Get the notification channel
            channel = get_notification_channel(notification.channel)
            
            if not channel:
                log.error(f"Notification channel {notification.channel} not found")
                self.update_notification_status(notification.id, NotificationStatus.FAILED)
                return
            
            # Render the notification
            if notification.template:
                # Get the template
                template = self.template_manager.get_template(notification.template)
                
                if not template:
                    log.error(f"Notification template {notification.template} not found")
                    self.update_notification_status(notification.id, NotificationStatus.FAILED)
                    return
                
                # Render the template
                title = template.render_title(notification.data)
                message = template.render_message(notification.data)
            else:
                title = notification.title
                message = notification.message
            
            # Send the notification
            success = channel.send(
                recipient=notification.recipient,
                title=title,
                message=message,
                data=notification.data,
            )
            
            # Update the notification status
            if success:
                self.update_notification_status(notification.id, NotificationStatus.SENT)
            else:
                self.update_notification_status(notification.id, NotificationStatus.FAILED)
        
        except Exception as e:
            log.error(f"Error sending notification {notification.id}: {e}")
            self.update_notification_status(notification.id, NotificationStatus.FAILED)


# Global notification manager instance
_notification_manager = None
_notification_manager_lock = threading.RLock()


def get_notification_manager() -> NotificationManager:
    """
    Get the global notification manager instance.
    
    Returns:
        The global notification manager instance.
    """
    global _notification_manager
    
    with _notification_manager_lock:
        if _notification_manager is None:
            _notification_manager = NotificationManager()
        
        return _notification_manager
