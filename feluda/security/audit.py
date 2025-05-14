"""
Audit module for Feluda.

This module provides audit logging for security events.
"""

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
from feluda.observability import get_logger

log = get_logger(__name__)


class AuditEventType(str, enum.Enum):
    """Enum for audit event types."""
    
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    USER_MANAGEMENT = "user_management"
    CUSTOM = "custom"


class AuditEvent(BaseModel):
    """
    Audit event.
    
    This class represents an audit event in the system.
    """
    
    id: str = Field(..., description="The event ID")
    type: AuditEventType = Field(..., description="The event type")
    action: str = Field(..., description="The action performed")
    timestamp: float = Field(..., description="The event timestamp")
    user_id: Optional[str] = Field(None, description="The user ID")
    resource: Optional[str] = Field(None, description="The resource affected")
    status: str = Field(..., description="The event status (success/failure)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    source_ip: Optional[str] = Field(None, description="The source IP address")
    user_agent: Optional[str] = Field(None, description="The user agent")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the audit event to a dictionary.
        
        Returns:
            A dictionary representation of the audit event.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """
        Create an audit event from a dictionary.
        
        Args:
            data: The dictionary to create the audit event from.
            
        Returns:
            An audit event.
        """
        return cls(**data)
    
    @classmethod
    def create(
        cls,
        type: AuditEventType,
        action: str,
        status: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> "AuditEvent":
        """
        Create a new audit event.
        
        Args:
            type: The event type.
            action: The action performed.
            status: The event status.
            user_id: The user ID.
            resource: The resource affected.
            details: Additional details.
            source_ip: The source IP address.
            user_agent: The user agent.
            
        Returns:
            A new audit event.
        """
        return cls(
            id=str(uuid.uuid4()),
            type=type,
            action=action,
            timestamp=time.time(),
            user_id=user_id,
            resource=resource,
            status=status,
            details=details or {},
            source_ip=source_ip,
            user_agent=user_agent,
        )


class AuditManager:
    """
    Audit manager.
    
    This class is responsible for managing audit events.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the audit manager.
        
        Args:
            db_path: The path to the SQLite database.
        """
        self.db_path = db_path or get_config().audit_db or "security/audit.db"
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
            
            # Create the audit events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    action TEXT,
                    timestamp REAL,
                    user_id TEXT,
                    resource TEXT,
                    status TEXT,
                    details TEXT,
                    source_ip TEXT,
                    user_agent TEXT
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events (type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events (user_id)")
            
            # Commit the changes
            conn.commit()
    
    def log_event(
        self,
        type: AuditEventType,
        action: str,
        status: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            type: The event type.
            action: The action performed.
            status: The event status.
            user_id: The user ID.
            resource: The resource affected.
            details: Additional details.
            source_ip: The source IP address.
            user_agent: The user agent.
            
        Returns:
            The logged audit event.
        """
        with self.lock:
            # Create the audit event
            event = AuditEvent.create(
                type=type,
                action=action,
                status=status,
                user_id=user_id,
                resource=resource,
                details=details,
                source_ip=source_ip,
                user_agent=user_agent,
            )
            
            # Save the audit event
            self._save_event(event)
            
            return event
    
    def _save_event(self, event: AuditEvent) -> None:
        """
        Save an audit event to the database.
        
        Args:
            event: The audit event to save.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Save the audit event
            conn.execute(
                """
                INSERT INTO audit_events (
                    id, type, action, timestamp, user_id, resource, status, details, source_ip, user_agent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.type,
                    event.action,
                    event.timestamp,
                    event.user_id,
                    event.resource,
                    event.status,
                    json.dumps(event.details),
                    event.source_ip,
                    event.user_agent,
                ),
            )
            
            # Commit the changes
            conn.commit()
    
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """
        Get an audit event by ID.
        
        Args:
            event_id: The audit event ID.
            
        Returns:
            The audit event, or None if the event is not found.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Get the audit event
            cursor = conn.execute(
                "SELECT * FROM audit_events WHERE id = ?",
                (event_id,),
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Create the audit event
            return AuditEvent(
                id=row["id"],
                type=row["type"],
                action=row["action"],
                timestamp=row["timestamp"],
                user_id=row["user_id"],
                resource=row["resource"],
                status=row["status"],
                details=json.loads(row["details"]),
                source_ip=row["source_ip"],
                user_agent=row["user_agent"],
            )
    
    def get_events(
        self,
        type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """
        Get audit events.
        
        Args:
            type: The event type.
            user_id: The user ID.
            start_time: The start timestamp.
            end_time: The end timestamp.
            limit: The maximum number of events to return.
            offset: The number of events to skip.
            
        Returns:
            A list of audit events.
        """
        with self.lock:
            # Get a connection to the database
            conn = self._get_connection()
            
            # Build the query
            query = "SELECT * FROM audit_events"
            params = []
            
            conditions = []
            
            if type:
                conditions.append("type = ?")
                params.append(type)
            
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute the query
            cursor = conn.execute(query, params)
            
            # Create the audit events
            return [
                AuditEvent(
                    id=row["id"],
                    type=row["type"],
                    action=row["action"],
                    timestamp=row["timestamp"],
                    user_id=row["user_id"],
                    resource=row["resource"],
                    status=row["status"],
                    details=json.loads(row["details"]),
                    source_ip=row["source_ip"],
                    user_agent=row["user_agent"],
                )
                for row in cursor.fetchall()
            ]


# Global audit manager instance
_audit_manager = None
_audit_manager_lock = threading.RLock()


def get_audit_manager() -> AuditManager:
    """
    Get the global audit manager instance.
    
    Returns:
        The global audit manager instance.
    """
    global _audit_manager
    
    with _audit_manager_lock:
        if _audit_manager is None:
            _audit_manager = AuditManager()
        
        return _audit_manager
