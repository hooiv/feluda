"""
Alerts module for Feluda.

This module provides alerts and notifications for Feluda.
"""

import abc
import enum
import json
import logging
import smtplib
import threading
import time
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import requests
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.monitoring.health import HealthStatus, get_health_check_manager
from feluda.monitoring.metrics import Metric, get_metric_manager
from feluda.observability import get_logger

log = get_logger(__name__)


class AlertLevel(str, enum.Enum):
    """Enum for alert levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, enum.Enum):
    """Enum for alert status."""
    
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


class AlertRule(BaseModel):
    """
    Alert rule.
    
    This class represents an alert rule.
    """
    
    id: str = Field(..., description="The alert rule ID")
    name: str = Field(..., description="The alert rule name")
    description: str = Field(..., description="The alert rule description")
    level: AlertLevel = Field(..., description="The alert level")
    condition: str = Field(..., description="The alert condition")
    check_interval: float = Field(60.0, description="The check interval in seconds")
    notification_channels: List[str] = Field(default_factory=list, description="The notification channels")
    enabled: bool = Field(True, description="Whether the alert rule is enabled")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the alert rule to a dictionary.
        
        Returns:
            A dictionary representation of the alert rule.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertRule":
        """
        Create an alert rule from a dictionary.
        
        Args:
            data: The dictionary to create the alert rule from.
            
        Returns:
            An alert rule.
        """
        return cls(**data)
    
    def evaluate(self) -> bool:
        """
        Evaluate the alert condition.
        
        Returns:
            True if the condition is met, False otherwise.
        """
        try:
            # Get the metric manager
            metric_manager = get_metric_manager()
            
            # Get the health check manager
            health_check_manager = get_health_check_manager()
            
            # Create a context for the condition
            context = {
                "metrics": metric_manager.get_metrics(),
                "health_checks": health_check_manager.get_health_checks(),
                "HealthStatus": HealthStatus,
            }
            
            # Evaluate the condition
            return eval(self.condition, {"__builtins__": {}}, context)
        
        except Exception as e:
            log.error(f"Error evaluating alert condition: {e}")
            return False


class Alert(BaseModel):
    """
    Alert.
    
    This class represents an alert.
    """
    
    id: str = Field(..., description="The alert ID")
    rule_id: str = Field(..., description="The alert rule ID")
    level: AlertLevel = Field(..., description="The alert level")
    status: AlertStatus = Field(..., description="The alert status")
    message: str = Field(..., description="The alert message")
    details: Dict[str, Any] = Field(default_factory=dict, description="The alert details")
    created_at: float = Field(..., description="The alert creation timestamp")
    updated_at: float = Field(..., description="The alert update timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the alert to a dictionary.
        
        Returns:
            A dictionary representation of the alert.
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """
        Create an alert from a dictionary.
        
        Args:
            data: The dictionary to create the alert from.
            
        Returns:
            An alert.
        """
        return cls(**data)
    
    @classmethod
    def create(cls, rule_id: str, level: AlertLevel, message: str, details: Optional[Dict[str, Any]] = None) -> "Alert":
        """
        Create a new alert.
        
        Args:
            rule_id: The alert rule ID.
            level: The alert level.
            message: The alert message.
            details: The alert details.
            
        Returns:
            A new alert.
        """
        now = time.time()
        
        return cls(
            id=str(uuid.uuid4()),
            rule_id=rule_id,
            level=level,
            status=AlertStatus.ACTIVE,
            message=message,
            details=details or {},
            created_at=now,
            updated_at=now,
        )


class NotificationChannel(abc.ABC):
    """
    Base class for notification channels.
    
    This class defines the interface for notification channels.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize a notification channel.
        
        Args:
            name: The notification channel name.
            config: The notification channel configuration.
        """
        self.name = name
        self.config = config
    
    @abc.abstractmethod
    def send_notification(self, notification: "Notification") -> bool:
        """
        Send a notification.
        
        Args:
            notification: The notification to send.
            
        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        pass


class EmailNotificationChannel(NotificationChannel):
    """
    Email notification channel.
    
    This class implements a notification channel that sends notifications via email.
    """
    
    def send_notification(self, notification: "Notification") -> bool:
        """
        Send a notification via email.
        
        Args:
            notification: The notification to send.
            
        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        try:
            # Get the email configuration
            smtp_host = self.config.get("smtp_host", "localhost")
            smtp_port = self.config.get("smtp_port", 25)
            smtp_username = self.config.get("smtp_username")
            smtp_password = self.config.get("smtp_password")
            from_email = self.config.get("from_email", "feluda@example.com")
            to_emails = self.config.get("to_emails", [])
            
            if not to_emails:
                log.error("No recipients specified for email notification")
                return False
            
            # Create the email message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = f"[{notification.alert.level.upper()}] {notification.alert.message}"
            
            # Add the email body
            body = f"""
            Alert: {notification.alert.message}
            Level: {notification.alert.level}
            Status: {notification.alert.status}
            Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(notification.alert.created_at))}
            Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(notification.alert.updated_at))}
            
            Details:
            {json.dumps(notification.alert.details, indent=2)}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            # Send the email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if smtp_username and smtp_password:
                    server.login(smtp_username, smtp_password)
                
                server.send_message(msg)
            
            return True
        
        except Exception as e:
            log.error(f"Error sending email notification: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """
    Slack notification channel.
    
    This class implements a notification channel that sends notifications via Slack.
    """
    
    def send_notification(self, notification: "Notification") -> bool:
        """
        Send a notification via Slack.
        
        Args:
            notification: The notification to send.
            
        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        try:
            # Get the Slack configuration
            webhook_url = self.config.get("webhook_url")
            
            if not webhook_url:
                log.error("No webhook URL specified for Slack notification")
                return False
            
            # Create the Slack message
            message = {
                "text": f"[{notification.alert.level.upper()}] {notification.alert.message}",
                "attachments": [
                    {
                        "color": self._get_color(notification.alert.level),
                        "fields": [
                            {
                                "title": "Level",
                                "value": notification.alert.level,
                                "short": True,
                            },
                            {
                                "title": "Status",
                                "value": notification.alert.status,
                                "short": True,
                            },
                            {
                                "title": "Created",
                                "value": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(notification.alert.created_at)),
                                "short": True,
                            },
                            {
                                "title": "Updated",
                                "value": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(notification.alert.updated_at)),
                                "short": True,
                            },
                            {
                                "title": "Details",
                                "value": f"```{json.dumps(notification.alert.details, indent=2)}```",
                                "short": False,
                            },
                        ],
                    },
                ],
            }
            
            # Send the Slack message
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
            
            return True
        
        except Exception as e:
            log.error(f"Error sending Slack notification: {e}")
            return False
    
    def _get_color(self, level: AlertLevel) -> str:
        """
        Get the color for an alert level.
        
        Args:
            level: The alert level.
            
        Returns:
            The color.
        """
        if level == AlertLevel.INFO:
            return "#2196F3"  # Blue
        elif level == AlertLevel.WARNING:
            return "#FFC107"  # Yellow
        elif level == AlertLevel.ERROR:
            return "#F44336"  # Red
        elif level == AlertLevel.CRITICAL:
            return "#9C27B0"  # Purple
        else:
            return "#9E9E9E"  # Grey


class WebhookNotificationChannel(NotificationChannel):
    """
    Webhook notification channel.
    
    This class implements a notification channel that sends notifications via webhook.
    """
    
    def send_notification(self, notification: "Notification") -> bool:
        """
        Send a notification via webhook.
        
        Args:
            notification: The notification to send.
            
        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        try:
            # Get the webhook configuration
            url = self.config.get("url")
            headers = self.config.get("headers", {})
            
            if not url:
                log.error("No URL specified for webhook notification")
                return False
            
            # Create the webhook payload
            payload = {
                "alert": notification.alert.to_dict(),
                "timestamp": time.time(),
            }
            
            # Send the webhook request
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            return True
        
        except Exception as e:
            log.error(f"Error sending webhook notification: {e}")
            return False


class Notification(BaseModel):
    """
    Notification.
    
    This class represents a notification.
    """
    
    id: str = Field(..., description="The notification ID")
    alert: Alert = Field(..., description="The alert")
    channel: str = Field(..., description="The notification channel")
    status: str = Field(..., description="The notification status")
    created_at: float = Field(..., description="The notification creation timestamp")
    sent_at: Optional[float] = Field(None, description="The notification sent timestamp")
    error: Optional[str] = Field(None, description="The notification error")
    
    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the notification to a dictionary.
        
        Returns:
            A dictionary representation of the notification.
        """
        return {
            "id": self.id,
            "alert": self.alert.to_dict(),
            "channel": self.channel,
            "status": self.status,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
            "error": self.error,
        }
    
    @classmethod
    def create(cls, alert: Alert, channel: str) -> "Notification":
        """
        Create a new notification.
        
        Args:
            alert: The alert.
            channel: The notification channel.
            
        Returns:
            A new notification.
        """
        return cls(
            id=str(uuid.uuid4()),
            alert=alert,
            channel=channel,
            status="pending",
            created_at=time.time(),
        )


class AlertManager:
    """
    Alert manager.
    
    This class is responsible for managing alerts and notifications.
    """
    
    def __init__(self):
        """
        Initialize the alert manager.
        """
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.notifications: Dict[str, Notification] = {}
        self.channels: Dict[str, NotificationChannel] = {}
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        
        # Register default notification channels
        self.register_notification_channel(
            "email",
            EmailNotificationChannel(
                name="email",
                config={
                    "smtp_host": "localhost",
                    "smtp_port": 25,
                    "from_email": "feluda@example.com",
                    "to_emails": ["admin@example.com"],
                },
            ),
        )
        
        self.register_notification_channel(
            "slack",
            SlackNotificationChannel(
                name="slack",
                config={
                    "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz",
                },
            ),
        )
        
        self.register_notification_channel(
            "webhook",
            WebhookNotificationChannel(
                name="webhook",
                config={
                    "url": "https://example.com/webhook",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer xxx",
                    },
                },
            ),
        )
    
    def register_alert_rule(self, rule: AlertRule) -> None:
        """
        Register an alert rule.
        
        Args:
            rule: The alert rule to register.
        """
        with self.lock:
            self.rules[rule.id] = rule
    
    def get_alert_rule(self, rule_id: str) -> Optional[AlertRule]:
        """
        Get an alert rule by ID.
        
        Args:
            rule_id: The alert rule ID.
            
        Returns:
            The alert rule, or None if the alert rule is not found.
        """
        with self.lock:
            return self.rules.get(rule_id)
    
    def get_alert_rules(self) -> Dict[str, AlertRule]:
        """
        Get all alert rules.
        
        Returns:
            A dictionary mapping alert rule IDs to alert rules.
        """
        with self.lock:
            return self.rules.copy()
    
    def register_notification_channel(self, name: str, channel: NotificationChannel) -> None:
        """
        Register a notification channel.
        
        Args:
            name: The notification channel name.
            channel: The notification channel.
        """
        with self.lock:
            self.channels[name] = channel
    
    def get_notification_channel(self, name: str) -> Optional[NotificationChannel]:
        """
        Get a notification channel by name.
        
        Args:
            name: The notification channel name.
            
        Returns:
            The notification channel, or None if the notification channel is not found.
        """
        with self.lock:
            return self.channels.get(name)
    
    def get_notification_channels(self) -> Dict[str, NotificationChannel]:
        """
        Get all notification channels.
        
        Returns:
            A dictionary mapping notification channel names to notification channels.
        """
        with self.lock:
            return self.channels.copy()
    
    def create_alert(self, rule_id: str, message: str, details: Optional[Dict[str, Any]] = None) -> Alert:
        """
        Create an alert.
        
        Args:
            rule_id: The alert rule ID.
            message: The alert message.
            details: The alert details.
            
        Returns:
            The created alert.
        """
        with self.lock:
            # Get the alert rule
            rule = self.get_alert_rule(rule_id)
            
            if not rule:
                raise ValueError(f"Alert rule {rule_id} not found")
            
            # Create the alert
            alert = Alert.create(rule_id, rule.level, message, details)
            
            # Store the alert
            self.alerts[alert.id] = alert
            
            # Create notifications
            for channel_name in rule.notification_channels:
                channel = self.get_notification_channel(channel_name)
                
                if not channel:
                    log.warning(f"Notification channel {channel_name} not found")
                    continue
                
                notification = Notification.create(alert, channel_name)
                self.notifications[notification.id] = notification
            
            return alert
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """
        Get an alert by ID.
        
        Args:
            alert_id: The alert ID.
            
        Returns:
            The alert, or None if the alert is not found.
        """
        with self.lock:
            return self.alerts.get(alert_id)
    
    def get_alerts(self, status: Optional[AlertStatus] = None) -> Dict[str, Alert]:
        """
        Get alerts.
        
        Args:
            status: The alert status. If None, get all alerts.
            
        Returns:
            A dictionary mapping alert IDs to alerts.
        """
        with self.lock:
            if status:
                return {
                    alert_id: alert
                    for alert_id, alert in self.alerts.items()
                    if alert.status == status
                }
            
            return self.alerts.copy()
    
    def update_alert_status(self, alert_id: str, status: AlertStatus) -> bool:
        """
        Update the status of an alert.
        
        Args:
            alert_id: The alert ID.
            status: The new alert status.
            
        Returns:
            True if the alert status was updated, False otherwise.
        """
        with self.lock:
            alert = self.get_alert(alert_id)
            
            if not alert:
                return False
            
            # Update the alert status
            alert.status = status
            alert.updated_at = time.time()
            
            return True
    
    def start(self) -> None:
        """
        Start the alert manager.
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._run_alert_manager)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self) -> None:
        """
        Stop the alert manager.
        """
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            if self.thread:
                self.thread.join()
                self.thread = None
    
    def _run_alert_manager(self) -> None:
        """
        Run the alert manager.
        """
        while self.running:
            try:
                # Check alert rules
                self._check_alert_rules()
                
                # Send notifications
                self._send_notifications()
                
                # Sleep for a short time
                time.sleep(1.0)
            
            except Exception as e:
                log.error(f"Error running alert manager: {e}")
                time.sleep(1.0)
    
    def _check_alert_rules(self) -> None:
        """
        Check alert rules.
        """
        # Get the alert rules
        rules = self.get_alert_rules()
        
        # Check each alert rule
        for rule_id, rule in rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check if the rule condition is met
                if rule.evaluate():
                    # Create an alert
                    self.create_alert(
                        rule_id=rule_id,
                        message=rule.description,
                        details={
                            "rule_name": rule.name,
                            "rule_description": rule.description,
                            "rule_level": rule.level,
                            "rule_condition": rule.condition,
                        },
                    )
            
            except Exception as e:
                log.error(f"Error checking alert rule {rule_id}: {e}")
    
    def _send_notifications(self) -> None:
        """
        Send notifications.
        """
        # Get the notifications
        notifications = self.get_notifications(status="pending")
        
        # Send each notification
        for notification_id, notification in notifications.items():
            try:
                # Get the notification channel
                channel = self.get_notification_channel(notification.channel)
                
                if not channel:
                    log.warning(f"Notification channel {notification.channel} not found")
                    notification.status = "failed"
                    notification.error = f"Notification channel {notification.channel} not found"
                    continue
                
                # Send the notification
                success = channel.send_notification(notification)
                
                if success:
                    notification.status = "sent"
                    notification.sent_at = time.time()
                else:
                    notification.status = "failed"
                    notification.error = "Failed to send notification"
            
            except Exception as e:
                log.error(f"Error sending notification {notification_id}: {e}")
                notification.status = "failed"
                notification.error = str(e)
    
    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """
        Get a notification by ID.
        
        Args:
            notification_id: The notification ID.
            
        Returns:
            The notification, or None if the notification is not found.
        """
        with self.lock:
            return self.notifications.get(notification_id)
    
    def get_notifications(self, status: Optional[str] = None) -> Dict[str, Notification]:
        """
        Get notifications.
        
        Args:
            status: The notification status. If None, get all notifications.
            
        Returns:
            A dictionary mapping notification IDs to notifications.
        """
        with self.lock:
            if status:
                return {
                    notification_id: notification
                    for notification_id, notification in self.notifications.items()
                    if notification.status == status
                }
            
            return self.notifications.copy()


# Global alert manager instance
_alert_manager = None
_alert_manager_lock = threading.RLock()


def get_alert_manager() -> AlertManager:
    """
    Get the global alert manager instance.
    
    Returns:
        The global alert manager instance.
    """
    global _alert_manager
    
    with _alert_manager_lock:
        if _alert_manager is None:
            _alert_manager = AlertManager()
            _alert_manager.start()
        
        return _alert_manager
