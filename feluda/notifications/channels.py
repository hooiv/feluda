"""
Notification channels for Feluda.

This module provides channels for sending notifications.
"""

import abc
import enum
import json
import logging
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import requests
from pydantic import BaseModel, Field

from feluda.config import get_config
from feluda.observability import get_logger

log = get_logger(__name__)


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
            name: The channel name.
            config: The channel configuration.
        """
        self.name = name
        self.config = config
    
    @abc.abstractmethod
    def send(
        self,
        recipient: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a notification.
        
        Args:
            recipient: The notification recipient.
            title: The notification title.
            message: The notification message.
            data: The notification data.
            
        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        pass


class EmailChannel(NotificationChannel):
    """
    Email notification channel.
    
    This class implements a notification channel that sends notifications via email.
    """
    
    def send(
        self,
        recipient: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a notification via email.
        
        Args:
            recipient: The notification recipient.
            title: The notification title.
            message: The notification message.
            data: The notification data.
            
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
            
            # Create the email message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = recipient
            msg["Subject"] = title
            
            # Add the email body
            msg.attach(MIMEText(message, "plain"))
            
            # Send the email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if smtp_username and smtp_password:
                    server.login(smtp_username, smtp_password)
                
                server.send_message(msg)
            
            return True
        
        except Exception as e:
            log.error(f"Error sending email notification: {e}")
            return False


class SlackChannel(NotificationChannel):
    """
    Slack notification channel.
    
    This class implements a notification channel that sends notifications via Slack.
    """
    
    def send(
        self,
        recipient: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a notification via Slack.
        
        Args:
            recipient: The notification recipient.
            title: The notification title.
            message: The notification message.
            data: The notification data.
            
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
            slack_message = {
                "text": title,
                "attachments": [
                    {
                        "title": title,
                        "text": message,
                        "color": data.get("color", "#2196F3") if data else "#2196F3",
                    }
                ],
            }
            
            # Send the Slack message
            response = requests.post(webhook_url, json=slack_message)
            response.raise_for_status()
            
            return True
        
        except Exception as e:
            log.error(f"Error sending Slack notification: {e}")
            return False


class WebhookChannel(NotificationChannel):
    """
    Webhook notification channel.
    
    This class implements a notification channel that sends notifications via webhook.
    """
    
    def send(
        self,
        recipient: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a notification via webhook.
        
        Args:
            recipient: The notification recipient.
            title: The notification title.
            message: The notification message.
            data: The notification data.
            
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
                "recipient": recipient,
                "title": title,
                "message": message,
                "data": data or {},
                "timestamp": import time; time.time(),
            }
            
            # Send the webhook request
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            return True
        
        except Exception as e:
            log.error(f"Error sending webhook notification: {e}")
            return False


class SMSChannel(NotificationChannel):
    """
    SMS notification channel.
    
    This class implements a notification channel that sends notifications via SMS.
    """
    
    def send(
        self,
        recipient: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a notification via SMS.
        
        Args:
            recipient: The notification recipient.
            title: The notification title.
            message: The notification message.
            data: The notification data.
            
        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        try:
            # Get the SMS configuration
            provider = self.config.get("provider", "twilio")
            
            if provider == "twilio":
                # Get the Twilio configuration
                account_sid = self.config.get("account_sid")
                auth_token = self.config.get("auth_token")
                from_number = self.config.get("from_number")
                
                if not account_sid or not auth_token or not from_number:
                    log.error("Incomplete Twilio configuration for SMS notification")
                    return False
                
                # Import the Twilio client
                try:
                    from twilio.rest import Client
                except ImportError:
                    log.error("Twilio client not installed")
                    return False
                
                # Create the Twilio client
                client = Client(account_sid, auth_token)
                
                # Send the SMS
                client.messages.create(
                    body=f"{title}\n\n{message}",
                    from_=from_number,
                    to=recipient,
                )
                
                return True
            else:
                log.error(f"Unsupported SMS provider: {provider}")
                return False
        
        except Exception as e:
            log.error(f"Error sending SMS notification: {e}")
            return False


class PushChannel(NotificationChannel):
    """
    Push notification channel.
    
    This class implements a notification channel that sends notifications via push.
    """
    
    def send(
        self,
        recipient: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a notification via push.
        
        Args:
            recipient: The notification recipient.
            title: The notification title.
            message: The notification message.
            data: The notification data.
            
        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        try:
            # Get the push configuration
            provider = self.config.get("provider", "firebase")
            
            if provider == "firebase":
                # Get the Firebase configuration
                api_key = self.config.get("api_key")
                
                if not api_key:
                    log.error("No API key specified for Firebase push notification")
                    return False
                
                # Create the Firebase message
                firebase_message = {
                    "message": {
                        "token": recipient,
                        "notification": {
                            "title": title,
                            "body": message,
                        },
                        "data": data or {},
                    },
                }
                
                # Send the Firebase message
                response = requests.post(
                    "https://fcm.googleapis.com/v1/projects/your-project-id/messages:send",
                    json=firebase_message,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                
                return True
            else:
                log.error(f"Unsupported push provider: {provider}")
                return False
        
        except Exception as e:
            log.error(f"Error sending push notification: {e}")
            return False


# Dictionary of notification channels
_notification_channels: Dict[str, NotificationChannel] = {}
_notification_channels_lock = threading.RLock()


def register_notification_channel(channel: NotificationChannel) -> None:
    """
    Register a notification channel.
    
    Args:
        channel: The notification channel to register.
    """
    with _notification_channels_lock:
        _notification_channels[channel.name] = channel


def get_notification_channel(name: str) -> Optional[NotificationChannel]:
    """
    Get a notification channel by name.
    
    Args:
        name: The notification channel name.
        
    Returns:
        The notification channel, or None if the channel is not found.
    """
    with _notification_channels_lock:
        return _notification_channels.get(name)


def get_notification_channels() -> Dict[str, NotificationChannel]:
    """
    Get all notification channels.
    
    Returns:
        A dictionary mapping notification channel names to channels.
    """
    with _notification_channels_lock:
        return _notification_channels.copy()


# Register default notification channels
register_notification_channel(
    EmailChannel(
        name="email",
        config={
            "smtp_host": "localhost",
            "smtp_port": 25,
            "from_email": "feluda@example.com",
        },
    ),
)

register_notification_channel(
    SlackChannel(
        name="slack",
        config={
            "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz",
        },
    ),
)

register_notification_channel(
    WebhookChannel(
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

register_notification_channel(
    SMSChannel(
        name="sms",
        config={
            "provider": "twilio",
            "account_sid": "xxx",
            "auth_token": "yyy",
            "from_number": "+1234567890",
        },
    ),
)

register_notification_channel(
    PushChannel(
        name="push",
        config={
            "provider": "firebase",
            "api_key": "xxx",
        },
    ),
)
