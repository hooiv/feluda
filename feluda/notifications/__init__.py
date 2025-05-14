"""
Notification system for Feluda.

This module provides a notification system for alerts and events.
"""

from feluda.notifications.notifications import (
    Notification,
    NotificationManager,
    NotificationPriority,
    NotificationStatus,
    get_notification_manager,
)
from feluda.notifications.channels import (
    NotificationChannel,
    EmailChannel,
    SlackChannel,
    WebhookChannel,
    SMSChannel,
    PushChannel,
    get_notification_channel,
)
from feluda.notifications.templates import (
    NotificationTemplate,
    TemplateManager,
    get_template_manager,
)
from feluda.notifications.subscriptions import (
    Subscription,
    SubscriptionManager,
    get_subscription_manager,
)

__all__ = [
    "EmailChannel",
    "Notification",
    "NotificationChannel",
    "NotificationManager",
    "NotificationPriority",
    "NotificationStatus",
    "NotificationTemplate",
    "PushChannel",
    "SMSChannel",
    "SlackChannel",
    "Subscription",
    "SubscriptionManager",
    "TemplateManager",
    "WebhookChannel",
    "get_notification_channel",
    "get_notification_manager",
    "get_subscription_manager",
    "get_template_manager",
]
