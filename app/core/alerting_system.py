"""
Advanced alerting and monitoring system for the Agentic AI platform.

This module provides comprehensive monitoring, alerting, and notification
capabilities for production environments.
"""

import asyncio
import smtplib
import json
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import redis
from collections import defaultdict, deque

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes
    max_alerts_per_hour: int = 10
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    created_at: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    channel_type: str
    config: Dict[str, Any]
    enabled: bool = True


class EmailNotifier:
    """Email notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email', 'alerts@agenticai.com')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            # Create email body
            body = f"""
            Alert: {alert.rule_name}
            Severity: {alert.severity.value.upper()}
            Message: {alert.message}
            Time: {alert.created_at.isoformat()}
            
            Details:
            {json.dumps(alert.details, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class WebhookNotifier:
    """Webhook notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 30)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not self.webhook_url:
            return False
        
        try:
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "details": alert.details,
                "created_at": alert.created_at.isoformat(),
                "status": alert.status.value
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class SlackNotifier:
    """Slack notification handler."""
    
    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'AgenticAI Bot')
        self.icon_emoji = config.get('icon_emoji', ':robot_face:')
        self.timeout = config.get('timeout', 30)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.webhook_url:
            return False
        
        try:
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            color = color_map.get(alert.severity, "good")
            
            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "attachments": [{
                    "color": color,
                    "title": f"Alert: {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Time", "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                        {"title": "Status", "value": alert.status.value, "short": True}
                    ],
                    "footer": "AgenticAI Monitoring",
                    "ts": int(alert.created_at.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class AlertingSystem:
    """Main alerting and monitoring system."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.notifiers: Dict[str, Union[EmailNotifier, WebhookNotifier, SlackNotifier]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, List[datetime]] = defaultdict(list)
        self.is_running = False
        
        # Initialize Redis connection
        self._initialize_redis()
        
        # Setup default notification channels
        self._setup_default_channels()
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established for alerting system")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _setup_default_channels(self):
        """Setup default notification channels."""
        # Email channel
        email_channel = NotificationChannel(
            name="email",
            channel_type="email",
            config={
                "smtp_server": "localhost",
                "smtp_port": 587,
                "from_email": "alerts@agenticai.com",
                "to_emails": ["admin@agenticai.com"]
            }
        )
        self.notification_channels["email"] = email_channel
        self.notifiers["email"] = EmailNotifier(email_channel.config)
        
        # Webhook channel
        webhook_channel = NotificationChannel(
            name="webhook",
            channel_type="webhook",
            config={
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "timeout": 30
            }
        )
        self.notification_channels["webhook"] = webhook_channel
        self.notifiers["webhook"] = WebhookNotifier(webhook_channel.config)
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        # High CPU usage
        self.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            condition=lambda metrics: metrics.get("cpu", {}).get("percent", 0) > 80,
            severity=AlertSeverity.WARNING,
            description="CPU usage is above 80%",
            notification_channels=["email", "webhook"]
        ))
        
        # High memory usage
        self.add_alert_rule(AlertRule(
            name="high_memory_usage",
            condition=lambda metrics: metrics.get("memory", {}).get("percent", 0) > 85,
            severity=AlertSeverity.WARNING,
            description="Memory usage is above 85%",
            notification_channels=["email", "webhook"]
        ))
        
        # High disk usage
        self.add_alert_rule(AlertRule(
            name="high_disk_usage",
            condition=lambda metrics: metrics.get("disk", {}).get("percent", 0) > 90,
            severity=AlertSeverity.CRITICAL,
            description="Disk usage is above 90%",
            notification_channels=["email", "webhook"]
        ))
        
        # High error rate
        self.add_alert_rule(AlertRule(
            name="high_error_rate",
            condition=lambda metrics: metrics.get("requests", {}).get("error_rate", 0) > 0.1,
            severity=AlertSeverity.ERROR,
            description="Error rate is above 10%",
            notification_channels=["email", "webhook"]
        ))
        
        # Service down
        self.add_alert_rule(AlertRule(
            name="service_down",
            condition=lambda metrics: metrics.get("status") == "down",
            severity=AlertSeverity.CRITICAL,
            description="Service is down",
            notification_channels=["email", "webhook"]
        ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.notification_channels[channel.name] = channel
        
        # Create notifier based on channel type
        if channel.channel_type == "email":
            self.notifiers[channel.name] = EmailNotifier(channel.config)
        elif channel.channel_type == "webhook":
            self.notifiers[channel.name] = WebhookNotifier(channel.config)
        elif channel.channel_type == "slack":
            self.notifiers[channel.name] = SlackNotifier(channel.config)
        
        logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_name: str) -> bool:
        """Remove a notification channel."""
        if channel_name in self.notification_channels:
            del self.notification_channels[channel_name]
            if channel_name in self.notifiers:
                del self.notifiers[channel_name]
            logger.info(f"Removed notification channel: {channel_name}")
            return True
        return False
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        if not self.is_running:
            return
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule_name in self.alert_cooldowns:
                if datetime.utcnow() < self.alert_cooldowns[rule_name]:
                    continue
            
            # Check rate limiting
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            recent_alerts = [t for t in self.alert_counts[rule_name] if t > hour_ago]
            
            if len(recent_alerts) >= rule.max_alerts_per_hour:
                continue
            
            # Check condition
            try:
                if rule.condition(metrics):
                    await self._trigger_alert(rule, metrics)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    async def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert."""
        alert_id = f"{rule.name}_{int(datetime.utcnow().timestamp())}"
        
        # Create alert
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=rule.description,
            details=metrics,
            created_at=datetime.utcnow()
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update cooldown
        self.alert_cooldowns[rule.name] = datetime.utcnow() + timedelta(seconds=rule.cooldown_seconds)
        
        # Update alert count
        self.alert_counts[rule.name].append(datetime.utcnow())
        
        # Send notifications
        await self._send_notifications(alert, rule.notification_channels)
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.description}")
    
    async def _send_notifications(self, alert: Alert, channels: List[str]):
        """Send notifications through specified channels."""
        for channel_name in channels:
            if channel_name not in self.notifiers:
                logger.warning(f"Notification channel not found: {channel_name}")
                continue
            
            notifier = self.notifiers[channel_name]
            try:
                success = await notifier.send_notification(alert)
                if success:
                    logger.info(f"Notification sent via {channel_name} for alert {alert.id}")
                else:
                    logger.error(f"Failed to send notification via {channel_name} for alert {alert.id}")
            except Exception as e:
                logger.error(f"Error sending notification via {channel_name}: {e}")
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.utcnow()
        
        logger.info(f"Alert {alert_id} acknowledged by {user}")
        return True
    
    async def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = user
        alert.resolved_at = datetime.utcnow()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved by {user}")
        return True
    
    async def suppress_alert(self, alert_id: str, user: str) -> bool:
        """Suppress an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        
        logger.info(f"Alert {alert_id} suppressed by {user}")
        return True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return list(self.alert_history)[-limit:]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": total_alerts - active_alerts,
            "severity_breakdown": dict(severity_counts),
            "alert_rules": len(self.alert_rules),
            "notification_channels": len(self.notification_channels)
        }
    
    async def start(self):
        """Start the alerting system."""
        self.is_running = True
        logger.info("Alerting system started")
    
    async def stop(self):
        """Stop the alerting system."""
        self.is_running = False
        logger.info("Alerting system stopped")
    
    async def cleanup_old_alerts(self, days: int = 7):
        """Clean up old alerts."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Remove old alerts from history
        old_alerts = [alert for alert in self.alert_history if alert.created_at < cutoff_date]
        for alert in old_alerts:
            self.alert_history.remove(alert)
        
        # Remove old alert counts
        for rule_name, timestamps in self.alert_counts.items():
            self.alert_counts[rule_name] = [t for t in timestamps if t > cutoff_date]
        
        logger.info(f"Cleaned up {len(old_alerts)} old alerts")


# Global alerting system instance
_alerting_system: Optional[AlertingSystem] = None


def get_alerting_system() -> AlertingSystem:
    """Get the global alerting system."""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    return _alerting_system


# Convenience functions
async def check_system_alerts(metrics: Dict[str, Any]):
    """Check system alerts with current metrics."""
    system = get_alerting_system()
    await system.check_alerts(metrics)


async def trigger_manual_alert(rule_name: str, message: str, severity: AlertSeverity, 
                             details: Dict[str, Any]):
    """Trigger a manual alert."""
    system = get_alerting_system()
    
    if rule_name not in system.alert_rules:
        logger.error(f"Alert rule not found: {rule_name}")
        return False
    
    rule = system.alert_rules[rule_name]
    await system._trigger_alert(rule, details)
    return True


# Export all components
__all__ = [
    "AlertSeverity", "AlertStatus", "AlertRule", "Alert", "NotificationChannel",
    "EmailNotifier", "WebhookNotifier", "SlackNotifier", "AlertingSystem",
    "get_alerting_system", "check_system_alerts", "trigger_manual_alert"
]


