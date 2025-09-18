"""
Advanced Access Controls for Multi-Agent Architecture.

This module provides sophisticated access control mechanisms including
role-based access control (RBAC), attribute-based access control (ABAC),
dynamic permissions, and comprehensive security auditing.

Features:
- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Dynamic permission evaluation
- Security policy enforcement
- Comprehensive audit logging
- Threat detection and prevention
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

import structlog
from pydantic import BaseModel, Field

from app.rag.core.agent_isolation_manager import AgentIsolationManager, ResourceType

logger = structlog.get_logger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for access control."""
    PUBLIC = "public"                 # No restrictions
    INTERNAL = "internal"             # Internal system access
    CONFIDENTIAL = "confidential"     # Confidential data access
    RESTRICTED = "restricted"         # Highly restricted access
    TOP_SECRET = "top_secret"         # Maximum security level


class AccessAction(str, Enum):
    """Types of access actions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    SHARE = "share"


class PolicyEffect(str, Enum):
    """Policy decision effects."""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


@dataclass
class AccessRule:
    """Individual access control rule."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Rule conditions
    subjects: Set[str] = field(default_factory=set)      # Agent IDs or roles
    resources: Set[str] = field(default_factory=set)     # Resource patterns
    actions: Set[AccessAction] = field(default_factory=set)
    
    # Contextual conditions
    time_constraints: Optional[Dict[str, Any]] = None
    location_constraints: Optional[Dict[str, Any]] = None
    attribute_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Rule effect
    effect: PolicyEffect = PolicyEffect.ALLOW
    conditions: List[str] = field(default_factory=list)
    
    # Metadata
    priority: int = 100  # Higher number = higher priority
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Usage tracking
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class AccessPolicy:
    """Collection of access rules forming a policy."""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Policy rules
    rules: List[AccessRule] = field(default_factory=list)
    
    # Policy metadata
    version: str = "1.0"
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    owner: str = ""
    
    # Policy status
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    # Enforcement settings
    strict_mode: bool = False  # Fail closed on evaluation errors
    audit_all_access: bool = True


@dataclass
class AccessAudit:
    """Access audit log entry."""
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Access attempt details
    subject_id: str = ""
    resource_id: str = ""
    action: AccessAction = AccessAction.READ
    
    # Decision details
    decision: PolicyEffect = PolicyEffect.DENY
    applied_rules: List[str] = field(default_factory=list)  # rule_ids
    evaluation_time_ms: float = 0.0
    
    # Context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    # Additional metadata
    risk_score: float = 0.0
    anomaly_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedAccessController:
    """
    Advanced Access Control System.
    
    Provides sophisticated access control with RBAC, ABAC, dynamic policies,
    and comprehensive security auditing for multi-agent systems.
    """
    
    def __init__(self, isolation_manager: AgentIsolationManager):
        """Initialize the advanced access controller."""
        self.isolation_manager = isolation_manager
        
        # Policy management
        self.policies: Dict[str, AccessPolicy] = {}
        self.global_rules: List[AccessRule] = []
        
        # Role management
        self.roles: Dict[str, Set[str]] = {}  # role_name -> permissions
        self.agent_roles: Dict[str, Set[str]] = {}  # agent_id -> roles
        
        # Attribute management
        self.agent_attributes: Dict[str, Dict[str, Any]] = {}  # agent_id -> attributes
        self.resource_attributes: Dict[str, Dict[str, Any]] = {}  # resource_id -> attributes
        
        # Security monitoring
        self.audit_log: List[AccessAudit] = []
        self.threat_indicators: Dict[str, List[Dict[str, Any]]] = {}  # agent_id -> indicators
        
        # Performance tracking
        self.evaluation_cache: Dict[str, Tuple[PolicyEffect, datetime]] = {}
        self.stats = {
            "total_evaluations": 0,
            "allow_decisions": 0,
            "deny_decisions": 0,
            "cache_hits": 0,
            "avg_evaluation_time": 0.0,
            "threats_detected": 0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info("Advanced access controller initialized")
    
    def _initialize_default_policies(self) -> None:
        """Initialize default security policies."""
        try:
            # Create default admin policy
            admin_policy = AccessPolicy(
                name="Default Admin Policy",
                description="Default administrative access policy",
                security_level=SecurityLevel.RESTRICTED,
                owner="system"
            )
            
            # Admin rule - full access for admin role
            admin_rule = AccessRule(
                name="Admin Full Access",
                description="Administrators have full access to all resources",
                subjects={"role:admin"},
                resources={"*"},
                actions={AccessAction.READ, AccessAction.WRITE, AccessAction.DELETE, AccessAction.ADMIN},
                effect=PolicyEffect.ALLOW,
                priority=1000
            )
            admin_policy.rules.append(admin_rule)
            
            # Create default user policy
            user_policy = AccessPolicy(
                name="Default User Policy",
                description="Default user access policy",
                security_level=SecurityLevel.INTERNAL,
                owner="system"
            )
            
            # User rule - read access to own resources
            user_rule = AccessRule(
                name="User Own Resource Access",
                description="Users can access their own resources",
                subjects={"*"},
                resources={"agent:{subject_id}:*"},
                actions={AccessAction.READ, AccessAction.WRITE},
                effect=PolicyEffect.ALLOW,
                priority=500
            )
            user_policy.rules.append(user_rule)
            
            # Deny rule - deny access to other agents' private resources
            deny_rule = AccessRule(
                name="Deny Private Resource Access",
                description="Deny access to other agents' private resources",
                subjects={"*"},
                resources={"agent:*:private:*"},
                actions={AccessAction.READ, AccessAction.WRITE, AccessAction.DELETE},
                effect=PolicyEffect.DENY,
                priority=800
            )
            user_policy.rules.append(deny_rule)
            
            # Store policies
            self.policies[admin_policy.policy_id] = admin_policy
            self.policies[user_policy.policy_id] = user_policy
            
            # Initialize default roles
            self.roles["admin"] = {
                "full_access", "user_management", "system_configuration",
                "security_management", "audit_access"
            }
            self.roles["user"] = {
                "basic_access", "own_resource_access", "collaboration"
            }
            self.roles["guest"] = {
                "read_only_access"
            }
            
            logger.info("Default security policies initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize default policies: {str(e)}")
    
    async def evaluate_access(
        self,
        subject_id: str,
        resource_id: str,
        action: AccessAction,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[PolicyEffect, List[str]]:
        """
        Evaluate access request against all policies.
        
        Args:
            subject_id: Agent requesting access
            resource_id: Resource being accessed
            action: Action being performed
            context: Additional context for evaluation
            
        Returns:
            Tuple of (decision, applied_rule_ids)
        """
        try:
            start_time = datetime.utcnow()
            context = context or {}
            
            # Check cache first
            cache_key = self._generate_cache_key(subject_id, resource_id, action, context)
            if cache_key in self.evaluation_cache:
                cached_decision, cached_time = self.evaluation_cache[cache_key]
                if (start_time - cached_time).total_seconds() < 300:  # 5 minute cache
                    self.stats["cache_hits"] += 1
                    return cached_decision, []
            
            # Collect all applicable rules
            applicable_rules = []
            
            # Add global rules
            applicable_rules.extend(self.global_rules)
            
            # Add rules from all enabled policies
            for policy in self.policies.values():
                if policy.enabled:
                    applicable_rules.extend(policy.rules)
            
            # Filter and sort rules by priority
            matching_rules = []
            for rule in applicable_rules:
                if await self._rule_matches(rule, subject_id, resource_id, action, context):
                    matching_rules.append(rule)
            
            # Sort by priority (highest first)
            matching_rules.sort(key=lambda r: r.priority, reverse=True)
            
            # Evaluate rules in priority order
            decision = PolicyEffect.DENY  # Default deny
            applied_rules = []
            
            for rule in matching_rules:
                if rule.effect == PolicyEffect.ALLOW:
                    decision = PolicyEffect.ALLOW
                    applied_rules.append(rule.rule_id)
                    break  # First allow wins
                elif rule.effect == PolicyEffect.DENY:
                    decision = PolicyEffect.DENY
                    applied_rules.append(rule.rule_id)
                    break  # Explicit deny wins
                elif rule.effect == PolicyEffect.CONDITIONAL:
                    # Evaluate conditions
                    if await self._evaluate_conditions(rule, subject_id, resource_id, action, context):
                        decision = PolicyEffect.ALLOW
                        applied_rules.append(rule.rule_id)
                        break
            
            # Update rule usage statistics
            for rule in matching_rules:
                if rule.rule_id in applied_rules:
                    rule.usage_count += 1
                    rule.last_used = start_time
            
            # Cache the decision
            self.evaluation_cache[cache_key] = (decision, start_time)
            
            # Calculate evaluation time
            evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create audit log entry
            audit_entry = AccessAudit(
                subject_id=subject_id,
                resource_id=resource_id,
                action=action,
                decision=decision,
                applied_rules=applied_rules,
                evaluation_time_ms=evaluation_time,
                metadata=context
            )
            
            # Detect anomalies
            audit_entry.anomaly_detected = await self._detect_anomalies(subject_id, resource_id, action, context)
            if audit_entry.anomaly_detected:
                self.stats["threats_detected"] += 1
            
            # Store audit entry
            self.audit_log.append(audit_entry)
            
            # Keep only recent audit entries
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-10000:]
            
            # Update statistics
            self.stats["total_evaluations"] += 1
            if decision == PolicyEffect.ALLOW:
                self.stats["allow_decisions"] += 1
            else:
                self.stats["deny_decisions"] += 1
            
            # Update average evaluation time
            self.stats["avg_evaluation_time"] = (
                (self.stats["avg_evaluation_time"] * (self.stats["total_evaluations"] - 1) + evaluation_time) /
                self.stats["total_evaluations"]
            )
            
            return decision, applied_rules
            
        except Exception as e:
            logger.error(f"Failed to evaluate access: {str(e)}")
            return PolicyEffect.DENY, []
    
    async def _rule_matches(
        self,
        rule: AccessRule,
        subject_id: str,
        resource_id: str,
        action: AccessAction,
        context: Dict[str, Any]
    ) -> bool:
        """Check if a rule matches the access request."""
        try:
            # Check if rule is enabled and not expired
            if not rule.enabled:
                return False
            
            if rule.expires_at and rule.expires_at < datetime.utcnow():
                return False
            
            # Check subject match
            if not await self._subject_matches(rule.subjects, subject_id):
                return False
            
            # Check resource match
            if not self._resource_matches(rule.resources, resource_id):
                return False
            
            # Check action match
            if rule.actions and action not in rule.actions:
                return False
            
            # Check time constraints
            if rule.time_constraints and not self._time_constraints_match(rule.time_constraints):
                return False
            
            # Check attribute conditions
            if rule.attribute_conditions and not await self._attribute_conditions_match(
                rule.attribute_conditions, subject_id, resource_id, context
            ):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check rule match: {str(e)}")
            return False
    
    async def _subject_matches(self, rule_subjects: Set[str], subject_id: str) -> bool:
        """Check if subject matches rule subjects."""
        try:
            if not rule_subjects or "*" in rule_subjects:
                return True
            
            # Direct subject match
            if subject_id in rule_subjects:
                return True
            
            # Role-based match
            agent_roles = self.agent_roles.get(subject_id, set())
            for role in agent_roles:
                if f"role:{role}" in rule_subjects:
                    return True
            
            # Attribute-based match
            agent_attrs = self.agent_attributes.get(subject_id, {})
            for subject_pattern in rule_subjects:
                if subject_pattern.startswith("attr:"):
                    attr_condition = subject_pattern[5:]  # Remove "attr:" prefix
                    if self._evaluate_attribute_condition(attr_condition, agent_attrs):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check subject match: {str(e)}")
            return False
    
    def _resource_matches(self, rule_resources: Set[str], resource_id: str) -> bool:
        """Check if resource matches rule resources."""
        try:
            if not rule_resources or "*" in rule_resources:
                return True
            
            for resource_pattern in rule_resources:
                if self._pattern_matches(resource_pattern, resource_id):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check resource match: {str(e)}")
            return False
    
    def _pattern_matches(self, pattern: str, value: str) -> bool:
        """Check if a pattern matches a value (supports wildcards)."""
        try:
            # Simple wildcard matching
            if "*" not in pattern:
                return pattern == value
            
            # Convert pattern to regex-like matching
            pattern_parts = pattern.split("*")
            value_pos = 0
            
            for i, part in enumerate(pattern_parts):
                if not part:  # Empty part from consecutive *
                    continue
                
                if i == 0:  # First part must match from start
                    if not value.startswith(part):
                        return False
                    value_pos = len(part)
                elif i == len(pattern_parts) - 1:  # Last part must match to end
                    if not value.endswith(part):
                        return False
                else:  # Middle parts must exist somewhere
                    pos = value.find(part, value_pos)
                    if pos == -1:
                        return False
                    value_pos = pos + len(part)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to match pattern: {str(e)}")
            return False
    
    def _time_constraints_match(self, time_constraints: Dict[str, Any]) -> bool:
        """Check if current time matches time constraints."""
        try:
            now = datetime.utcnow()
            
            # Check time range
            if "start_time" in time_constraints and "end_time" in time_constraints:
                start_time = datetime.fromisoformat(time_constraints["start_time"])
                end_time = datetime.fromisoformat(time_constraints["end_time"])
                if not (start_time <= now <= end_time):
                    return False
            
            # Check day of week
            if "allowed_days" in time_constraints:
                allowed_days = time_constraints["allowed_days"]
                current_day = now.weekday()  # 0 = Monday, 6 = Sunday
                if current_day not in allowed_days:
                    return False
            
            # Check time of day
            if "allowed_hours" in time_constraints:
                allowed_hours = time_constraints["allowed_hours"]
                current_hour = now.hour
                if current_hour not in allowed_hours:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check time constraints: {str(e)}")
            return False
    
    async def _attribute_conditions_match(
        self,
        attribute_conditions: Dict[str, Any],
        subject_id: str,
        resource_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if attribute conditions are satisfied."""
        try:
            # Get subject attributes
            subject_attrs = self.agent_attributes.get(subject_id, {})
            
            # Get resource attributes
            resource_attrs = self.resource_attributes.get(resource_id, {})
            
            # Combine all attributes
            all_attrs = {
                **subject_attrs,
                **resource_attrs,
                **context,
                "subject_id": subject_id,
                "resource_id": resource_id
            }
            
            # Evaluate each condition
            for condition_key, condition_value in attribute_conditions.items():
                if not self._evaluate_attribute_condition(f"{condition_key}={condition_value}", all_attrs):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check attribute conditions: {str(e)}")
            return False
    
    def _evaluate_attribute_condition(self, condition: str, attributes: Dict[str, Any]) -> bool:
        """Evaluate a single attribute condition."""
        try:
            # Simple condition evaluation (attr=value, attr>value, etc.)
            if "=" in condition:
                attr_name, expected_value = condition.split("=", 1)
                actual_value = attributes.get(attr_name.strip())
                return str(actual_value) == expected_value.strip()
            
            # Add more complex condition evaluation as needed
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate attribute condition: {str(e)}")
            return False
    
    async def _evaluate_conditions(
        self,
        rule: AccessRule,
        subject_id: str,
        resource_id: str,
        action: AccessAction,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate conditional rule conditions."""
        try:
            # Evaluate each condition in the rule
            for condition in rule.conditions:
                if not await self._evaluate_single_condition(condition, subject_id, resource_id, action, context):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate conditions: {str(e)}")
            return False
    
    async def _evaluate_single_condition(
        self,
        condition: str,
        subject_id: str,
        resource_id: str,
        action: AccessAction,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition."""
        try:
            # Simple condition evaluation
            # This could be extended with a proper expression evaluator
            
            if condition.startswith("time_of_day"):
                # Example: time_of_day >= 9 AND time_of_day <= 17
                current_hour = datetime.utcnow().hour
                return 9 <= current_hour <= 17
            
            elif condition.startswith("risk_score"):
                # Example: risk_score < 0.5
                risk_score = context.get("risk_score", 0.0)
                return risk_score < 0.5
            
            # Add more condition types as needed
            return True
            
        except Exception as e:
            logger.error(f"Failed to evaluate single condition: {str(e)}")
            return False
    
    async def _detect_anomalies(
        self,
        subject_id: str,
        resource_id: str,
        action: AccessAction,
        context: Dict[str, Any]
    ) -> bool:
        """Detect anomalous access patterns."""
        try:
            # Simple anomaly detection
            # This could be enhanced with ML-based detection
            
            # Check for unusual access patterns
            recent_accesses = [
                audit for audit in self.audit_log[-100:]
                if audit.subject_id == subject_id and 
                (datetime.utcnow() - audit.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            # Too many access attempts
            if len(recent_accesses) > 100:
                return True
            
            # Too many failed attempts
            failed_attempts = [audit for audit in recent_accesses if audit.decision == PolicyEffect.DENY]
            if len(failed_attempts) > 10:
                return True
            
            # Unusual time access
            current_hour = datetime.utcnow().hour
            if current_hour < 6 or current_hour > 22:  # Outside business hours
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            return False
    
    def _generate_cache_key(
        self,
        subject_id: str,
        resource_id: str,
        action: AccessAction,
        context: Dict[str, Any]
    ) -> str:
        """Generate cache key for access evaluation."""
        try:
            # Create deterministic cache key
            key_data = {
                "subject": subject_id,
                "resource": resource_id,
                "action": action.value,
                "context": sorted(context.items()) if context else []
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate cache key: {str(e)}")
            return f"{subject_id}:{resource_id}:{action.value}"
    
    def assign_role_to_agent(self, agent_id: str, role: str) -> bool:
        """Assign a role to an agent."""
        try:
            if role not in self.roles:
                logger.warning(f"Role {role} does not exist")
                return False
            
            if agent_id not in self.agent_roles:
                self.agent_roles[agent_id] = set()
            
            self.agent_roles[agent_id].add(role)
            logger.info(f"Assigned role {role} to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role: {str(e)}")
            return False
    
    def set_agent_attributes(self, agent_id: str, attributes: Dict[str, Any]) -> None:
        """Set attributes for an agent."""
        try:
            self.agent_attributes[agent_id] = attributes
            logger.info(f"Set attributes for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to set agent attributes: {str(e)}")
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get access control statistics."""
        return {
            **self.stats,
            "total_policies": len(self.policies),
            "total_rules": sum(len(policy.rules) for policy in self.policies.values()),
            "total_roles": len(self.roles),
            "agents_with_roles": len(self.agent_roles),
            "cache_size": len(self.evaluation_cache),
            "audit_log_size": len(self.audit_log)
        }
