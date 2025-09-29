"""
Metadata-Driven Decision Engine

A completely configuration-driven decision engine that uses tool metadata
instead of hardcoded logic for autonomous agent decision making.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

from .decision_engine import DecisionOption, DecisionResult
from app.tools.metadata import ToolMetadataRegistry, get_global_registry

logger = structlog.get_logger(__name__)


@dataclass
class DecisionPattern:
    """Configuration-driven decision pattern."""
    name: str
    description: str
    trigger_conditions: List[str]
    tool_preferences: Dict[str, float]  # tool_name -> preference_weight
    confidence_modifiers: Dict[str, float]  # condition -> modifier
    execution_order: int = 0
    enabled: bool = True


@dataclass
class BehavioralRule:
    """Configuration-driven behavioral rule."""
    name: str
    condition: str
    action: str
    parameters: Dict[str, Any]
    priority: int = 0
    enabled: bool = True


class MetadataDrivenDecisionEngine:
    """Decision engine that uses tool metadata and configuration instead of hardcoded logic."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata_registry: ToolMetadataRegistry = get_global_registry()
        
        # Load decision patterns from config
        self.decision_patterns = self._load_decision_patterns()
        self.behavioral_rules = self._load_behavioral_rules()
        
        # Configuration-driven thresholds
        self.confidence_threshold = config.get("decision_threshold", 0.6)
        self.reasoning_penalty_per_iteration = config.get("reasoning_penalty_per_iteration", 0.15)
        self.tool_boost_for_execution_tasks = config.get("tool_boost_for_execution_tasks", 0.4)
        
        logger.info("Initialized metadata-driven decision engine", 
                   patterns=len(self.decision_patterns),
                   rules=len(self.behavioral_rules))
    
    async def make_decision(self, context: Dict[str, Any]) -> DecisionResult:
        """Make a decision using metadata and configuration."""
        try:
            # Generate decision options using metadata
            options = await self._generate_metadata_driven_options(context)
            
            if not options:
                return self._create_fallback_decision(context)
            
            # Apply behavioral rules
            options = self._apply_behavioral_rules(options, context)
            
            # Select best option
            best_option = self._select_best_option(options, context)
            
            # Create decision result
            return DecisionResult(
                selected_option=best_option,
                all_options=options,
                confidence=best_option.confidence_estimate,
                reasoning=self._generate_decision_reasoning(best_option, options, context),
                expected_outcome=best_option.expected_outcome
            )
            
        except Exception as e:
            logger.error("Metadata-driven decision making failed", error=str(e))
            return self._create_fallback_decision(context)
    
    async def _generate_metadata_driven_options(self, context: Dict[str, Any]) -> List[DecisionOption]:
        """Generate decision options using tool metadata."""
        options = []

        # Get available tools from context
        available_tools = context.get("tools_available", [])
        logger.debug(f"Available tools for decision making: {available_tools}")

        # Generate tool-based options using metadata
        tool_options_created = 0
        for tool_name in available_tools:
            # Try to get metadata with original name first
            tool_metadata = self.metadata_registry.get_tool_metadata(tool_name)

            # If not found and tool name ends with '_tool', try without suffix
            if not tool_metadata and tool_name.endswith('_tool'):
                base_name = tool_name[:-5]  # Remove '_tool' suffix
                tool_metadata = self.metadata_registry.get_tool_metadata(base_name)
                logger.debug(f"Trying base name '{base_name}' for tool '{tool_name}'")

            if tool_metadata:
                option = self._create_tool_option(tool_metadata, context)
                if option:
                    options.append(option)
                    tool_options_created += 1
                    logger.debug(f"Created tool option for {tool_name} with confidence {option.confidence_estimate}")
            else:
                logger.debug(f"No metadata found for tool: {tool_name}")

        logger.debug(f"Created {tool_options_created} tool options out of {len(available_tools)} available tools")

        # Generate reasoning option if configured
        if self._should_include_reasoning_option(context):
            reasoning_option = self._create_reasoning_option(context)
            if reasoning_option:
                options.append(reasoning_option)
                logger.debug(f"Created reasoning option with confidence {reasoning_option.confidence_estimate}")

        # Generate goal-based options
        goal_options = self._create_goal_options(context)
        options.extend(goal_options)
        logger.debug(f"Created {len(goal_options)} goal options")

        logger.debug(f"Total options generated: {len(options)}")
        return options
    
    def _create_tool_option(self, tool_metadata, context: Dict[str, Any]) -> Optional[DecisionOption]:
        """Create a decision option for a tool using its metadata."""
        try:
            # Calculate base confidence from metadata
            base_confidence = tool_metadata.matches_context(context)
            
            # Apply decision patterns
            pattern_confidence = self._apply_decision_patterns(tool_metadata.name, context)
            
            # Apply metadata confidence modifiers
            metadata_confidence = self.metadata_registry.calculate_tool_confidence(tool_metadata.name, context)
            
            # Combine confidences
            final_confidence = (base_confidence + pattern_confidence + metadata_confidence) / 3.0
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            # Generate parameters using metadata
            from app.tools.metadata.parameter_generator import ParameterGenerator
            param_generator = ParameterGenerator()
            suggested_params = param_generator.generate_parameters(tool_metadata, context)
            
            return DecisionOption(
                action=f"use_tool_{tool_metadata.name}",
                type="tool_use",
                parameters={"tool": tool_metadata.name, "args": suggested_params},
                expected_outcome={"tool_result": f"result_from_{tool_metadata.name}"},
                confidence_estimate=final_confidence
            )
            
        except Exception as e:
            logger.warning(f"Failed to create tool option for {tool_metadata.name}", error=str(e))
            return None
    
    def _create_reasoning_option(self, context: Dict[str, Any]) -> Optional[DecisionOption]:
        """Create reasoning option with configuration-driven confidence."""
        try:
            # Get reasoning configuration
            reasoning_config = self.config.get("reasoning_behavior", {})
            
            # Calculate reasoning confidence based on context and iteration count
            base_confidence = reasoning_config.get("base_confidence", 0.7)
            
            # Check reasoning iteration count
            autonomous_reasoning = context.get("autonomous_reasoning", "")
            reasoning_iterations = autonomous_reasoning.lower().count("reasoning") + autonomous_reasoning.lower().count("analyze")
            
            # Apply penalty for excessive reasoning
            confidence_penalty = reasoning_iterations * self.reasoning_penalty_per_iteration
            reasoning_confidence = max(0.1, base_confidence - confidence_penalty)
            
            # Check if task requires execution (from config patterns)
            if self._task_requires_execution(context):
                reasoning_confidence *= reasoning_config.get("execution_task_penalty", 0.3)
            
            return DecisionOption(
                action="autonomous_reasoning",
                type="reasoning",
                parameters={"prompt": "Analyze current situation and determine best course of action"},
                expected_outcome={"reasoning_result": "analysis_and_recommendations"},
                confidence_estimate=reasoning_confidence
            )
            
        except Exception as e:
            logger.warning("Failed to create reasoning option", error=str(e))
            return None
    
    def _create_goal_options(self, context: Dict[str, Any]) -> List[DecisionOption]:
        """Create goal-based options."""
        options = []
        current_goals = context.get("current_goals", [])
        
        for goal in current_goals:
            if isinstance(goal, dict):
                goal_action = goal.get("action", "pursue_goal")
                
                option = DecisionOption(
                    action=goal_action,
                    type="goal_pursuit",
                    parameters={"goal": goal},
                    expected_outcome={"goal_progress": goal.get("expected_progress", 0.1)},
                    confidence_estimate=goal.get("feasibility", 0.5)
                )
                options.append(option)
        
        return options
    
    def _apply_decision_patterns(self, tool_name: str, context: Dict[str, Any]) -> float:
        """Apply decision patterns to calculate confidence modifier."""
        total_modifier = 0.0
        
        for pattern in self.decision_patterns:
            if not pattern.enabled:
                continue
                
            # Check if pattern triggers
            if self._pattern_triggers(pattern, context):
                # Apply tool preference
                tool_preference = pattern.tool_preferences.get(tool_name, 0.0)
                total_modifier += tool_preference
                
                # Apply confidence modifiers
                for condition, modifier in pattern.confidence_modifiers.items():
                    if self._condition_matches(condition, context):
                        total_modifier += modifier
        
        return total_modifier
    
    def _apply_behavioral_rules(self, options: List[DecisionOption], context: Dict[str, Any]) -> List[DecisionOption]:
        """Apply behavioral rules to modify options."""
        modified_options = []
        
        for option in options:
            modified_option = option
            
            for rule in self.behavioral_rules:
                if not rule.enabled:
                    continue
                    
                if self._condition_matches(rule.condition, context):
                    modified_option = self._apply_behavioral_rule(modified_option, rule, context)
            
            modified_options.append(modified_option)
        
        return modified_options
    
    def _select_best_option(self, options: List[DecisionOption], context: Dict[str, Any]) -> DecisionOption:
        """Select the best option based on confidence and configuration."""
        if not options:
            return self._create_fallback_option(context)
        
        # Sort by confidence
        options.sort(key=lambda x: x.confidence_estimate, reverse=True)
        
        # Apply selection strategy from config
        selection_strategy = self.config.get("tool_selection_strategy", "highest_confidence")
        
        if selection_strategy == "highest_confidence":
            return options[0]
        elif selection_strategy == "creative_chaos":
            return self._select_creative_chaos_option(options, context)
        else:
            return options[0]
    
    def _select_creative_chaos_option(self, options: List[DecisionOption], context: Dict[str, Any]) -> DecisionOption:
        """Select option using creative chaos strategy."""
        # Prefer tool usage over reasoning for creative tasks
        tool_options = [opt for opt in options if opt.type == "tool_use"]
        reasoning_options = [opt for opt in options if opt.type == "reasoning"]
        
        # If we have high-confidence tool options, prefer them
        high_confidence_tools = [opt for opt in tool_options if opt.confidence_estimate > 0.7]
        if high_confidence_tools:
            return high_confidence_tools[0]
        
        # Otherwise, return highest confidence option
        return options[0] if options else self._create_fallback_option(context)
    
    def _should_include_reasoning_option(self, context: Dict[str, Any]) -> bool:
        """Check if reasoning option should be included based on config."""
        reasoning_config = self.config.get("reasoning_behavior", {})
        return reasoning_config.get("enabled", True)
    
    def _task_requires_execution(self, context: Dict[str, Any]) -> bool:
        """Check if task requires execution based on configured patterns."""
        execution_patterns = self.config.get("execution_task_patterns", [])
        current_task = context.get("current_task", "").lower()
        
        return any(pattern.lower() in current_task for pattern in execution_patterns)
    
    def _pattern_triggers(self, pattern: DecisionPattern, context: Dict[str, Any]) -> bool:
        """Check if a decision pattern triggers for the given context."""
        for condition in pattern.trigger_conditions:
            if self._condition_matches(condition, context):
                return True
        return False
    
    def _condition_matches(self, condition: str, context: Dict[str, Any]) -> bool:
        """Check if a condition matches the context."""
        try:
            if ":" in condition:
                key, value = condition.split(":", 1)
                context_value = str(context.get(key, "")).lower()
                return value.lower() in context_value
            else:
                # Check if condition exists as a key or in text
                if condition in context:
                    return True
                
                # Check in text fields
                text_fields = ["current_task", "goal", "user_input", "description"]
                for field in text_fields:
                    if field in context and condition.lower() in str(context[field]).lower():
                        return True
                
                return False
                
        except Exception as e:
            logger.warning(f"Condition evaluation failed", condition=condition, error=str(e))
            return False
    
    def _apply_behavioral_rule(self, option: DecisionOption, rule: BehavioralRule, context: Dict[str, Any]) -> DecisionOption:
        """Apply a behavioral rule to modify an option."""
        try:
            if rule.action == "boost_confidence":
                boost_amount = rule.parameters.get("amount", 0.1)
                option.confidence_estimate = min(1.0, option.confidence_estimate + boost_amount)
            elif rule.action == "reduce_confidence":
                reduction_amount = rule.parameters.get("amount", 0.1)
                option.confidence_estimate = max(0.0, option.confidence_estimate - reduction_amount)
            elif rule.action == "modify_parameters":
                new_params = rule.parameters.get("parameters", {})
                option.parameters.update(new_params)
            
            return option
            
        except Exception as e:
            logger.warning(f"Failed to apply behavioral rule {rule.name}", error=str(e))
            return option
    
    def _load_decision_patterns(self) -> List[DecisionPattern]:
        """Load decision patterns from configuration."""
        patterns = []
        pattern_configs = self.config.get("decision_patterns", [])
        
        for pattern_config in pattern_configs:
            try:
                pattern = DecisionPattern(
                    name=pattern_config["name"],
                    description=pattern_config.get("description", ""),
                    trigger_conditions=pattern_config.get("trigger_conditions", []),
                    tool_preferences=pattern_config.get("tool_preferences", {}),
                    confidence_modifiers=pattern_config.get("confidence_modifiers", {}),
                    execution_order=pattern_config.get("execution_order", 0),
                    enabled=pattern_config.get("enabled", True)
                )
                patterns.append(pattern)
            except KeyError as e:
                logger.warning(f"Invalid decision pattern configuration", missing_key=str(e))
        
        return patterns
    
    def _load_behavioral_rules(self) -> List[BehavioralRule]:
        """Load behavioral rules from configuration."""
        rules = []
        rule_configs = self.config.get("behavioral_rules", [])
        
        for rule_config in rule_configs:
            try:
                rule = BehavioralRule(
                    name=rule_config["name"],
                    condition=rule_config["condition"],
                    action=rule_config["action"],
                    parameters=rule_config.get("parameters", {}),
                    priority=rule_config.get("priority", 0),
                    enabled=rule_config.get("enabled", True)
                )
                rules.append(rule)
            except KeyError as e:
                logger.warning(f"Invalid behavioral rule configuration", missing_key=str(e))
        
        return rules
    
    def _create_fallback_decision(self, context: Dict[str, Any]) -> DecisionResult:
        """Create fallback decision when normal decision making fails."""
        fallback_option = self._create_fallback_option(context)
        
        return DecisionResult(
            selected_option=fallback_option,
            all_options=[fallback_option],
            confidence=0.1,
            reasoning=["Decision making failed", "Using fallback reasoning option"],
            expected_outcome={"status": "fallback_reasoning"}
        )
    
    def _create_fallback_option(self, context: Dict[str, Any]) -> DecisionOption:
        """Create fallback option."""
        return DecisionOption(
            action="autonomous_reasoning",
            type="reasoning",
            parameters={"prompt": "Analyze situation and determine next steps"},
            expected_outcome={"reasoning_result": "fallback_analysis"},
            confidence_estimate=0.1
        )
    
    def _generate_decision_reasoning(self, selected_option: DecisionOption, all_options: List[DecisionOption], context: Dict[str, Any]) -> List[str]:
        """Generate reasoning for the decision."""
        reasoning = []
        
        reasoning.append(f"Selected action: {selected_option.action}")
        reasoning.append(f"Confidence: {selected_option.confidence_estimate:.2f}")
        reasoning.append(f"Considered {len(all_options)} options")
        
        if selected_option.type == "tool_use":
            tool_name = selected_option.parameters.get("tool", "unknown")
            reasoning.append(f"Tool selection: {tool_name}")
            reasoning.append("Decision based on tool metadata and context matching")
        elif selected_option.type == "reasoning":
            reasoning.append("Reasoning selected for analysis and planning")
        
        return reasoning
