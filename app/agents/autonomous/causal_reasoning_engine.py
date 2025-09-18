"""
Causal Reasoning Engine for Truly Agentic AI.

This module implements sophisticated causal inference capabilities including:
- Cause-effect relationship understanding
- Intervention planning and execution
- Counterfactual reasoning
- Causal model construction and updating
- Causal discovery from observational data
"""

import asyncio
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict, deque

import structlog
from langchain_core.language_models import BaseLanguageModel

logger = structlog.get_logger(__name__)


class CausalRelationType(str, Enum):
    """Types of causal relationships."""
    DIRECT_CAUSE = "direct_cause"           # X directly causes Y
    INDIRECT_CAUSE = "indirect_cause"       # X causes Y through intermediates
    COMMON_CAUSE = "common_cause"           # X and Y share a common cause
    COMMON_EFFECT = "common_effect"         # X and Y both cause Z
    SPURIOUS = "spurious"                   # Correlation without causation
    BIDIRECTIONAL = "bidirectional"         # X causes Y and Y causes X
    CONDITIONAL = "conditional"             # X causes Y under certain conditions


class InterventionType(str, Enum):
    """Types of causal interventions."""
    DO_INTERVENTION = "do_intervention"     # Set variable to specific value
    SOFT_INTERVENTION = "soft_intervention" # Nudge variable in direction
    STRUCTURAL = "structural"               # Change causal structure
    TEMPORAL = "temporal"                   # Time-based intervention
    CONDITIONAL = "conditional"             # Conditional intervention


class CausalConfidence(str, Enum):
    """Confidence levels for causal relationships."""
    VERY_HIGH = "very_high"    # > 0.9
    HIGH = "high"              # 0.7 - 0.9
    MEDIUM = "medium"          # 0.5 - 0.7
    LOW = "low"                # 0.3 - 0.5
    VERY_LOW = "very_low"      # < 0.3


@dataclass
class CausalVariable:
    """Represents a variable in the causal model."""
    variable_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    variable_type: str = "continuous"  # continuous, discrete, binary, categorical
    domain: Optional[List[Any]] = None  # Possible values for discrete variables
    current_value: Optional[Any] = None
    observed_values: List[Any] = field(default_factory=list)
    is_observable: bool = True
    is_controllable: bool = False  # Can we intervene on this variable?
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelation:
    """Represents a causal relationship between variables."""
    relation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cause_variable: str = ""  # Variable ID
    effect_variable: str = ""  # Variable ID
    relation_type: CausalRelationType = CausalRelationType.DIRECT_CAUSE
    strength: float = 0.0  # Causal strength (-1.0 to 1.0)
    confidence: float = 0.0  # Confidence in this relationship (0.0 to 1.0)
    confidence_level: CausalConfidence = CausalConfidence.MEDIUM
    
    # Conditional information
    conditions: Dict[str, Any] = field(default_factory=dict)
    moderators: List[str] = field(default_factory=list)  # Variables that moderate this relation
    
    # Evidence and discovery
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    discovery_method: str = "observation"  # observation, experiment, reasoning
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    
    # Temporal information
    time_lag: Optional[float] = None  # Time delay between cause and effect
    duration: Optional[float] = None  # How long the effect lasts
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalIntervention:
    """Represents a planned or executed causal intervention."""
    intervention_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_variable: str = ""  # Variable to intervene on
    intervention_type: InterventionType = InterventionType.DO_INTERVENTION
    intervention_value: Any = None
    
    # Planning information
    intended_effects: List[str] = field(default_factory=list)  # Expected effect variables
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)  # Potential unintended effects
    
    # Execution information
    planned_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    actual_outcomes: Dict[str, Any] = field(default_factory=dict)
    success: Optional[bool] = None
    effectiveness: float = 0.0  # How well did it achieve intended effects
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualQuery:
    """Represents a counterfactual reasoning query."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""  # Natural language description
    
    # Factual world (what actually happened)
    factual_evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Counterfactual world (what if scenario)
    counterfactual_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Query variables
    query_variables: List[str] = field(default_factory=list)  # What we want to know
    
    # Results
    counterfactual_outcomes: Dict[str, Any] = field(default_factory=dict)
    probability: float = 0.0  # Probability of counterfactual outcome
    confidence: float = 0.0  # Confidence in the analysis
    
    # Reasoning chain
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalReasoningEngine:
    """
    Advanced causal reasoning engine for autonomous agents.
    
    Implements Pearl's causal hierarchy:
    1. Association (seeing/observing)
    2. Intervention (doing/acting)
    3. Counterfactuals (imagining/reasoning)
    """
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseLanguageModel,
        max_variables: int = 1000,
        max_relations: int = 5000
    ):
        """Initialize the causal reasoning engine."""
        self.agent_id = agent_id
        self.llm = llm
        self.max_variables = max_variables
        self.max_relations = max_relations
        
        # Causal model components
        self.variables: Dict[str, CausalVariable] = {}
        self.relations: Dict[str, CausalRelation] = {}
        self.causal_graph = nx.DiGraph()  # Directed graph for causal structure
        
        # Intervention tracking
        self.interventions: Dict[str, CausalIntervention] = {}
        self.intervention_history: List[str] = []
        
        # Counterfactual reasoning
        self.counterfactual_queries: Dict[str, CounterfactualQuery] = {}
        
        # Learning and discovery
        self.observational_data: List[Dict[str, Any]] = []
        self.experimental_data: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.reasoning_stats = {
            "total_inferences": 0,
            "successful_interventions": 0,
            "counterfactual_queries": 0,
            "causal_discoveries": 0,
            "model_updates": 0
        }
        
        logger.info(
            "Causal reasoning engine initialized",
            agent_id=agent_id,
            max_variables=max_variables,
            max_relations=max_relations
        )
    
    async def add_variable(
        self,
        name: str,
        description: str = "",
        variable_type: str = "continuous",
        domain: Optional[List[Any]] = None,
        is_controllable: bool = False
    ) -> str:
        """Add a new variable to the causal model."""
        try:
            variable = CausalVariable(
                name=name,
                description=description,
                variable_type=variable_type,
                domain=domain,
                is_controllable=is_controllable
            )
            
            self.variables[variable.variable_id] = variable
            self.causal_graph.add_node(variable.variable_id, **{
                "name": name,
                "type": variable_type,
                "controllable": is_controllable
            })
            
            logger.debug(
                "Variable added to causal model",
                variable_id=variable.variable_id,
                name=name,
                type=variable_type
            )
            
            return variable.variable_id
            
        except Exception as e:
            logger.error("Failed to add variable", name=name, error=str(e))
            raise
    
    async def add_causal_relation(
        self,
        cause_name: str,
        effect_name: str,
        relation_type: CausalRelationType = CausalRelationType.DIRECT_CAUSE,
        strength: float = 0.5,
        confidence: float = 0.5,
        evidence: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Add a causal relationship between variables."""
        try:
            # Find variables by name
            cause_var = self._find_variable_by_name(cause_name)
            effect_var = self._find_variable_by_name(effect_name)
            
            if not cause_var or not effect_var:
                raise ValueError(f"Variables not found: {cause_name}, {effect_name}")
            
            # Create causal relation
            relation = CausalRelation(
                cause_variable=cause_var.variable_id,
                effect_variable=effect_var.variable_id,
                relation_type=relation_type,
                strength=strength,
                confidence=confidence,
                evidence=evidence or []
            )
            
            # Determine confidence level
            if confidence >= 0.9:
                relation.confidence_level = CausalConfidence.VERY_HIGH
            elif confidence >= 0.7:
                relation.confidence_level = CausalConfidence.HIGH
            elif confidence >= 0.5:
                relation.confidence_level = CausalConfidence.MEDIUM
            elif confidence >= 0.3:
                relation.confidence_level = CausalConfidence.LOW
            else:
                relation.confidence_level = CausalConfidence.VERY_LOW
            
            self.relations[relation.relation_id] = relation
            
            # Add edge to causal graph
            self.causal_graph.add_edge(
                cause_var.variable_id,
                effect_var.variable_id,
                relation_id=relation.relation_id,
                strength=strength,
                confidence=confidence,
                type=relation_type.value
            )
            
            self.reasoning_stats["causal_discoveries"] += 1
            
            logger.info(
                "Causal relation added",
                relation_id=relation.relation_id,
                cause=cause_name,
                effect=effect_name,
                strength=strength,
                confidence=confidence
            )
            
            return relation.relation_id
            
        except Exception as e:
            logger.error("Failed to add causal relation", cause=cause_name, effect=effect_name, error=str(e))
            raise

    def _find_variable_by_name(self, name: str) -> Optional[CausalVariable]:
        """Find a variable by its name."""
        for variable in self.variables.values():
            if variable.name == name:
                return variable
        return None

    async def observe_data(self, observations: Dict[str, Any]) -> None:
        """Add observational data to the causal model."""
        try:
            # Store observation
            observation = {
                "timestamp": datetime.utcnow().isoformat(),
                "data": observations.copy(),
                "type": "observational"
            }
            self.observational_data.append(observation)

            # Update variable values
            for var_name, value in observations.items():
                variable = self._find_variable_by_name(var_name)
                if variable:
                    variable.current_value = value
                    variable.observed_values.append(value)

                    # Keep only recent observations to manage memory
                    if len(variable.observed_values) > 1000:
                        variable.observed_values = variable.observed_values[-1000:]

            # Trigger causal discovery if we have enough data
            if len(self.observational_data) % 100 == 0:  # Every 100 observations
                await self._discover_causal_relations()

            logger.debug("Observational data added", variables=list(observations.keys()))

        except Exception as e:
            logger.error("Failed to observe data", error=str(e))

    async def plan_intervention(
        self,
        target_variable: str,
        desired_outcome: Dict[str, Any],
        intervention_type: InterventionType = InterventionType.DO_INTERVENTION
    ) -> Optional[CausalIntervention]:
        """Plan a causal intervention to achieve desired outcomes."""
        try:
            # Find target variable
            target_var = self._find_variable_by_name(target_variable)
            if not target_var:
                raise ValueError(f"Target variable '{target_variable}' not found")

            if not target_var.is_controllable:
                raise ValueError(f"Variable '{target_variable}' is not controllable")

            # Analyze causal paths to desired outcomes
            intervention_plan = await self._analyze_intervention_effects(
                target_var.variable_id, desired_outcome
            )

            if not intervention_plan:
                logger.warning("No viable intervention plan found")
                return None

            # Create intervention
            intervention = CausalIntervention(
                target_variable=target_var.variable_id,
                intervention_type=intervention_type,
                intervention_value=intervention_plan["recommended_value"],
                intended_effects=intervention_plan["target_variables"],
                expected_outcomes=intervention_plan["expected_outcomes"],
                side_effects=intervention_plan["potential_side_effects"]
            )

            self.interventions[intervention.intervention_id] = intervention

            logger.info(
                "Intervention planned",
                intervention_id=intervention.intervention_id,
                target=target_variable,
                expected_effects=len(intervention.intended_effects)
            )

            return intervention

        except Exception as e:
            logger.error("Failed to plan intervention", target=target_variable, error=str(e))
            return None

    async def execute_intervention(self, intervention_id: str) -> bool:
        """Execute a planned intervention."""
        try:
            intervention = self.interventions.get(intervention_id)
            if not intervention:
                raise ValueError(f"Intervention {intervention_id} not found")

            if intervention.executed_at:
                logger.warning("Intervention already executed", intervention_id=intervention_id)
                return False

            # Record execution
            intervention.executed_at = datetime.utcnow()

            # Simulate intervention execution (in real system, this would
            # interface with actual control systems)
            success = await self._simulate_intervention_execution(intervention)

            if success:
                intervention.success = True
                self.reasoning_stats["successful_interventions"] += 1

                # Add to intervention history
                self.intervention_history.append(intervention_id)

                logger.info(
                    "Intervention executed successfully",
                    intervention_id=intervention_id,
                    target_variable=intervention.target_variable
                )
            else:
                intervention.success = False
                logger.warning("Intervention execution failed", intervention_id=intervention_id)

            intervention.completed_at = datetime.utcnow()
            return success

        except Exception as e:
            logger.error("Failed to execute intervention", intervention_id=intervention_id, error=str(e))
            return False

    async def reason_counterfactually(
        self,
        query_text: str,
        factual_evidence: Dict[str, Any],
        counterfactual_conditions: Dict[str, Any],
        query_variables: List[str]
    ) -> Optional[CounterfactualQuery]:
        """Perform counterfactual reasoning."""
        try:
            # Create counterfactual query
            query = CounterfactualQuery(
                query_text=query_text,
                factual_evidence=factual_evidence,
                counterfactual_conditions=counterfactual_conditions,
                query_variables=query_variables
            )

            # Perform counterfactual analysis
            analysis_result = await self._analyze_counterfactual(query)

            if analysis_result:
                query.counterfactual_outcomes = analysis_result["outcomes"]
                query.probability = analysis_result["probability"]
                query.confidence = analysis_result["confidence"]
                query.reasoning_steps = analysis_result["reasoning_steps"]

                self.counterfactual_queries[query.query_id] = query
                self.reasoning_stats["counterfactual_queries"] += 1

                logger.info(
                    "Counterfactual reasoning completed",
                    query_id=query.query_id,
                    probability=query.probability,
                    confidence=query.confidence
                )

                return query
            else:
                logger.warning("Counterfactual analysis failed")
                return None

        except Exception as e:
            logger.error("Failed to perform counterfactual reasoning", error=str(e))
            return None

    async def _analyze_intervention_effects(
        self,
        target_variable_id: str,
        desired_outcome: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze the effects of intervening on a target variable."""
        try:
            # Find causal paths from target to outcome variables
            outcome_variables = []
            for outcome_name in desired_outcome.keys():
                outcome_var = self._find_variable_by_name(outcome_name)
                if outcome_var:
                    outcome_variables.append(outcome_var.variable_id)

            if not outcome_variables:
                return None

            # Calculate intervention effects using causal graph
            intervention_effects = {}
            potential_side_effects = []

            for outcome_var_id in outcome_variables:
                # Find causal path
                try:
                    path = nx.shortest_path(self.causal_graph, target_variable_id, outcome_var_id)

                    # Calculate expected effect strength
                    path_strength = 1.0
                    for i in range(len(path) - 1):
                        edge_data = self.causal_graph.get_edge_data(path[i], path[i + 1])
                        if edge_data:
                            path_strength *= edge_data.get("strength", 0.5)

                    outcome_var = self.variables[outcome_var_id]
                    intervention_effects[outcome_var.name] = {
                        "path_strength": path_strength,
                        "expected_change": path_strength * 0.5  # Simplified calculation
                    }

                except nx.NetworkXNoPath:
                    # No direct causal path
                    continue

            # Find potential side effects (other variables affected by target)
            for successor in self.causal_graph.successors(target_variable_id):
                successor_var = self.variables[successor]
                if successor not in outcome_variables:
                    potential_side_effects.append(successor_var.name)

            return {
                "recommended_value": 1.0,  # Simplified - would be calculated based on desired outcome
                "target_variables": list(intervention_effects.keys()),
                "expected_outcomes": intervention_effects,
                "potential_side_effects": potential_side_effects
            }

        except Exception as e:
            logger.error("Failed to analyze intervention effects", error=str(e))
            return None

    async def _simulate_intervention_execution(self, intervention: CausalIntervention) -> bool:
        """Simulate the execution of an intervention."""
        try:
            # In a real system, this would interface with actual control mechanisms
            # For simulation, we'll just record the intervention and update the model

            target_var = self.variables[intervention.target_variable]

            # Update target variable value
            target_var.current_value = intervention.intervention_value

            # Simulate propagation of effects through causal graph
            affected_variables = {}
            for successor in self.causal_graph.successors(intervention.target_variable):
                edge_data = self.causal_graph.get_edge_data(intervention.target_variable, successor)
                if edge_data:
                    strength = edge_data.get("strength", 0.5)
                    # Simplified effect calculation
                    effect = float(intervention.intervention_value) * strength

                    successor_var = self.variables[successor]
                    if successor_var.current_value is not None:
                        new_value = float(successor_var.current_value) + effect
                        successor_var.current_value = new_value
                        affected_variables[successor_var.name] = new_value

            # Record actual outcomes
            intervention.actual_outcomes = affected_variables

            # Calculate effectiveness
            intended_count = len(intervention.intended_effects)
            achieved_count = len([var for var in intervention.intended_effects if var in affected_variables])

            if intended_count > 0:
                intervention.effectiveness = achieved_count / intended_count
            else:
                intervention.effectiveness = 1.0

            return True

        except Exception as e:
            logger.error("Failed to simulate intervention execution", error=str(e))
            return False

    async def _analyze_counterfactual(self, query: CounterfactualQuery) -> Optional[Dict[str, Any]]:
        """Analyze a counterfactual query using the causal model."""
        try:
            reasoning_steps = []

            # Step 1: Identify the factual world
            reasoning_steps.append({
                "step": "factual_world_analysis",
                "description": "Analyzing what actually happened",
                "evidence": query.factual_evidence
            })

            # Step 2: Construct counterfactual world
            reasoning_steps.append({
                "step": "counterfactual_construction",
                "description": "Constructing alternative scenario",
                "conditions": query.counterfactual_conditions
            })

            # Step 3: Trace causal effects in counterfactual world
            counterfactual_outcomes = {}
            total_probability = 1.0

            for var_name in query.query_variables:
                var = self._find_variable_by_name(var_name)
                if not var:
                    continue

                # Find causal influences on this variable
                influences = []
                for pred in self.causal_graph.predecessors(var.variable_id):
                    pred_var = self.variables[pred]
                    edge_data = self.causal_graph.get_edge_data(pred, var.variable_id)

                    if edge_data:
                        influences.append({
                            "variable": pred_var.name,
                            "strength": edge_data.get("strength", 0.5),
                            "confidence": edge_data.get("confidence", 0.5)
                        })

                # Calculate counterfactual outcome
                if var.name in query.factual_evidence:
                    factual_value = query.factual_evidence[var.name]
                else:
                    factual_value = var.current_value

                # Simulate counterfactual value
                counterfactual_value = self._simulate_counterfactual_value(
                    var, influences, query.counterfactual_conditions
                )

                counterfactual_outcomes[var.name] = {
                    "factual_value": factual_value,
                    "counterfactual_value": counterfactual_value,
                    "difference": counterfactual_value - factual_value if isinstance(counterfactual_value, (int, float)) and isinstance(factual_value, (int, float)) else None
                }

                # Update probability based on causal strength
                avg_confidence = np.mean([inf["confidence"] for inf in influences]) if influences else 0.5
                total_probability *= avg_confidence

            reasoning_steps.append({
                "step": "outcome_calculation",
                "description": "Calculating counterfactual outcomes",
                "outcomes": counterfactual_outcomes
            })

            return {
                "outcomes": counterfactual_outcomes,
                "probability": total_probability,
                "confidence": min(total_probability, 0.9),  # Cap confidence
                "reasoning_steps": reasoning_steps
            }

        except Exception as e:
            logger.error("Failed to analyze counterfactual", error=str(e))
            return None

    def _simulate_counterfactual_value(
        self,
        variable: CausalVariable,
        influences: List[Dict[str, Any]],
        counterfactual_conditions: Dict[str, Any]
    ) -> Any:
        """Simulate the value of a variable in a counterfactual world."""
        try:
            # If variable is directly set in counterfactual conditions
            if variable.name in counterfactual_conditions:
                return counterfactual_conditions[variable.name]

            # Calculate based on causal influences
            if not influences:
                return variable.current_value

            # Simplified calculation - in reality would use structural equations
            total_effect = 0.0
            for influence in influences:
                influence_var = self._find_variable_by_name(influence["variable"])
                if influence_var:
                    influence_value = counterfactual_conditions.get(
                        influence["variable"],
                        influence_var.current_value
                    )

                    if isinstance(influence_value, (int, float)):
                        effect = influence_value * influence["strength"]
                        total_effect += effect

            # Return modified value
            if isinstance(variable.current_value, (int, float)):
                return variable.current_value + total_effect * 0.1  # Dampening factor
            else:
                return variable.current_value

        except Exception as e:
            logger.error("Failed to simulate counterfactual value", error=str(e))
            return variable.current_value

    async def _discover_causal_relations(self) -> None:
        """Discover causal relations from observational data."""
        try:
            if len(self.observational_data) < 50:  # Need minimum data
                return

            # Extract variable correlations
            correlations = self._calculate_correlations()

            # Apply causal discovery heuristics
            for (var1, var2), correlation in correlations.items():
                if abs(correlation) > 0.3:  # Significant correlation
                    # Simple heuristic: temporal precedence suggests causation
                    # In reality, would use sophisticated causal discovery algorithms

                    var1_obj = self._find_variable_by_name(var1)
                    var2_obj = self._find_variable_by_name(var2)

                    if var1_obj and var2_obj:
                        # Check if relation already exists
                        existing_relation = self._find_relation(var1_obj.variable_id, var2_obj.variable_id)

                        if not existing_relation:
                            # Create new causal relation
                            await self.add_causal_relation(
                                cause_name=var1,
                                effect_name=var2,
                                relation_type=CausalRelationType.DIRECT_CAUSE,
                                strength=correlation,
                                confidence=min(abs(correlation), 0.8),
                                evidence=[{"type": "correlation", "value": correlation}]
                            )

            self.reasoning_stats["model_updates"] += 1
            logger.info("Causal discovery completed", new_relations=len(correlations))

        except Exception as e:
            logger.error("Failed to discover causal relations", error=str(e))

    def _calculate_correlations(self) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between variables from observational data."""
        try:
            correlations = {}

            # Get all variable names from recent observations
            variable_names = set()
            for obs in self.observational_data[-100:]:  # Last 100 observations
                variable_names.update(obs["data"].keys())

            variable_names = list(variable_names)

            # Calculate pairwise correlations
            for i, var1 in enumerate(variable_names):
                for j, var2 in enumerate(variable_names):
                    if i >= j:  # Avoid duplicates and self-correlation
                        continue

                    # Extract values for both variables
                    var1_values = []
                    var2_values = []

                    for obs in self.observational_data[-100:]:
                        if var1 in obs["data"] and var2 in obs["data"]:
                            val1 = obs["data"][var1]
                            val2 = obs["data"][var2]

                            # Convert to numeric if possible
                            try:
                                val1 = float(val1)
                                val2 = float(val2)
                                var1_values.append(val1)
                                var2_values.append(val2)
                            except (ValueError, TypeError):
                                continue

                    # Calculate correlation if we have enough data
                    if len(var1_values) >= 10:
                        correlation = np.corrcoef(var1_values, var2_values)[0, 1]
                        if not np.isnan(correlation):
                            correlations[(var1, var2)] = correlation

            return correlations

        except Exception as e:
            logger.error("Failed to calculate correlations", error=str(e))
            return {}

    def _find_relation(self, cause_id: str, effect_id: str) -> Optional[CausalRelation]:
        """Find an existing causal relation between two variables."""
        for relation in self.relations.values():
            if relation.cause_variable == cause_id and relation.effect_variable == effect_id:
                return relation
        return None

    def get_causal_explanation(self, effect_variable: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get causal explanation for why a variable has its current value."""
        try:
            effect_var = self._find_variable_by_name(effect_variable)
            if not effect_var:
                return {"error": f"Variable '{effect_variable}' not found"}

            # Find all causal influences
            explanations = []
            total_explained_variance = 0.0

            for pred in self.causal_graph.predecessors(effect_var.variable_id):
                pred_var = self.variables[pred]
                edge_data = self.causal_graph.get_edge_data(pred, effect_var.variable_id)

                if edge_data:
                    relation_id = edge_data.get("relation_id")
                    relation = self.relations.get(relation_id) if relation_id else None

                    explanation = {
                        "cause_variable": pred_var.name,
                        "cause_value": pred_var.current_value,
                        "causal_strength": edge_data.get("strength", 0.0),
                        "confidence": edge_data.get("confidence", 0.0),
                        "relation_type": relation.relation_type.value if relation else "unknown"
                    }

                    explanations.append(explanation)
                    total_explained_variance += edge_data.get("strength", 0.0) ** 2

            return {
                "effect_variable": effect_variable,
                "current_value": effect_var.current_value,
                "causal_explanations": explanations,
                "explained_variance": min(total_explained_variance, 1.0),
                "unexplained_variance": max(0.0, 1.0 - total_explained_variance)
            }

        except Exception as e:
            logger.error("Failed to get causal explanation", variable=effect_variable, error=str(e))
            return {"error": str(e)}

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about causal reasoning performance."""
        return {
            "total_variables": len(self.variables),
            "total_relations": len(self.relations),
            "total_interventions": len(self.interventions),
            "total_counterfactual_queries": len(self.counterfactual_queries),
            "observational_data_points": len(self.observational_data),
            "experimental_data_points": len(self.experimental_data),
            "reasoning_stats": self.reasoning_stats.copy(),
            "graph_complexity": {
                "nodes": self.causal_graph.number_of_nodes(),
                "edges": self.causal_graph.number_of_edges(),
                "density": nx.density(self.causal_graph) if self.causal_graph.number_of_nodes() > 0 else 0.0
            }
        }
