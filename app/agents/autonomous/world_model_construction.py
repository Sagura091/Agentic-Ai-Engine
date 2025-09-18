"""
World Model Construction for Truly Agentic AI.

This module implements dynamic internal world models for prediction and simulation, including:
- Environment modeling and state representation
- Action outcome prediction and simulation
- Temporal dynamics and state transitions
- Continuous model updating based on experience
- Multi-scale world modeling (local to global)
"""

import asyncio
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict, deque

import structlog
from langchain_core.language_models import BaseLanguageModel

logger = structlog.get_logger(__name__)


class WorldModelType(str, Enum):
    """Types of world models."""
    PHYSICAL = "physical"           # Physical environment and objects
    SOCIAL = "social"              # Social dynamics and relationships
    TEMPORAL = "temporal"          # Time-based patterns and sequences
    CAUSAL = "causal"              # Cause-effect relationships
    SPATIAL = "spatial"            # Spatial relationships and navigation
    ABSTRACT = "abstract"          # Abstract concepts and rules
    HYBRID = "hybrid"              # Combination of multiple types


class ModelScope(str, Enum):
    """Scope of world model coverage."""
    LOCAL = "local"                # Immediate environment
    REGIONAL = "regional"          # Extended local area
    GLOBAL = "global"              # Entire known world
    UNIVERSAL = "universal"        # Abstract universal principles


class StateType(str, Enum):
    """Types of world states."""
    OBSERVABLE = "observable"      # Directly observable states
    HIDDEN = "hidden"              # Inferred hidden states
    PREDICTED = "predicted"        # Predicted future states
    COUNTERFACTUAL = "counterfactual"  # Alternative possible states


class UpdateMechanism(str, Enum):
    """Mechanisms for updating world models."""
    OBSERVATION = "observation"    # Direct observation
    INFERENCE = "inference"        # Logical inference
    PREDICTION = "prediction"      # Predictive updates
    CORRECTION = "correction"      # Error correction
    LEARNING = "learning"          # Experience-based learning


@dataclass
class WorldState:
    """Represents a state in the world model."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    state_type: StateType = StateType.OBSERVABLE
    
    # State content
    entities: Dict[str, Any] = field(default_factory=dict)  # Objects, agents, etc.
    properties: Dict[str, Any] = field(default_factory=dict)  # Environmental properties
    relationships: Dict[str, Any] = field(default_factory=dict)  # Entity relationships
    
    # Spatial information
    spatial_layout: Dict[str, Any] = field(default_factory=dict)
    coordinates: Optional[Tuple[float, ...]] = None
    
    # Temporal information
    duration: Optional[float] = None  # How long this state lasts
    stability: float = 1.0  # How stable/persistent this state is
    
    # Uncertainty and confidence
    confidence: float = 1.0  # Confidence in this state representation
    uncertainty: Dict[str, float] = field(default_factory=dict)  # Per-property uncertainty
    
    # Metadata
    source: str = "observation"  # How this state was obtained
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransition:
    """Represents a transition between world states."""
    transition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_state: str = ""  # Source state ID
    to_state: str = ""    # Target state ID
    
    # Transition mechanism
    action: Optional[str] = None  # Action that caused transition
    trigger: Optional[str] = None  # Event that triggered transition
    conditions: Dict[str, Any] = field(default_factory=dict)  # Required conditions
    
    # Transition properties
    probability: float = 1.0  # Probability of this transition
    duration: float = 0.0     # Time taken for transition
    cost: float = 0.0         # Cost/effort of transition
    
    # Learning information
    observed_count: int = 0   # How many times observed
    success_rate: float = 1.0 # Success rate of this transition
    
    # Metadata
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_observed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldEntity:
    """Represents an entity in the world model."""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: str = "object"  # object, agent, location, concept, etc.
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    # Spatial information
    position: Optional[Tuple[float, ...]] = None
    size: Optional[Tuple[float, ...]] = None
    orientation: Optional[float] = None
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    lifespan: Optional[float] = None  # Expected lifespan
    
    # Behavioral patterns (for agents)
    behavior_patterns: List[Dict[str, Any]] = field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Uncertainty
    confidence: float = 1.0
    uncertainty: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of a world model prediction."""
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    predicted_state: WorldState
    prediction_horizon: float  # How far into the future
    confidence: float = 0.0
    
    # Prediction details
    prediction_method: str = "simulation"
    assumptions: List[str] = field(default_factory=list)
    uncertainty_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Validation
    actual_outcome: Optional[WorldState] = None
    accuracy: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorldModelConstructor:
    """
    Advanced world model construction system for autonomous agents.
    
    Builds and maintains dynamic internal models of the world that enable
    prediction, simulation, and planning capabilities essential for truly
    agentic behavior.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseLanguageModel,
        model_types: List[WorldModelType] = None,
        max_states: int = 10000,
        max_entities: int = 5000
    ):
        """Initialize the world model constructor."""
        self.agent_id = agent_id
        self.llm = llm
        self.model_types = model_types or [WorldModelType.PHYSICAL, WorldModelType.TEMPORAL]
        self.max_states = max_states
        self.max_entities = max_entities
        
        # World model components
        self.world_states: Dict[str, WorldState] = {}
        self.state_transitions: Dict[str, StateTransition] = {}
        self.world_entities: Dict[str, WorldEntity] = {}
        
        # Model structure
        self.state_graph = nx.DiGraph()  # Graph of state transitions
        self.entity_graph = nx.Graph()   # Graph of entity relationships
        
        # Current state tracking
        self.current_state_id: Optional[str] = None
        self.state_history: deque = deque(maxlen=1000)
        
        # Prediction and simulation
        self.predictions: Dict[str, PredictionResult] = {}
        self.simulation_cache: Dict[str, Any] = {}
        
        # Learning and adaptation
        self.model_updates: List[Dict[str, Any]] = []
        self.prediction_accuracy_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.model_stats = {
            "total_states": 0,
            "total_transitions": 0,
            "total_entities": 0,
            "predictions_made": 0,
            "prediction_accuracy": 0.0,
            "model_updates": 0,
            "simulation_runs": 0
        }
        
        logger.info(
            "World model constructor initialized",
            agent_id=agent_id,
            model_types=[t.value for t in self.model_types],
            max_states=max_states
        )
    
    async def observe_world_state(
        self,
        observations: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> str:
        """Observe and record a new world state."""
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Create new world state
            state = WorldState(
                timestamp=timestamp,
                state_type=StateType.OBSERVABLE,
                source="direct_observation"
            )
            
            # Process observations into structured format
            await self._process_observations(state, observations)
            
            # Store the state
            self.world_states[state.state_id] = state
            self.state_graph.add_node(state.state_id, timestamp=timestamp)
            
            # Update current state
            previous_state_id = self.current_state_id
            self.current_state_id = state.state_id
            self.state_history.append(state.state_id)
            
            # Create transition if we have a previous state
            if previous_state_id:
                await self._create_state_transition(previous_state_id, state.state_id)
            
            # Update entities based on observations
            await self._update_entities_from_state(state)
            
            # Trigger model updates
            await self._update_world_model(state)
            
            self.model_stats["total_states"] += 1
            
            logger.debug(
                "World state observed",
                state_id=state.state_id,
                entities_count=len(state.entities),
                properties_count=len(state.properties)
            )
            
            return state.state_id
            
        except Exception as e:
            logger.error("Failed to observe world state", error=str(e))
            raise
    
    async def predict_future_state(
        self,
        horizon: float,
        conditions: Optional[Dict[str, Any]] = None,
        actions: Optional[List[str]] = None
    ) -> Optional[PredictionResult]:
        """Predict future world state based on current model."""
        try:
            if not self.current_state_id:
                logger.warning("No current state available for prediction")
                return None
            
            current_state = self.world_states[self.current_state_id]
            
            # Run simulation to predict future state
            predicted_state = await self._simulate_future_state(
                current_state, horizon, conditions, actions
            )
            
            if not predicted_state:
                return None
            
            # Calculate prediction confidence
            confidence = await self._calculate_prediction_confidence(
                current_state, predicted_state, horizon
            )
            
            # Create prediction result
            prediction = PredictionResult(
                predicted_state=predicted_state,
                prediction_horizon=horizon,
                confidence=confidence,
                assumptions=conditions.get("assumptions", []) if conditions else []
            )
            
            self.predictions[prediction.prediction_id] = prediction
            self.model_stats["predictions_made"] += 1
            
            logger.info(
                "Future state predicted",
                prediction_id=prediction.prediction_id,
                horizon=horizon,
                confidence=confidence
            )
            
            return prediction
            
        except Exception as e:
            logger.error("Failed to predict future state", horizon=horizon, error=str(e))
            return None

    async def simulate_action_outcome(
        self,
        action: str,
        action_parameters: Dict[str, Any] = None,
        target_entities: List[str] = None
    ) -> Optional[WorldState]:
        """Simulate the outcome of taking a specific action."""
        try:
            if not self.current_state_id:
                logger.warning("No current state available for action simulation")
                return None

            current_state = self.world_states[self.current_state_id]
            action_parameters = action_parameters or {}
            target_entities = target_entities or []

            # Create simulated state
            simulated_state = WorldState(
                timestamp=datetime.utcnow(),
                state_type=StateType.PREDICTED,
                source="action_simulation"
            )

            # Copy current state as baseline
            simulated_state.entities = current_state.entities.copy()
            simulated_state.properties = current_state.properties.copy()
            simulated_state.relationships = current_state.relationships.copy()
            simulated_state.spatial_layout = current_state.spatial_layout.copy()

            # Apply action effects
            await self._apply_action_effects(
                simulated_state, action, action_parameters, target_entities
            )

            # Update confidence based on action familiarity
            action_confidence = await self._calculate_action_confidence(action, action_parameters)
            simulated_state.confidence = action_confidence

            self.model_stats["simulation_runs"] += 1

            logger.debug(
                "Action outcome simulated",
                action=action,
                confidence=action_confidence,
                affected_entities=len(target_entities)
            )

            return simulated_state

        except Exception as e:
            logger.error("Failed to simulate action outcome", action=action, error=str(e))
            return None

    async def update_model_from_experience(
        self,
        experience: Dict[str, Any],
        update_mechanism: UpdateMechanism = UpdateMechanism.LEARNING
    ) -> bool:
        """Update world model based on new experience."""
        try:
            update_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "mechanism": update_mechanism.value,
                "experience": experience,
                "changes_made": []
            }

            # Extract relevant information from experience
            if "state_before" in experience and "state_after" in experience:
                # Learn state transition
                await self._learn_state_transition(
                    experience["state_before"],
                    experience["state_after"],
                    experience.get("action"),
                    experience.get("conditions", {})
                )
                update_record["changes_made"].append("state_transition_learned")

            if "entity_changes" in experience:
                # Update entity models
                await self._update_entity_models(experience["entity_changes"])
                update_record["changes_made"].append("entity_models_updated")

            if "prediction_validation" in experience:
                # Validate and improve predictions
                await self._validate_prediction(experience["prediction_validation"])
                update_record["changes_made"].append("prediction_validated")

            if "causal_relationships" in experience:
                # Update causal understanding
                await self._update_causal_relationships(experience["causal_relationships"])
                update_record["changes_made"].append("causal_relationships_updated")

            # Store update record
            self.model_updates.append(update_record)
            self.model_stats["model_updates"] += 1

            # Trigger model optimization if needed
            if len(self.model_updates) % 50 == 0:
                await self._optimize_world_model()

            logger.info(
                "World model updated from experience",
                mechanism=update_mechanism.value,
                changes=len(update_record["changes_made"])
            )

            return True

        except Exception as e:
            logger.error("Failed to update model from experience", error=str(e))
            return False

    async def _process_observations(self, state: WorldState, observations: Dict[str, Any]) -> None:
        """Process raw observations into structured world state."""
        try:
            # Extract entities
            if "entities" in observations:
                for entity_data in observations["entities"]:
                    entity_id = entity_data.get("id", str(uuid.uuid4()))
                    state.entities[entity_id] = {
                        "type": entity_data.get("type", "unknown"),
                        "properties": entity_data.get("properties", {}),
                        "position": entity_data.get("position"),
                        "state": entity_data.get("state", "active")
                    }

            # Extract environmental properties
            if "environment" in observations:
                state.properties.update(observations["environment"])

            # Extract relationships
            if "relationships" in observations:
                state.relationships.update(observations["relationships"])

            # Extract spatial layout
            if "spatial" in observations:
                state.spatial_layout.update(observations["spatial"])

            # Set coordinates if provided
            if "coordinates" in observations:
                state.coordinates = tuple(observations["coordinates"])

            # Calculate overall confidence
            confidence_factors = []
            if "confidence" in observations:
                confidence_factors.append(observations["confidence"])
            if "sensor_quality" in observations:
                confidence_factors.append(observations["sensor_quality"])

            if confidence_factors:
                state.confidence = np.mean(confidence_factors)

        except Exception as e:
            logger.error("Failed to process observations", error=str(e))

    async def _create_state_transition(self, from_state_id: str, to_state_id: str) -> None:
        """Create a state transition between two states."""
        try:
            from_state = self.world_states[from_state_id]
            to_state = self.world_states[to_state_id]

            # Calculate transition duration
            duration = (to_state.timestamp - from_state.timestamp).total_seconds()

            # Create transition
            transition = StateTransition(
                from_state=from_state_id,
                to_state=to_state_id,
                duration=duration,
                observed_count=1
            )

            # Analyze what changed
            changes = await self._analyze_state_changes(from_state, to_state)
            transition.metadata["changes"] = changes

            # Store transition
            self.state_transitions[transition.transition_id] = transition

            # Add edge to state graph
            self.state_graph.add_edge(
                from_state_id,
                to_state_id,
                transition_id=transition.transition_id,
                duration=duration,
                changes=len(changes)
            )

            self.model_stats["total_transitions"] += 1

        except Exception as e:
            logger.error("Failed to create state transition", error=str(e))

    async def _update_entities_from_state(self, state: WorldState) -> None:
        """Update entity models based on observed state."""
        try:
            for entity_id, entity_data in state.entities.items():
                if entity_id in self.world_entities:
                    # Update existing entity
                    entity = self.world_entities[entity_id]
                    entity.properties.update(entity_data.get("properties", {}))
                    entity.last_updated = state.timestamp

                    # Update position if provided
                    if "position" in entity_data:
                        entity.position = tuple(entity_data["position"])
                else:
                    # Create new entity
                    entity = WorldEntity(
                        entity_id=entity_id,
                        name=entity_data.get("name", f"entity_{entity_id[:8]}"),
                        entity_type=entity_data.get("type", "unknown"),
                        properties=entity_data.get("properties", {}),
                        position=tuple(entity_data["position"]) if "position" in entity_data else None,
                        created_at=state.timestamp
                    )

                    self.world_entities[entity_id] = entity
                    self.entity_graph.add_node(entity_id, **{
                        "name": entity.name,
                        "type": entity.entity_type
                    })

                    self.model_stats["total_entities"] += 1

            # Update entity relationships
            for rel_id, rel_data in state.relationships.items():
                if "entities" in rel_data and len(rel_data["entities"]) >= 2:
                    entity_ids = rel_data["entities"]
                    for i in range(len(entity_ids)):
                        for j in range(i + 1, len(entity_ids)):
                            if entity_ids[i] in self.world_entities and entity_ids[j] in self.world_entities:
                                self.entity_graph.add_edge(
                                    entity_ids[i],
                                    entity_ids[j],
                                    relationship=rel_data.get("type", "related"),
                                    strength=rel_data.get("strength", 1.0)
                                )

        except Exception as e:
            logger.error("Failed to update entities from state", error=str(e))

    async def _update_world_model(self, state: WorldState) -> None:
        """Update the overall world model based on new state."""
        try:
            # Update temporal patterns
            await self._update_temporal_patterns(state)

            # Update spatial understanding
            await self._update_spatial_model(state)

            # Update entity behavior patterns
            await self._update_behavior_patterns(state)

            # Prune old data if necessary
            if len(self.world_states) > self.max_states:
                await self._prune_old_states()

            if len(self.world_entities) > self.max_entities:
                await self._prune_old_entities()

        except Exception as e:
            logger.error("Failed to update world model", error=str(e))

    async def _simulate_future_state(
        self,
        current_state: WorldState,
        horizon: float,
        conditions: Optional[Dict[str, Any]],
        actions: Optional[List[str]]
    ) -> Optional[WorldState]:
        """Simulate future state using world model."""
        try:
            # Create future state as copy of current
            future_state = WorldState(
                timestamp=current_state.timestamp + timedelta(seconds=horizon),
                state_type=StateType.PREDICTED,
                source="simulation"
            )

            # Copy current state
            future_state.entities = current_state.entities.copy()
            future_state.properties = current_state.properties.copy()
            future_state.relationships = current_state.relationships.copy()
            future_state.spatial_layout = current_state.spatial_layout.copy()

            # Apply temporal evolution
            await self._apply_temporal_evolution(future_state, horizon)

            # Apply planned actions if provided
            if actions:
                for action in actions:
                    await self._apply_action_effects(future_state, action, {}, [])

            # Apply external conditions
            if conditions:
                await self._apply_external_conditions(future_state, conditions)

            # Calculate uncertainty
            future_state.confidence = max(0.1, current_state.confidence * (1.0 - horizon / 3600.0))

            return future_state

        except Exception as e:
            logger.error("Failed to simulate future state", error=str(e))
            return None

    async def _apply_action_effects(
        self,
        state: WorldState,
        action: str,
        parameters: Dict[str, Any],
        target_entities: List[str]
    ) -> None:
        """Apply the effects of an action to a world state."""
        try:
            # Simple action effect simulation - in reality would be much more sophisticated
            if action == "move":
                # Update agent position
                if "agent_id" in parameters and "destination" in parameters:
                    agent_id = parameters["agent_id"]
                    if agent_id in state.entities:
                        state.entities[agent_id]["position"] = parameters["destination"]

            elif action == "interact":
                # Update entity states based on interaction
                for entity_id in target_entities:
                    if entity_id in state.entities:
                        state.entities[entity_id]["state"] = "interacted"
                        state.entities[entity_id]["last_interaction"] = datetime.utcnow().isoformat()

            elif action == "modify":
                # Modify entity properties
                if "entity_id" in parameters and "property" in parameters:
                    entity_id = parameters["entity_id"]
                    if entity_id in state.entities:
                        prop_name = parameters["property"]
                        prop_value = parameters.get("value", "modified")
                        state.entities[entity_id]["properties"][prop_name] = prop_value

            # Add action to state metadata
            if "actions_taken" not in state.metadata:
                state.metadata["actions_taken"] = []

            state.metadata["actions_taken"].append({
                "action": action,
                "parameters": parameters,
                "targets": target_entities,
                "timestamp": datetime.utcnow().isoformat()
            })

        except Exception as e:
            logger.error("Failed to apply action effects", action=action, error=str(e))

    async def _calculate_action_confidence(self, action: str, parameters: Dict[str, Any]) -> float:
        """Calculate confidence in action outcome prediction."""
        try:
            # Base confidence
            confidence = 0.5

            # Increase confidence for familiar actions
            action_history = [
                update for update in self.model_updates
                if "action" in update.get("experience", {}) and
                update["experience"]["action"] == action
            ]

            familiarity_bonus = min(0.4, len(action_history) * 0.05)
            confidence += familiarity_bonus

            # Decrease confidence for complex parameters
            complexity_penalty = min(0.3, len(parameters) * 0.05)
            confidence -= complexity_penalty

            return max(0.1, min(1.0, confidence))

        except Exception as e:
            logger.error("Failed to calculate action confidence", error=str(e))
            return 0.5

    async def _calculate_prediction_confidence(
        self,
        current_state: WorldState,
        predicted_state: WorldState,
        horizon: float
    ) -> float:
        """Calculate confidence in a prediction."""
        try:
            # Base confidence decreases with time horizon
            base_confidence = max(0.1, 1.0 - (horizon / 3600.0))  # Decreases over 1 hour

            # Adjust based on model accuracy history
            if self.prediction_accuracy_history:
                avg_accuracy = np.mean(list(self.prediction_accuracy_history))
                accuracy_factor = avg_accuracy
            else:
                accuracy_factor = 0.5

            # Adjust based on state complexity
            state_complexity = len(predicted_state.entities) + len(predicted_state.properties)
            complexity_factor = max(0.5, 1.0 - (state_complexity / 100.0))

            # Combine factors
            confidence = base_confidence * accuracy_factor * complexity_factor

            return max(0.1, min(1.0, confidence))

        except Exception as e:
            logger.error("Failed to calculate prediction confidence", error=str(e))
            return 0.5

    async def _analyze_state_changes(self, from_state: WorldState, to_state: WorldState) -> List[Dict[str, Any]]:
        """Analyze changes between two states."""
        try:
            changes = []

            # Entity changes
            for entity_id in set(from_state.entities.keys()) | set(to_state.entities.keys()):
                if entity_id not in from_state.entities:
                    changes.append({
                        "type": "entity_added",
                        "entity_id": entity_id,
                        "entity_data": to_state.entities[entity_id]
                    })
                elif entity_id not in to_state.entities:
                    changes.append({
                        "type": "entity_removed",
                        "entity_id": entity_id,
                        "entity_data": from_state.entities[entity_id]
                    })
                else:
                    # Check for property changes
                    from_props = from_state.entities[entity_id].get("properties", {})
                    to_props = to_state.entities[entity_id].get("properties", {})

                    for prop in set(from_props.keys()) | set(to_props.keys()):
                        if from_props.get(prop) != to_props.get(prop):
                            changes.append({
                                "type": "entity_property_changed",
                                "entity_id": entity_id,
                                "property": prop,
                                "from_value": from_props.get(prop),
                                "to_value": to_props.get(prop)
                            })

            # Property changes
            for prop in set(from_state.properties.keys()) | set(to_state.properties.keys()):
                if from_state.properties.get(prop) != to_state.properties.get(prop):
                    changes.append({
                        "type": "environment_property_changed",
                        "property": prop,
                        "from_value": from_state.properties.get(prop),
                        "to_value": to_state.properties.get(prop)
                    })

            return changes

        except Exception as e:
            logger.error("Failed to analyze state changes", error=str(e))
            return []

    async def _apply_temporal_evolution(self, state: WorldState, time_delta: float) -> None:
        """Apply temporal evolution to a state."""
        try:
            # Simple temporal evolution - entities age, properties change over time
            for entity_id, entity_data in state.entities.items():
                # Age entities
                if "age" in entity_data.get("properties", {}):
                    current_age = entity_data["properties"]["age"]
                    entity_data["properties"]["age"] = current_age + time_delta

                # Apply decay to temporary properties
                if "temporary_effects" in entity_data.get("properties", {}):
                    effects = entity_data["properties"]["temporary_effects"]
                    for effect_name, effect_data in list(effects.items()):
                        remaining_time = effect_data.get("duration", 0) - time_delta
                        if remaining_time <= 0:
                            del effects[effect_name]
                        else:
                            effect_data["duration"] = remaining_time

            # Apply environmental changes over time
            if "weather" in state.properties:
                # Simple weather evolution
                current_temp = state.properties.get("temperature", 20.0)
                # Add some random variation
                temp_change = np.random.normal(0, 0.1) * time_delta / 3600.0
                state.properties["temperature"] = current_temp + temp_change

        except Exception as e:
            logger.error("Failed to apply temporal evolution", error=str(e))

    async def _apply_external_conditions(self, state: WorldState, conditions: Dict[str, Any]) -> None:
        """Apply external conditions to a state."""
        try:
            # Apply environmental conditions
            if "environment" in conditions:
                state.properties.update(conditions["environment"])

            # Apply entity modifications
            if "entity_modifications" in conditions:
                for entity_id, modifications in conditions["entity_modifications"].items():
                    if entity_id in state.entities:
                        state.entities[entity_id].update(modifications)

            # Apply global effects
            if "global_effects" in conditions:
                for effect_name, effect_value in conditions["global_effects"].items():
                    state.metadata[f"global_{effect_name}"] = effect_value

        except Exception as e:
            logger.error("Failed to apply external conditions", error=str(e))

    async def _update_temporal_patterns(self, state: WorldState) -> None:
        """Update understanding of temporal patterns."""
        try:
            # Analyze patterns in state history
            if len(self.state_history) >= 10:
                # Look for cyclical patterns
                recent_states = list(self.state_history)[-10:]

                # Simple pattern detection - in reality would be much more sophisticated
                pattern_data = {
                    "recent_state_count": len(recent_states),
                    "average_transition_time": 0.0,
                    "detected_cycles": []
                }

                # Calculate average transition time
                if len(recent_states) >= 2:
                    transition_times = []
                    for i in range(1, len(recent_states)):
                        prev_state = self.world_states[recent_states[i-1]]
                        curr_state = self.world_states[recent_states[i]]
                        time_diff = (curr_state.timestamp - prev_state.timestamp).total_seconds()
                        transition_times.append(time_diff)

                    if transition_times:
                        pattern_data["average_transition_time"] = np.mean(transition_times)

                # Store pattern information
                state.metadata["temporal_patterns"] = pattern_data

        except Exception as e:
            logger.error("Failed to update temporal patterns", error=str(e))

    async def _update_spatial_model(self, state: WorldState) -> None:
        """Update spatial understanding of the world."""
        try:
            # Update spatial relationships between entities
            spatial_relationships = {}

            for entity_id, entity_data in state.entities.items():
                if "position" in entity_data:
                    pos = entity_data["position"]

                    # Find nearby entities
                    nearby_entities = []
                    for other_id, other_data in state.entities.items():
                        if other_id != entity_id and "position" in other_data:
                            other_pos = other_data["position"]

                            # Calculate distance (simplified 2D)
                            if len(pos) >= 2 and len(other_pos) >= 2:
                                distance = np.sqrt((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)
                                if distance < 10.0:  # Within 10 units
                                    nearby_entities.append({
                                        "entity_id": other_id,
                                        "distance": distance
                                    })

                    spatial_relationships[entity_id] = nearby_entities

            state.metadata["spatial_relationships"] = spatial_relationships

        except Exception as e:
            logger.error("Failed to update spatial model", error=str(e))

    async def _update_behavior_patterns(self, state: WorldState) -> None:
        """Update understanding of entity behavior patterns."""
        try:
            # Analyze behavior patterns for agent entities
            for entity_id, entity_data in state.entities.items():
                if entity_data.get("type") == "agent":
                    entity = self.world_entities.get(entity_id)
                    if entity:
                        # Record current behavior
                        behavior_record = {
                            "timestamp": state.timestamp.isoformat(),
                            "position": entity_data.get("position"),
                            "state": entity_data.get("state"),
                            "properties": entity_data.get("properties", {})
                        }

                        entity.interaction_history.append(behavior_record)

                        # Keep only recent history
                        if len(entity.interaction_history) > 100:
                            entity.interaction_history = entity.interaction_history[-100:]

                        # Analyze patterns (simplified)
                        if len(entity.interaction_history) >= 5:
                            recent_positions = [
                                record.get("position") for record in entity.interaction_history[-5:]
                                if record.get("position")
                            ]

                            if len(recent_positions) >= 3:
                                # Detect movement patterns
                                movement_pattern = {
                                    "type": "movement",
                                    "positions": recent_positions,
                                    "detected_at": datetime.utcnow().isoformat()
                                }

                                entity.behavior_patterns.append(movement_pattern)

                                # Keep only recent patterns
                                if len(entity.behavior_patterns) > 20:
                                    entity.behavior_patterns = entity.behavior_patterns[-20:]

        except Exception as e:
            logger.error("Failed to update behavior patterns", error=str(e))

    def get_world_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world model."""
        try:
            return {
                "model_stats": self.model_stats.copy(),
                "current_state_id": self.current_state_id,
                "total_states": len(self.world_states),
                "total_entities": len(self.world_entities),
                "total_transitions": len(self.state_transitions),
                "model_types": [t.value for t in self.model_types],
                "state_history_length": len(self.state_history),
                "predictions_made": len(self.predictions),
                "recent_accuracy": np.mean(list(self.prediction_accuracy_history)) if self.prediction_accuracy_history else 0.0,
                "graph_complexity": {
                    "state_graph_nodes": self.state_graph.number_of_nodes(),
                    "state_graph_edges": self.state_graph.number_of_edges(),
                    "entity_graph_nodes": self.entity_graph.number_of_nodes(),
                    "entity_graph_edges": self.entity_graph.number_of_edges()
                }
            }
        except Exception as e:
            logger.error("Failed to get world model summary", error=str(e))
            return {"error": str(e)}
