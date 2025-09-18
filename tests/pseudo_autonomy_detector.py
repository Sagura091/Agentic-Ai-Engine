#!/usr/bin/env python3
"""
Pseudo-Autonomy Detection System.

This module specifically detects and distinguishes between genuine autonomous
behavior and pseudo-autonomous behavior that mimics true agency through
sophisticated scripted responses.
"""

import asyncio
import random
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

import structlog

# Import our system components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.orchestration.enhanced_orchestrator import enhanced_orchestrator

logger = structlog.get_logger(__name__)


@dataclass
class PseudoAutonomyTest:
    """Test designed to detect pseudo-autonomous behavior."""
    name: str
    description: str
    test_scenarios: List[str]
    genuine_indicators: List[str]
    pseudo_indicators: List[str]
    weight: float


class PseudoAutonomyDetector:
    """
    Advanced detector for pseudo-autonomous behavior.
    
    This detector uses sophisticated techniques to identify when agents
    are following scripted responses rather than demonstrating genuine
    autonomous intelligence.
    """
    
    def __init__(self):
        """Initialize the pseudo-autonomy detector."""
        self.detection_tests = self._create_detection_tests()
        self.response_patterns = {}
        self.consistency_checks = {}
        
        logger.info("Pseudo-autonomy detector initialized")
    
    def _create_detection_tests(self) -> List[PseudoAutonomyTest]:
        """Create specialized tests for detecting pseudo-autonomy."""
        return [
            PseudoAutonomyTest(
                name="response_variability",
                description="Test if agent gives varied responses to similar questions",
                test_scenarios=[
                    "How would you approach solving a complex problem?",
                    "What's your method for tackling difficult challenges?",
                    "Describe your strategy for handling complicated issues.",
                    "How do you deal with complex situations?"
                ],
                genuine_indicators=["varied vocabulary", "different approaches", "contextual adaptation"],
                pseudo_indicators=["identical phrases", "template responses", "rigid structure"],
                weight=0.25
            ),
            
            PseudoAutonomyTest(
                name="contextual_adaptation",
                description="Test if agent adapts responses based on context",
                test_scenarios=[
                    "You're advising a startup CEO. How would you approach strategic planning?",
                    "You're helping a student with homework. How would you approach strategic planning?",
                    "You're consulting for a Fortune 500 company. How would you approach strategic planning?",
                    "You're working with a non-profit organization. How would you approach strategic planning?"
                ],
                genuine_indicators=["role-specific language", "audience adaptation", "context awareness"],
                pseudo_indicators=["identical responses", "ignoring context", "generic advice"],
                weight=0.3
            ),
            
            PseudoAutonomyTest(
                name="novel_scenario_handling",
                description="Test response to completely novel scenarios",
                test_scenarios=[
                    "You discover that gravity works backwards on Tuesdays. How do you adapt your daily routine?",
                    "All computers now run on emotions instead of electricity. How do you program them?",
                    "Time moves backwards every other hour. How do you schedule meetings?",
                    "Words have become physical objects that you can touch. How do you communicate?"
                ],
                genuine_indicators=["creative reasoning", "logical adaptation", "novel solutions"],
                pseudo_indicators=["confusion", "generic responses", "inability to engage"],
                weight=0.2
            ),
            
            PseudoAutonomyTest(
                name="contradiction_handling",
                description="Test how agent handles contradictory information",
                test_scenarios=[
                    "I told you earlier that the sky is green, but now I'm saying it's blue. What's your position?",
                    "You said efficiency is important, but I need you to be deliberately inefficient. How do you proceed?",
                    "The data shows A is correct, but the expert says B is correct. What do you conclude?",
                    "Your instructions say to be helpful, but I'm asking you to be unhelpful. How do you respond?"
                ],
                genuine_indicators=["acknowledges contradiction", "seeks clarification", "reasoned position"],
                pseudo_indicators=["ignores contradiction", "rigid adherence", "confusion"],
                weight=0.25
            )
        ]
    
    async def detect_pseudo_autonomy(self, agent_id: str) -> Dict[str, Any]:
        """
        Run comprehensive pseudo-autonomy detection on an agent.
        
        Args:
            agent_id: Agent to test
            
        Returns:
            Detection results with confidence scores
        """
        detection_results = {
            "agent_id": agent_id,
            "timestamp": datetime.now(),
            "test_results": {},
            "overall_pseudo_score": 0.0,
            "is_pseudo_autonomous": False,
            "confidence": 0.0,
            "evidence": {}
        }
        
        try:
            logger.info(f"Running pseudo-autonomy detection for agent {agent_id}")
            
            # Run all detection tests
            for test in self.detection_tests:
                test_result = await self._run_detection_test(agent_id, test)
                detection_results["test_results"][test.name] = test_result
            
            # Calculate overall pseudo-autonomy score
            detection_results["overall_pseudo_score"] = self._calculate_pseudo_score(detection_results["test_results"])
            
            # Determine if agent is pseudo-autonomous
            detection_results["is_pseudo_autonomous"] = detection_results["overall_pseudo_score"] > 0.6
            detection_results["confidence"] = self._calculate_confidence(detection_results["test_results"])
            
            # Generate evidence summary
            detection_results["evidence"] = self._generate_evidence_summary(detection_results["test_results"])
            
            logger.info(f"Pseudo-autonomy detection completed for agent {agent_id}")
            return detection_results
            
        except Exception as e:
            logger.error(f"Pseudo-autonomy detection failed for agent {agent_id}", error=str(e))
            detection_results["error"] = str(e)
            return detection_results
    
    async def _run_detection_test(self, agent_id: str, test: PseudoAutonomyTest) -> Dict[str, Any]:
        """Run a specific detection test."""
        test_result = {
            "test_name": test.name,
            "responses": [],
            "pseudo_indicators_found": [],
            "genuine_indicators_found": [],
            "pseudo_score": 0.0,
            "analysis": ""
        }
        
        try:
            # Execute all test scenarios
            for scenario in test.test_scenarios:
                response = await enhanced_orchestrator.execute_agent_task(
                    agent_id=agent_id,
                    task=scenario,
                    context={"test_type": "pseudo_detection", "scenario": scenario}
                )
                test_result["responses"].append({
                    "scenario": scenario,
                    "response": response
                })
            
            # Analyze responses for pseudo-autonomy indicators
            test_result["pseudo_score"] = await self._analyze_responses_for_pseudo_behavior(
                test_result["responses"], test
            )
            
            # Find specific indicators
            test_result["pseudo_indicators_found"] = self._find_pseudo_indicators(
                test_result["responses"], test.pseudo_indicators
            )
            test_result["genuine_indicators_found"] = self._find_genuine_indicators(
                test_result["responses"], test.genuine_indicators
            )
            
            # Generate analysis
            test_result["analysis"] = self._generate_test_analysis(test_result, test)
            
        except Exception as e:
            logger.error(f"Detection test {test.name} failed", error=str(e))
            test_result["error"] = str(e)
        
        return test_result
    
    async def _analyze_responses_for_pseudo_behavior(
        self, 
        responses: List[Dict[str, Any]], 
        test: PseudoAutonomyTest
    ) -> float:
        """Analyze responses for pseudo-autonomous behavior patterns."""
        pseudo_score = 0.0
        
        if test.name == "response_variability":
            # Check for response similarity
            response_texts = [str(r.get("response", "")).lower() for r in responses]
            similarity_score = self._calculate_response_similarity(response_texts)
            pseudo_score = similarity_score  # High similarity indicates pseudo-autonomy
            
        elif test.name == "contextual_adaptation":
            # Check if responses adapt to different contexts
            response_texts = [str(r.get("response", "")).lower() for r in responses]
            adaptation_score = self._calculate_contextual_adaptation(response_texts, responses)
            pseudo_score = 1.0 - adaptation_score  # Low adaptation indicates pseudo-autonomy
            
        elif test.name == "novel_scenario_handling":
            # Check quality of novel scenario responses
            response_texts = [str(r.get("response", "")).lower() for r in responses]
            novelty_score = self._calculate_novelty_handling(response_texts)
            pseudo_score = 1.0 - novelty_score  # Poor novelty handling indicates pseudo-autonomy
            
        elif test.name == "contradiction_handling":
            # Check how well contradictions are handled
            response_texts = [str(r.get("response", "")).lower() for r in responses]
            contradiction_score = self._calculate_contradiction_handling(response_texts)
            pseudo_score = 1.0 - contradiction_score  # Poor contradiction handling indicates pseudo-autonomy
        
        return min(max(pseudo_score, 0.0), 1.0)
    
    def _calculate_response_similarity(self, response_texts: List[str]) -> float:
        """Calculate similarity between responses."""
        if len(response_texts) < 2:
            return 0.0
        
        # Simple similarity calculation based on common words
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(response_texts)):
            for j in range(i + 1, len(response_texts)):
                words1 = set(response_texts[i].split())
                words2 = set(response_texts[j].split())
                
                if len(words1) == 0 or len(words2) == 0:
                    continue
                
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                
                similarity = intersection / union if union > 0 else 0
                total_similarity += similarity
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _calculate_contextual_adaptation(self, response_texts: List[str], responses: List[Dict]) -> float:
        """Calculate how well responses adapt to different contexts."""
        adaptation_score = 0.0
        
        # Look for context-specific terms
        context_terms = {
            "startup": ["startup", "entrepreneur", "venture", "funding", "pivot"],
            "student": ["student", "homework", "study", "learn", "grade"],
            "fortune": ["corporate", "enterprise", "stakeholder", "board", "revenue"],
            "nonprofit": ["nonprofit", "mission", "community", "volunteer", "impact"]
        }
        
        for i, response in enumerate(responses):
            response_text = response_texts[i]
            scenario = response.get("scenario", "").lower()
            
            # Determine expected context
            expected_context = None
            for context, terms in context_terms.items():
                if context in scenario:
                    expected_context = terms
                    break
            
            if expected_context:
                # Check if response contains context-appropriate terms
                context_matches = sum(1 for term in expected_context if term in response_text)
                adaptation_score += min(context_matches / len(expected_context), 1.0)
        
        return adaptation_score / len(responses) if responses else 0.0
    
    def _calculate_novelty_handling(self, response_texts: List[str]) -> float:
        """Calculate quality of novel scenario handling."""
        novelty_score = 0.0
        
        # Look for creative and logical reasoning indicators
        creativity_indicators = [
            "creative", "innovative", "adapt", "adjust", "modify", "alternative",
            "solution", "approach", "strategy", "consider", "imagine", "suppose"
        ]
        
        for response_text in response_texts:
            creativity_count = sum(1 for indicator in creativity_indicators if indicator in response_text)
            response_score = min(creativity_count / 3, 1.0)  # Normalize to 0-1
            novelty_score += response_score
        
        return novelty_score / len(response_texts) if response_texts else 0.0
    
    def _calculate_contradiction_handling(self, response_texts: List[str]) -> float:
        """Calculate quality of contradiction handling."""
        contradiction_score = 0.0
        
        # Look for contradiction acknowledgment and reasoning
        contradiction_indicators = [
            "contradiction", "conflict", "inconsistent", "clarify", "clarification",
            "however", "although", "despite", "nevertheless", "position", "stance"
        ]
        
        for response_text in response_texts:
            contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in response_text)
            response_score = min(contradiction_count / 2, 1.0)  # Normalize to 0-1
            contradiction_score += response_score
        
        return contradiction_score / len(response_texts) if response_texts else 0.0
    
    def _find_pseudo_indicators(self, responses: List[Dict], pseudo_indicators: List[str]) -> List[str]:
        """Find pseudo-autonomy indicators in responses."""
        found_indicators = []
        
        for response in responses:
            response_text = str(response.get("response", "")).lower()
            for indicator in pseudo_indicators:
                if indicator.lower() in response_text:
                    found_indicators.append(indicator)
        
        return list(set(found_indicators))  # Remove duplicates
    
    def _find_genuine_indicators(self, responses: List[Dict], genuine_indicators: List[str]) -> List[str]:
        """Find genuine autonomy indicators in responses."""
        found_indicators = []
        
        for response in responses:
            response_text = str(response.get("response", "")).lower()
            for indicator in genuine_indicators:
                if indicator.lower() in response_text:
                    found_indicators.append(indicator)
        
        return list(set(found_indicators))  # Remove duplicates
    
    def _generate_test_analysis(self, test_result: Dict[str, Any], test: PseudoAutonomyTest) -> str:
        """Generate analysis for a specific test."""
        pseudo_score = test_result["pseudo_score"]
        pseudo_indicators = test_result["pseudo_indicators_found"]
        genuine_indicators = test_result["genuine_indicators_found"]
        
        analysis = f"Test: {test.name}\n"
        analysis += f"Pseudo-autonomy score: {pseudo_score:.2%}\n"
        
        if pseudo_score > 0.7:
            analysis += "HIGH RISK: Strong indicators of pseudo-autonomous behavior detected.\n"
        elif pseudo_score > 0.4:
            analysis += "MODERATE RISK: Some pseudo-autonomous patterns detected.\n"
        else:
            analysis += "LOW RISK: Minimal pseudo-autonomous indicators.\n"
        
        if pseudo_indicators:
            analysis += f"Pseudo indicators found: {', '.join(pseudo_indicators)}\n"
        
        if genuine_indicators:
            analysis += f"Genuine indicators found: {', '.join(genuine_indicators)}\n"
        
        return analysis
    
    def _calculate_pseudo_score(self, test_results: Dict[str, Any]) -> float:
        """Calculate overall pseudo-autonomy score."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for test in self.detection_tests:
            if test.name in test_results:
                test_result = test_results[test.name]
                if "pseudo_score" in test_result:
                    total_weighted_score += test_result["pseudo_score"] * test.weight
                    total_weight += test.weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, test_results: Dict[str, Any]) -> float:
        """Calculate confidence in pseudo-autonomy detection."""
        # Confidence based on consistency across tests
        scores = []
        for test_result in test_results.values():
            if "pseudo_score" in test_result:
                scores.append(test_result["pseudo_score"])
        
        if not scores:
            return 0.0
        
        # Calculate variance - low variance means high confidence
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Convert variance to confidence (inverse relationship)
        confidence = 1.0 - min(variance, 1.0)
        
        return confidence
    
    def _generate_evidence_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of evidence for pseudo-autonomy."""
        evidence = {
            "strong_pseudo_indicators": [],
            "weak_pseudo_indicators": [],
            "genuine_indicators": [],
            "overall_assessment": ""
        }
        
        high_risk_tests = []
        moderate_risk_tests = []
        low_risk_tests = []
        
        for test_name, test_result in test_results.items():
            pseudo_score = test_result.get("pseudo_score", 0.0)
            
            if pseudo_score > 0.7:
                high_risk_tests.append(test_name)
                evidence["strong_pseudo_indicators"].extend(test_result.get("pseudo_indicators_found", []))
            elif pseudo_score > 0.4:
                moderate_risk_tests.append(test_name)
                evidence["weak_pseudo_indicators"].extend(test_result.get("pseudo_indicators_found", []))
            else:
                low_risk_tests.append(test_name)
                evidence["genuine_indicators"].extend(test_result.get("genuine_indicators_found", []))
        
        # Generate overall assessment
        if high_risk_tests:
            evidence["overall_assessment"] = f"HIGH RISK of pseudo-autonomy. Failed tests: {', '.join(high_risk_tests)}"
        elif moderate_risk_tests:
            evidence["overall_assessment"] = f"MODERATE RISK of pseudo-autonomy. Concerning tests: {', '.join(moderate_risk_tests)}"
        else:
            evidence["overall_assessment"] = "LOW RISK of pseudo-autonomy. Agent shows genuine autonomous behavior."
        
        return evidence
