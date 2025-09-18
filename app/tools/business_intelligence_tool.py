"""
Comprehensive Business Intelligence Tool for Agentic AI Systems.

This tool provides advanced business analysis capabilities including financial analysis,
market research, competitive intelligence, and strategic planning. It demonstrates
complex tool usage patterns with multiple analysis types and data sources.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta
from enum import Enum

import structlog
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun

from .dynamic_tool_factory import BaseDynamicTool, ToolMetadata, ToolCategory, ToolComplexity

logger = structlog.get_logger(__name__)


class AnalysisType(str, Enum):
    """Types of business intelligence analysis."""
    FINANCIAL = "financial"
    MARKET_RESEARCH = "market_research"
    COMPETITIVE = "competitive"
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    RISK_ASSESSMENT = "risk_assessment"
    GROWTH_ANALYSIS = "growth_analysis"
    CUSTOMER_ANALYSIS = "customer_analysis"


class BusinessIntelligenceInput(BaseModel):
    """Input schema for business intelligence tool."""
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    business_context: Dict[str, Any] = Field(..., description="Business context and data for analysis")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on")
    time_horizon: str = Field(default="6_months", description="Analysis time horizon (3_months, 6_months, 1_year, 3_years)")
    include_recommendations: bool = Field(default=True, description="Include actionable recommendations")
    detail_level: str = Field(default="comprehensive", description="Level of detail (summary, detailed, comprehensive)")


class BusinessIntelligenceTool(BaseDynamicTool):
    """Comprehensive business intelligence and analysis tool."""

    name: str = "business_intelligence"
    description: str = "Perform comprehensive business analysis including financial, market, competitive, and strategic intelligence"
    args_schema: Type[BaseModel] = BusinessIntelligenceInput

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self):
        metadata = ToolMetadata(
            name="business_intelligence",
            description="Advanced business intelligence and analysis tool",
            category=ToolCategory.ANALYSIS,
            complexity=ToolComplexity.ADVANCED,
            tags=["business", "analysis", "intelligence", "strategy", "finance", "market"],
            dependencies=["data_analysis", "market_data"],
            permissions=["data_access", "external_apis"],
            safety_level="safe"
        )
        super().__init__(
            metadata=metadata,
            name=metadata.name,
            description=metadata.description,
            args_schema=BusinessIntelligenceInput
        )
        self.analysis_cache = {}
        self.analysis_history = []
    
    def _run(
        self,
        analysis_type: str,
        business_context: Dict[str, Any],
        focus_areas: List[str] = None,
        time_horizon: str = "6_months",
        include_recommendations: bool = True,
        detail_level: str = "comprehensive",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute business intelligence analysis synchronously."""
        return asyncio.run(self._arun_implementation(
            analysis_type, business_context, focus_areas or [],
            time_horizon, include_recommendations, detail_level
        ))
    
    async def _arun_implementation(
        self,
        analysis_type: str,
        business_context: Dict[str, Any],
        focus_areas: List[str] = None,
        time_horizon: str = "6_months",
        include_recommendations: bool = True,
        detail_level: str = "comprehensive",
        **kwargs
    ) -> str:
        """Execute business intelligence analysis asynchronously."""
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        
        try:
            # Update usage statistics
            self.metadata.usage_count += 1
            self.metadata.last_used = datetime.utcnow()
            
            # Log analysis start
            logger.info(
                "Business intelligence analysis started",
                tool_name=self.name,
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                time_horizon=time_horizon,
                detail_level=detail_level,
                usage_count=self.metadata.usage_count
            )
            
            # Validate inputs
            self._validate_inputs(analysis_type, business_context, time_horizon)
            
            # Check cache for similar analysis
            cache_key = self._generate_cache_key(analysis_type, business_context, time_horizon)
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    logger.info("Using cached analysis result", analysis_id=analysis_id)
                    return self._format_cached_result(cached_result)
            
            # Perform analysis based on type
            analysis_result = await self._perform_analysis(
                analysis_type, business_context, focus_areas or [], 
                time_horizon, include_recommendations, detail_level
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(execution_time, success=True)
            
            # Cache result
            self.analysis_cache[cache_key] = {
                "result": analysis_result,
                "timestamp": datetime.utcnow(),
                "analysis_id": analysis_id
            }
            
            # Record analysis
            analysis_record = {
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_type": analysis_type,
                "time_horizon": time_horizon,
                "detail_level": detail_level,
                "execution_time": execution_time,
                "success": True,
                "focus_areas": focus_areas or [],
                "business_context_keys": list(business_context.keys())
            }
            self.analysis_history.append(analysis_record)
            
            # Log successful analysis
            logger.info(
                "Business intelligence analysis completed successfully",
                tool_name=self.name,
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                execution_time=execution_time,
                usage_count=self.metadata.usage_count
            )
            
            return self._format_analysis_result(analysis_result, analysis_id)
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update performance metrics for failure
            self._update_performance_metrics(execution_time, success=False)
            
            # Record failed analysis
            analysis_record = {
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_type": analysis_type,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }
            self.analysis_history.append(analysis_record)
            
            # Log analysis error
            logger.error(
                "Business intelligence analysis failed",
                tool_name=self.name,
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                error=str(e),
                execution_time=execution_time,
                usage_count=self.metadata.usage_count
            )
            
            return f"Business intelligence analysis error: {str(e)}"
    
    def _validate_inputs(self, analysis_type: str, business_context: Dict[str, Any], time_horizon: str):
        """Validate analysis inputs."""
        valid_analysis_types = [e.value for e in AnalysisType]
        if analysis_type not in valid_analysis_types:
            raise ValueError(f"Invalid analysis type. Must be one of: {valid_analysis_types}")
        
        valid_time_horizons = ["3_months", "6_months", "1_year", "3_years"]
        if time_horizon not in valid_time_horizons:
            raise ValueError(f"Invalid time horizon. Must be one of: {valid_time_horizons}")
        
        if not business_context:
            raise ValueError("Business context cannot be empty")
    
    async def _perform_analysis(
        self, 
        analysis_type: str, 
        business_context: Dict[str, Any],
        focus_areas: List[str],
        time_horizon: str,
        include_recommendations: bool,
        detail_level: str
    ) -> Dict[str, Any]:
        """Perform the actual business intelligence analysis."""
        
        # Simulate analysis processing time
        await asyncio.sleep(0.5)
        
        if analysis_type == AnalysisType.FINANCIAL:
            return await self._financial_analysis(business_context, time_horizon, detail_level)
        elif analysis_type == AnalysisType.MARKET_RESEARCH:
            return await self._market_research_analysis(business_context, focus_areas, time_horizon)
        elif analysis_type == AnalysisType.COMPETITIVE:
            return await self._competitive_analysis(business_context, focus_areas, time_horizon)
        elif analysis_type == AnalysisType.STRATEGIC:
            return await self._strategic_analysis(business_context, time_horizon, include_recommendations)
        elif analysis_type == AnalysisType.GROWTH_ANALYSIS:
            return await self._growth_analysis(business_context, time_horizon, detail_level)
        else:
            return await self._general_business_analysis(business_context, analysis_type, time_horizon)
    
    async def _financial_analysis(self, context: Dict[str, Any], time_horizon: str, detail_level: str) -> Dict[str, Any]:
        """Perform comprehensive financial analysis."""
        await asyncio.sleep(0.3)
        
        # Extract financial data
        revenue = context.get("revenue", 0)
        expenses = context.get("expenses", 0)
        cash = context.get("cash", 0)
        
        # Calculate key metrics
        profit = revenue - expenses
        profit_margin = (profit / revenue * 100) if revenue > 0 else 0
        burn_rate = expenses
        runway_months = (cash / burn_rate) if burn_rate > 0 else float('inf')
        
        return {
            "analysis_type": "financial",
            "key_metrics": {
                "monthly_revenue": revenue,
                "monthly_expenses": expenses,
                "monthly_profit": profit,
                "profit_margin_percent": round(profit_margin, 2),
                "cash_position": cash,
                "burn_rate": burn_rate,
                "runway_months": round(runway_months, 1) if runway_months != float('inf') else "Infinite"
            },
            "financial_health": {
                "profitability": "Profitable" if profit > 0 else "Loss-making",
                "cash_flow": "Positive" if profit > 0 else "Negative",
                "sustainability": "High" if runway_months > 12 else "Medium" if runway_months > 6 else "Low"
            },
            "projections": {
                "6_month_revenue_projection": revenue * 6 * 1.15,  # Assume 15% growth
                "break_even_timeline": "Already profitable" if profit > 0 else "6-12 months with optimization"
            },
            "recommendations": [
                "Monitor cash flow closely" if runway_months < 12 else "Strong financial position",
                "Focus on revenue growth" if profit_margin < 20 else "Optimize operational efficiency",
                "Consider fundraising" if runway_months < 6 else "Maintain current trajectory"
            ]
        }
    
    async def _market_research_analysis(self, context: Dict[str, Any], focus_areas: List[str], time_horizon: str) -> Dict[str, Any]:
        """Perform market research analysis."""
        await asyncio.sleep(0.4)
        
        industry = context.get("industry", "Technology")
        target_market = context.get("target_market", "SMB")
        
        return {
            "analysis_type": "market_research",
            "market_overview": {
                "industry": industry,
                "market_size": "$2.4B",
                "growth_rate": "12.5% CAGR",
                "maturity": "Growth stage",
                "key_trends": ["Digital transformation", "AI adoption", "Remote work"]
            },
            "target_market_analysis": {
                "segment": target_market,
                "size": "450,000 businesses",
                "characteristics": ["Cost-conscious", "Technology adopters", "Growth-focused"],
                "pain_points": ["Limited resources", "Need for efficiency", "Competitive pressure"]
            },
            "opportunities": [
                "Underserved niche markets",
                "Emerging technology adoption",
                "Geographic expansion potential"
            ],
            "threats": [
                "Increasing competition",
                "Economic uncertainty",
                "Technology disruption"
            ],
            "recommendations": [
                "Focus on differentiation",
                "Invest in customer acquisition",
                "Monitor competitive landscape"
            ]
        }
    
    async def _competitive_analysis(self, context: Dict[str, Any], focus_areas: List[str], time_horizon: str) -> Dict[str, Any]:
        """Perform competitive analysis."""
        await asyncio.sleep(0.3)
        
        return {
            "analysis_type": "competitive",
            "competitive_landscape": {
                "market_concentration": "Moderate",
                "number_of_competitors": "15-20 direct competitors",
                "competitive_intensity": "High"
            },
            "key_competitors": [
                {
                    "name": "Market Leader Corp",
                    "market_share": "35%",
                    "strengths": ["Brand recognition", "Resources", "Distribution"],
                    "weaknesses": ["High prices", "Slow innovation"],
                    "threat_level": "High"
                },
                {
                    "name": "Innovation Startup",
                    "market_share": "8%",
                    "strengths": ["Technology", "Agility", "Customer focus"],
                    "weaknesses": ["Limited resources", "Brand awareness"],
                    "threat_level": "Medium"
                }
            ],
            "competitive_advantages": [
                "Unique value proposition",
                "Superior customer experience",
                "Cost efficiency"
            ],
            "strategic_recommendations": [
                "Differentiate through innovation",
                "Build strategic partnerships",
                "Focus on customer retention"
            ]
        }
    
    async def _strategic_analysis(self, context: Dict[str, Any], time_horizon: str, include_recommendations: bool) -> Dict[str, Any]:
        """Perform strategic analysis."""
        await asyncio.sleep(0.4)
        
        return {
            "analysis_type": "strategic",
            "strategic_position": {
                "current_stage": "Growth",
                "core_competencies": ["Technology", "Customer service", "Innovation"],
                "strategic_assets": ["IP portfolio", "Customer base", "Team expertise"]
            },
            "swot_analysis": {
                "strengths": ["Strong product", "Experienced team", "Market traction"],
                "weaknesses": ["Limited resources", "Brand awareness", "Geographic reach"],
                "opportunities": ["Market expansion", "New products", "Partnerships"],
                "threats": ["Competition", "Economic downturn", "Technology changes"]
            },
            "strategic_options": [
                {
                    "option": "Market penetration",
                    "description": "Increase market share in current markets",
                    "investment_required": "Medium",
                    "risk_level": "Low",
                    "potential_return": "Medium"
                },
                {
                    "option": "Product development",
                    "description": "Develop new products for current markets",
                    "investment_required": "High",
                    "risk_level": "Medium",
                    "potential_return": "High"
                }
            ],
            "recommended_strategy": "Focus on market penetration while investing in product development",
            "key_success_factors": [
                "Customer acquisition efficiency",
                "Product-market fit optimization",
                "Operational scalability"
            ]
        }
    
    async def _growth_analysis(self, context: Dict[str, Any], time_horizon: str, detail_level: str) -> Dict[str, Any]:
        """Perform growth analysis."""
        await asyncio.sleep(0.3)
        
        current_revenue = context.get("revenue", 50000)
        growth_target = context.get("growth_target", 50)  # 50% growth
        
        return {
            "analysis_type": "growth_analysis",
            "current_metrics": {
                "monthly_revenue": current_revenue,
                "growth_rate": "15% MoM",
                "customer_acquisition_cost": "$150",
                "customer_lifetime_value": "$2400"
            },
            "growth_projections": {
                "target_growth": f"{growth_target}%",
                "target_revenue": current_revenue * (1 + growth_target/100),
                "timeline": time_horizon,
                "probability": "75%"
            },
            "growth_drivers": [
                "Customer acquisition scaling",
                "Product expansion",
                "Market penetration",
                "Pricing optimization"
            ],
            "growth_barriers": [
                "Resource constraints",
                "Market saturation",
                "Competitive pressure"
            ],
            "action_plan": [
                "Increase marketing spend by 40%",
                "Hire 2 additional sales reps",
                "Launch referral program",
                "Optimize conversion funnel"
            ]
        }
    
    async def _general_business_analysis(self, context: Dict[str, Any], analysis_type: str, time_horizon: str) -> Dict[str, Any]:
        """Perform general business analysis."""
        await asyncio.sleep(0.2)
        
        return {
            "analysis_type": analysis_type,
            "summary": f"Comprehensive {analysis_type} analysis completed",
            "key_findings": [
                "Business shows strong fundamentals",
                "Growth opportunities identified",
                "Risk factors manageable"
            ],
            "recommendations": [
                "Continue current strategy",
                "Monitor key metrics",
                "Prepare for scaling"
            ]
        }
    
    def _generate_cache_key(self, analysis_type: str, business_context: Dict[str, Any], time_horizon: str) -> str:
        """Generate cache key for analysis."""
        context_hash = hash(json.dumps(business_context, sort_keys=True))
        return f"{analysis_type}_{time_horizon}_{context_hash}"
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid."""
        cache_time = cached_result["timestamp"]
        return datetime.utcnow() - cache_time < timedelta(hours=1)
    
    def _format_cached_result(self, cached_result: Dict[str, Any]) -> str:
        """Format cached analysis result."""
        result = cached_result["result"]
        return f"Business Intelligence Analysis (Cached)\n{json.dumps(result, indent=2)}"
    
    def _format_analysis_result(self, analysis_result: Dict[str, Any], analysis_id: str) -> str:
        """Format analysis result for output."""
        return f"Business Intelligence Analysis (ID: {analysis_id})\n{json.dumps(analysis_result, indent=2)}"
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update tool performance metrics."""
        # Update average execution time
        total_executions = self.metadata.usage_count
        current_avg = self.metadata.average_execution_time
        
        new_avg = ((current_avg * (total_executions - 1)) + execution_time) / total_executions
        self.metadata.average_execution_time = new_avg
        
        # Update success rate
        if total_executions == 1:
            self.metadata.success_rate = 1.0 if success else 0.0
        else:
            successful_executions = int(self.metadata.success_rate * (total_executions - 1))
            if success:
                successful_executions += 1
            
            self.metadata.success_rate = successful_executions / total_executions
        
        # Update last updated timestamp
        self.metadata.last_updated = datetime.utcnow()
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history.copy()
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        return {
            "tool_name": self.name,
            "total_usage": self.metadata.usage_count,
            "success_rate": self.metadata.success_rate,
            "average_execution_time": self.metadata.average_execution_time,
            "last_used": self.metadata.last_used.isoformat() if self.metadata.last_used else None,
            "total_analyses": len(self.analysis_history),
            "successful_analyses": sum(1 for analysis in self.analysis_history if analysis["success"]),
            "failed_analyses": sum(1 for analysis in self.analysis_history if not analysis["success"]),
            "analysis_types_used": list(set(analysis["analysis_type"] for analysis in self.analysis_history)),
            "cache_hit_rate": len(self.analysis_cache) / max(1, self.metadata.usage_count),
            "recent_analyses": self.analysis_history[-3:] if self.analysis_history else []
        }


# Create tool instance for registration
business_intelligence_tool = BusinessIntelligenceTool()
