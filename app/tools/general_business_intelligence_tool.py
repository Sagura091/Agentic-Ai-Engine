"""
General Business Intelligence Tool - Generates realistic business data and analysis
for any company context without relying on stock market data.
"""

import asyncio
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, ClassVar
from pydantic import BaseModel, Field

import structlog
from langchain_core.tools import BaseTool

logger = structlog.get_logger(__name__)


class GeneralBusinessIntelligenceTool(BaseTool):
    """
    General Business Intelligence Tool that generates realistic business metrics
    and analysis based on company context, industry, and size.
    """
    
    name: str = "general_business_intelligence"
    description: str = """
    Generate comprehensive business intelligence analysis including financial metrics,
    market insights, and strategic recommendations for any business context.
    
    This tool creates realistic business data based on:
    - Company size and industry
    - Revenue range and employee count
    - Market position and growth stage
    - Geographic presence and target market
    
    Perfect for business planning, financial modeling, and strategic analysis.
    """

    # Class-level constants for industry benchmarks
    INDUSTRY_BENCHMARKS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "technology": {
            "profit_margin": (15, 25),
            "growth_rate": (20, 40),
            "cash_ratio": (0.3, 0.5),
            "expense_ratio": (0.7, 0.85)
        },
        "retail": {
            "profit_margin": (5, 15),
            "growth_rate": (5, 15),
            "cash_ratio": (0.1, 0.3),
            "expense_ratio": (0.85, 0.95)
        },
        "manufacturing": {
            "profit_margin": (8, 18),
            "growth_rate": (3, 12),
            "cash_ratio": (0.2, 0.4),
            "expense_ratio": (0.75, 0.90)
        },
        "healthcare": {
            "profit_margin": (10, 20),
            "growth_rate": (8, 18),
            "cash_ratio": (0.25, 0.45),
            "expense_ratio": (0.75, 0.88)
        },
        "finance": {
            "profit_margin": (20, 35),
            "growth_rate": (10, 25),
            "cash_ratio": (0.4, 0.6),
            "expense_ratio": (0.65, 0.80)
        }
    }

    SIZE_MULTIPLIERS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "small": {"revenue": (50000, 500000), "employees": (5, 50)},
        "medium": {"revenue": (500000, 10000000), "employees": (50, 500)},
        "large": {"revenue": (10000000, 100000000), "employees": (500, 5000)},
        "enterprise": {"revenue": (100000000, 1000000000), "employees": (5000, 50000)}
    }

    def __init__(self):
        super().__init__()
        self._usage_count = 0
        self._success_rate = 1.0
        self._average_execution_time = 0.0
        self._last_used = None

    def _run(
        self,
        analysis_type: str = "financial",
        business_context: Dict[str, Any] = None,
        focus_areas: Optional[List[str]] = None,
        time_horizon: str = "6_months",
        include_recommendations: bool = True,
        detail_level: str = "comprehensive"
    ) -> str:
        """Generate business intelligence analysis."""
        
        start_time = time.time()
        analysis_id = str(uuid.uuid4())[:8]
        
        try:
            # Update usage statistics
            self._usage_count += 1
            self._last_used = datetime.utcnow()
            
            # Validate and set defaults
            if not business_context:
                business_context = {}
                
            # Log analysis start
            logger.info(
                "General business intelligence analysis started",
                tool_name=self.name,
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                time_horizon=time_horizon,
                detail_level=detail_level,
                usage_count=self._usage_count
            )
            
            # Generate realistic business data
            business_data = self._generate_business_data(business_context)
            
            # Perform analysis
            if analysis_type == "financial":
                analysis_result = self._financial_analysis(business_data, time_horizon, detail_level)
            elif analysis_type == "market":
                analysis_result = self._market_analysis(business_data, focus_areas or [])
            elif analysis_type == "competitive":
                analysis_result = self._competitive_analysis(business_data)
            elif analysis_type == "strategic":
                analysis_result = self._strategic_analysis(business_data, include_recommendations)
            else:
                analysis_result = self._comprehensive_analysis(business_data, time_horizon, include_recommendations)
            
            # Add metadata
            analysis_result.update({
                "analysis_id": analysis_id,
                "generated_at": datetime.now().isoformat(),
                "business_context": business_context,
                "data_source": "Generated Business Intelligence",
                "tool_version": "1.0"
            })
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, success=True)
            
            logger.info(
                "General business intelligence analysis completed successfully",
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                execution_time=execution_time,
                tool_name=self.name,
                usage_count=self._usage_count
            )
            
            return f"General Business Intelligence Analysis (ID: {analysis_id})\n{analysis_result}"
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, success=False)
            logger.error(f"General business intelligence analysis failed: {e}", analysis_id=analysis_id)
            return f"Analysis failed: {str(e)}"

    def _generate_business_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic business data based on context."""
        
        # Extract context information
        company_name = context.get("company_name", "Sample Corp")
        industry = context.get("industry", "technology").lower()
        company_size = context.get("company_size", "medium").lower()
        revenue_range = context.get("revenue_range", "1M-10M")
        employees = context.get("employees", 100)
        market_position = context.get("market_position", "growing")
        
        # Get industry benchmarks
        benchmarks = self.INDUSTRY_BENCHMARKS.get(industry, self.INDUSTRY_BENCHMARKS["technology"])

        # Generate base revenue based on size and range
        if company_size in self.SIZE_MULTIPLIERS:
            revenue_min, revenue_max = self.SIZE_MULTIPLIERS[company_size]["revenue"]
        else:
            revenue_min, revenue_max = 1000000, 10000000
            
        # Parse revenue range if provided
        if "-" in revenue_range:
            try:
                range_parts = revenue_range.replace("M", "000000").replace("K", "000").replace("$", "").split("-")
                revenue_min = int(float(range_parts[0]) * (1000000 if "M" in revenue_range else 1000 if "K" in revenue_range else 1))
                revenue_max = int(float(range_parts[1]) * (1000000 if "M" in revenue_range else 1000 if "K" in revenue_range else 1))
            except:
                pass
        
        # Generate monthly revenue
        annual_revenue = random.randint(revenue_min, revenue_max)
        monthly_revenue = annual_revenue // 12
        
        # Generate expenses based on industry benchmarks
        expense_ratio = random.uniform(*benchmarks["expense_ratio"])
        monthly_expenses = int(monthly_revenue * expense_ratio)
        
        # Calculate profit
        monthly_profit = monthly_revenue - monthly_expenses
        profit_margin = (monthly_profit / monthly_revenue * 100) if monthly_revenue > 0 else 0
        
        # Generate cash position
        cash_ratio = random.uniform(*benchmarks["cash_ratio"])
        cash_position = int(monthly_revenue * cash_ratio * 6)  # 6 months of cash
        
        # Calculate runway
        burn_rate = monthly_expenses - monthly_revenue if monthly_expenses > monthly_revenue else 0
        runway_months = (cash_position / burn_rate) if burn_rate > 0 else float('inf')
        
        return {
            "company_name": company_name,
            "industry": industry,
            "company_size": company_size,
            "employees": employees,
            "market_position": market_position,
            "monthly_revenue": monthly_revenue,
            "monthly_expenses": monthly_expenses,
            "monthly_profit": monthly_profit,
            "profit_margin_percent": round(profit_margin, 2),
            "cash_position": cash_position,
            "burn_rate": max(0, burn_rate),
            "runway_months": round(runway_months, 1) if runway_months != float('inf') else "Infinite",
            "annual_revenue": annual_revenue,
            "growth_rate": random.uniform(*benchmarks["growth_rate"]),
            "benchmarks": benchmarks
        }

    def _financial_analysis(self, data: Dict[str, Any], time_horizon: str, detail_level: str) -> Dict[str, Any]:
        """Perform comprehensive financial analysis."""
        
        return {
            "analysis_type": "financial",
            "key_metrics": {
                "monthly_revenue": data["monthly_revenue"],
                "monthly_expenses": data["monthly_expenses"],
                "monthly_profit": data["monthly_profit"],
                "profit_margin_percent": data["profit_margin_percent"],
                "cash_position": data["cash_position"],
                "burn_rate": data["burn_rate"],
                "runway_months": data["runway_months"]
            },
            "financial_health": {
                "profitability": "Profitable" if data["monthly_profit"] > 0 else "Loss-making",
                "cash_flow": "Positive" if data["monthly_profit"] > 0 else "Negative",
                "sustainability": "High" if data["runway_months"] == "Infinite" or data["runway_months"] > 12 else "Medium" if data["runway_months"] > 6 else "Low"
            },
            "projections": {
                "6_month_revenue_projection": data["monthly_revenue"] * 6 * (1 + data["growth_rate"]/100/2),
                "12_month_revenue_projection": data["annual_revenue"] * (1 + data["growth_rate"]/100),
                "break_even_timeline": "Already profitable" if data["monthly_profit"] > 0 else "6-12 months with optimization"
            },
            "recommendations": self._generate_recommendations(data)
        }

    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations based on business data."""
        recommendations = []
        
        if data["monthly_profit"] <= 0:
            recommendations.append("Focus on revenue growth and cost optimization")
        else:
            recommendations.append("Strong financial position - consider expansion opportunities")
            
        if data["profit_margin_percent"] < 10:
            recommendations.append("Improve profit margins through operational efficiency")
        elif data["profit_margin_percent"] > 20:
            recommendations.append("Excellent margins - maintain competitive advantage")
            
        if data["runway_months"] != "Infinite" and data["runway_months"] < 12:
            recommendations.append("Monitor cash flow closely and consider fundraising")
        else:
            recommendations.append("Healthy cash position supports growth initiatives")
            
        return recommendations

    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update tool performance metrics."""
        # Update average execution time
        if self._usage_count > 1:
            self._average_execution_time = (
                (self._average_execution_time * (self._usage_count - 1) + execution_time) / self._usage_count
            )
        else:
            self._average_execution_time = execution_time
            
        # Update success rate
        if success:
            self._success_rate = (self._success_rate * (self._usage_count - 1) + 1.0) / self._usage_count
        else:
            self._success_rate = (self._success_rate * (self._usage_count - 1) + 0.0) / self._usage_count

    def _comprehensive_analysis(self, data: Dict[str, Any], time_horizon: str, include_recommendations: bool) -> Dict[str, Any]:
        """Perform comprehensive business analysis."""
        financial = self._financial_analysis(data, time_horizon, "comprehensive")
        market = self._market_analysis(data, ["growth", "competition"])

        return {
            "analysis_type": "comprehensive",
            "company_overview": {
                "name": data["company_name"],
                "industry": data["industry"].title(),
                "size": data["company_size"].title(),
                "employees": data["employees"],
                "market_position": data["market_position"].title()
            },
            "financial_metrics": financial["key_metrics"],
            "financial_health": financial["financial_health"],
            "market_insights": market["market_overview"],
            "projections": financial["projections"],
            "recommendations": financial["recommendations"] if include_recommendations else []
        }

    def _market_analysis(self, data: Dict[str, Any], focus_areas: List[str]) -> Dict[str, Any]:
        """Perform market analysis."""
        industry = data["industry"]

        # Industry-specific market data
        market_data = {
            "technology": {"size": "$5.2T", "growth": "8.2%", "trends": ["AI/ML", "Cloud Computing", "Cybersecurity"]},
            "retail": {"size": "$27.7T", "growth": "4.1%", "trends": ["E-commerce", "Omnichannel", "Sustainability"]},
            "manufacturing": {"size": "$14.3T", "growth": "3.8%", "trends": ["Industry 4.0", "Automation", "Supply Chain"]},
            "healthcare": {"size": "$8.3T", "growth": "5.4%", "trends": ["Telemedicine", "Personalized Medicine", "AI Diagnostics"]},
            "finance": {"size": "$26.5T", "growth": "6.1%", "trends": ["Fintech", "Digital Banking", "Blockchain"]}
        }

        market_info = market_data.get(industry, market_data["technology"])

        return {
            "analysis_type": "market",
            "market_overview": {
                "industry": industry.title(),
                "market_size": market_info["size"],
                "growth_rate": market_info["growth"],
                "key_trends": market_info["trends"],
                "competitive_intensity": "High" if industry in ["technology", "retail"] else "Medium"
            },
            "opportunities": [
                f"Leverage {market_info['trends'][0]} for competitive advantage",
                f"Expand into adjacent {industry} markets",
                "Develop strategic partnerships"
            ],
            "threats": [
                "Increasing competition from new entrants",
                "Economic uncertainty affecting demand",
                "Regulatory changes in the industry"
            ]
        }

    def _competitive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform competitive analysis."""
        return {
            "analysis_type": "competitive",
            "competitive_position": {
                "market_share": f"{random.randint(5, 25)}%",
                "competitive_advantage": "Strong product differentiation" if data["profit_margin_percent"] > 15 else "Cost leadership",
                "key_competitors": 3 + random.randint(0, 2),
                "barriers_to_entry": "High" if data["industry"] in ["technology", "finance"] else "Medium"
            },
            "strengths": [
                "Strong financial performance" if data["monthly_profit"] > 0 else "Established market presence",
                "Experienced team and leadership",
                "Solid customer base"
            ],
            "weaknesses": [
                "Limited cash reserves" if data["runway_months"] != "Infinite" and data["runway_months"] < 12 else "Market concentration risk",
                "Operational inefficiencies" if data["profit_margin_percent"] < 10 else "Technology debt"
            ]
        }

    def _strategic_analysis(self, data: Dict[str, Any], include_recommendations: bool) -> Dict[str, Any]:
        """Perform strategic analysis."""
        return {
            "analysis_type": "strategic",
            "strategic_priorities": [
                "Revenue growth and market expansion",
                "Operational efficiency improvement",
                "Technology and innovation investment"
            ],
            "growth_opportunities": [
                "New product development",
                "Geographic expansion",
                "Strategic acquisitions"
            ],
            "risk_factors": [
                "Market volatility" if data["industry"] in ["technology", "finance"] else "Supply chain disruption",
                "Competitive pressure",
                "Regulatory compliance"
            ],
            "recommendations": self._generate_strategic_recommendations(data) if include_recommendations else []
        }

    def _generate_strategic_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []

        if data["growth_rate"] > 20:
            recommendations.append("Capitalize on high growth momentum with aggressive expansion")
        elif data["growth_rate"] < 5:
            recommendations.append("Focus on innovation and market differentiation to drive growth")

        if data["company_size"] == "small":
            recommendations.append("Consider strategic partnerships to accelerate growth")
        elif data["company_size"] == "large":
            recommendations.append("Explore acquisition opportunities for market consolidation")

        if data["profit_margin_percent"] > 20:
            recommendations.append("Invest excess profits in R&D and market expansion")
        else:
            recommendations.append("Implement cost optimization initiatives to improve margins")

        return recommendations

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "usage_count": self._usage_count,
            "success_rate": round(self._success_rate, 3),
            "average_execution_time": round(self._average_execution_time, 3),
            "last_used": self._last_used.isoformat() if self._last_used else None
        }
