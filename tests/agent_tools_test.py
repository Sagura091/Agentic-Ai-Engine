#!/usr/bin/env python3
"""
Agent Tools Test - Create and test real tools for agents to use.
This demonstrates the difference between regular LLM responses and true agentic behavior.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Tool implementations
class CalculatorTool:
    """Simple calculator tool for mathematical operations."""
    
    def __init__(self):
        self.name = "calculator"
        self.description = "Perform mathematical calculations including basic arithmetic, percentages, and financial calculations"
        self.usage_count = 0
    
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute a mathematical operation."""
        self.usage_count += 1
        
        try:
            # Parse the operation
            if "+" in operation:
                parts = operation.split("+")
                result = sum(float(p.strip()) for p in parts)
                operation_type = "addition"
            elif "-" in operation:
                parts = operation.split("-")
                result = float(parts[0].strip()) - sum(float(p.strip()) for p in parts[1:])
                operation_type = "subtraction"
            elif "*" in operation or "×" in operation:
                parts = operation.replace("×", "*").split("*")
                result = 1
                for p in parts:
                    result *= float(p.strip())
                operation_type = "multiplication"
            elif "/" in operation or "÷" in operation:
                parts = operation.replace("÷", "/").split("/")
                result = float(parts[0].strip())
                for p in parts[1:]:
                    result /= float(p.strip())
                operation_type = "division"
            elif "%" in operation:
                # Handle percentage calculations
                if " of " in operation:
                    # e.g., "25% of 100"
                    percent_part, base_part = operation.split(" of ")
                    percent = float(percent_part.replace("%", "").strip())
                    base = float(base_part.strip())
                    result = (percent / 100) * base
                    operation_type = "percentage"
                else:
                    # Simple percentage conversion
                    result = float(operation.replace("%", "").strip()) / 100
                    operation_type = "percentage_conversion"
            else:
                # Try to evaluate as a simple expression
                result = eval(operation)
                operation_type = "expression"
            
            return {
                "success": True,
                "result": result,
                "operation": operation,
                "operation_type": operation_type,
                "usage_count": self.usage_count,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": operation,
                "usage_count": self.usage_count,
                "timestamp": datetime.now().isoformat()
            }

class WebSearchTool:
    """Comprehensive web search and research tool."""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for information, analyze market data, research competitors, and gather business intelligence"
        self.usage_count = 0
        self.search_history = []
    
    async def execute(self, query: str, search_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Execute a web search operation."""
        self.usage_count += 1
        search_id = str(uuid.uuid4())
        
        # Simulate different types of searches
        if search_type == "market_research":
            results = await self._simulate_market_research(query)
        elif search_type == "competitor_analysis":
            results = await self._simulate_competitor_analysis(query)
        elif search_type == "financial_data":
            results = await self._simulate_financial_data(query)
        elif search_type == "technology_trends":
            results = await self._simulate_technology_trends(query)
        else:
            results = await self._simulate_general_search(query)
        
        search_record = {
            "search_id": search_id,
            "query": query,
            "search_type": search_type,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results.get("results", [])),
            "success": results.get("success", False)
        }
        
        self.search_history.append(search_record)
        
        return {
            "search_id": search_id,
            "query": query,
            "search_type": search_type,
            "usage_count": self.usage_count,
            "timestamp": datetime.now().isoformat(),
            **results
        }
    
    async def _simulate_market_research(self, query: str) -> Dict[str, Any]:
        """Simulate market research results."""
        await asyncio.sleep(0.5)  # Simulate API call delay
        
        return {
            "success": True,
            "results": [
                {
                    "title": f"Market Analysis: {query}",
                    "summary": f"The {query} market is experiencing significant growth with a projected CAGR of 12.5% over the next 5 years.",
                    "market_size": "$2.4B",
                    "growth_rate": "12.5%",
                    "key_players": ["Company A", "Company B", "Company C"],
                    "trends": ["Digital transformation", "AI integration", "Sustainability focus"]
                },
                {
                    "title": f"Consumer Behavior in {query}",
                    "summary": "Recent studies show 78% of consumers prefer digital-first solutions in this space.",
                    "consumer_preferences": ["Digital-first", "Personalization", "Sustainability"],
                    "price_sensitivity": "Medium",
                    "adoption_rate": "78%"
                }
            ],
            "analysis": {
                "market_opportunity": "High",
                "competition_level": "Medium",
                "entry_barriers": "Low to Medium",
                "recommended_strategy": "Focus on digital innovation and customer experience"
            }
        }
    
    async def _simulate_competitor_analysis(self, query: str) -> Dict[str, Any]:
        """Simulate competitor analysis results."""
        await asyncio.sleep(0.7)  # Simulate API call delay
        
        return {
            "success": True,
            "results": [
                {
                    "competitor": "Market Leader Corp",
                    "market_share": "35%",
                    "strengths": ["Brand recognition", "Distribution network", "R&D investment"],
                    "weaknesses": ["High prices", "Slow innovation", "Customer service"],
                    "recent_moves": ["Acquired startup X", "Launched product Y", "Expanded to Asia"]
                },
                {
                    "competitor": "Innovation Inc",
                    "market_share": "22%",
                    "strengths": ["Technology leadership", "Agile development", "Customer focus"],
                    "weaknesses": ["Limited resources", "Small team", "Market reach"],
                    "recent_moves": ["Raised $50M Series B", "Partnered with Tech Giant", "Launched AI feature"]
                }
            ],
            "competitive_landscape": {
                "market_concentration": "Moderate",
                "innovation_pace": "High",
                "price_competition": "Intense",
                "differentiation_opportunities": ["Customer experience", "Niche markets", "Technology integration"]
            }
        }
    
    async def _simulate_financial_data(self, query: str) -> Dict[str, Any]:
        """Simulate financial data search."""
        await asyncio.sleep(0.3)
        
        return {
            "success": True,
            "results": [
                {
                    "metric": "Revenue Growth",
                    "value": "15.2%",
                    "period": "YoY",
                    "trend": "Increasing"
                },
                {
                    "metric": "Market Valuation",
                    "value": "$1.2B",
                    "change": "+8.5%",
                    "timeframe": "Last quarter"
                }
            ]
        }
    
    async def _simulate_technology_trends(self, query: str) -> Dict[str, Any]:
        """Simulate technology trends search."""
        await asyncio.sleep(0.4)
        
        return {
            "success": True,
            "results": [
                {
                    "trend": "AI Integration",
                    "adoption_rate": "67%",
                    "impact": "High",
                    "timeline": "2024-2025"
                },
                {
                    "trend": "Cloud Migration",
                    "adoption_rate": "89%",
                    "impact": "Medium",
                    "timeline": "Ongoing"
                }
            ]
        }
    
    async def _simulate_general_search(self, query: str) -> Dict[str, Any]:
        """Simulate general web search."""
        await asyncio.sleep(0.2)
        
        return {
            "success": True,
            "results": [
                {
                    "title": f"Information about {query}",
                    "summary": f"Comprehensive information and latest updates about {query}.",
                    "relevance": "High",
                    "source": "Industry Report"
                }
            ]
        }

class DataAnalysisTool:
    """Advanced data analysis and visualization tool."""
    
    def __init__(self):
        self.name = "data_analysis"
        self.description = "Analyze data sets, create visualizations, perform statistical analysis, and generate insights"
        self.usage_count = 0
        self.analysis_history = []
    
    async def execute(self, data: Any, analysis_type: str = "descriptive", **kwargs) -> Dict[str, Any]:
        """Execute data analysis operation."""
        self.usage_count += 1
        analysis_id = str(uuid.uuid4())
        
        if analysis_type == "financial":
            results = await self._analyze_financial_data(data)
        elif analysis_type == "trend":
            results = await self._analyze_trends(data)
        elif analysis_type == "comparison":
            results = await self._compare_data(data)
        else:
            results = await self._descriptive_analysis(data)
        
        analysis_record = {
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "success": results.get("success", False)
        }
        
        self.analysis_history.append(analysis_record)
        
        return {
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "usage_count": self.usage_count,
            "timestamp": datetime.now().isoformat(),
            **results
        }
    
    async def _analyze_financial_data(self, data: Any) -> Dict[str, Any]:
        """Analyze financial data."""
        await asyncio.sleep(0.5)
        
        return {
            "success": True,
            "insights": [
                "Revenue shows consistent 15% quarterly growth",
                "Profit margins improved by 3.2% year-over-year",
                "Cash flow is positive and increasing"
            ],
            "metrics": {
                "growth_rate": "15%",
                "profit_margin": "23.5%",
                "roi": "18.7%"
            },
            "recommendations": [
                "Continue current growth strategy",
                "Consider expanding to new markets",
                "Optimize operational efficiency"
            ]
        }
    
    async def _analyze_trends(self, data: Any) -> Dict[str, Any]:
        """Analyze trends in data."""
        await asyncio.sleep(0.4)
        
        return {
            "success": True,
            "trends": [
                {
                    "trend": "Upward trajectory",
                    "confidence": "95%",
                    "duration": "6 months",
                    "strength": "Strong"
                }
            ],
            "predictions": [
                "25% increase expected in next quarter",
                "Seasonal peak anticipated in Q4"
            ]
        }
    
    async def _compare_data(self, data: Any) -> Dict[str, Any]:
        """Compare data sets."""
        await asyncio.sleep(0.3)
        
        return {
            "success": True,
            "comparison": {
                "performance_vs_benchmark": "+12%",
                "year_over_year": "+18%",
                "vs_competitors": "Above average"
            },
            "key_differences": [
                "Higher customer retention rate",
                "Lower acquisition costs",
                "Faster time to market"
            ]
        }
    
    async def _descriptive_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform descriptive analysis."""
        await asyncio.sleep(0.2)
        
        return {
            "success": True,
            "summary": {
                "total_records": 1000,
                "average": 75.5,
                "median": 72.0,
                "std_deviation": 12.3
            },
            "insights": [
                "Data shows normal distribution",
                "No significant outliers detected",
                "Quality score: 92%"
            ]
        }

class AgentToolWrapper:
    """Wrapper to make tools compatible with LangChain agent interface."""

    def __init__(self, tool_instance):
        self.tool = tool_instance
        self.name = tool_instance.name
        self.description = tool_instance.description

    async def ainvoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain-compatible async invoke method."""
        # Extract the main argument (usually the first one)
        if isinstance(args, dict):
            if "operation" in args:
                return await self.tool.execute(args["operation"], **args)
            elif "query" in args:
                return await self.tool.execute(args["query"], **args)
            elif "data" in args:
                return await self.tool.execute(args["data"], **args)
            else:
                # Use the first value as the main argument
                main_arg = list(args.values())[0] if args else ""
                return await self.tool.execute(main_arg, **args)
        else:
            return await self.tool.execute(str(args))

    def invoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sync version for compatibility."""
        import asyncio
        return asyncio.run(self.ainvoke(args))

# Tool registry for agents
AVAILABLE_TOOLS = {
    "calculator": AgentToolWrapper(CalculatorTool()),
    "web_search": AgentToolWrapper(WebSearchTool()),
    "data_analysis": AgentToolWrapper(DataAnalysisTool())
}

# Raw tools for direct testing
RAW_TOOLS = {
    "calculator": CalculatorTool(),
    "web_search": WebSearchTool(),
    "data_analysis": DataAnalysisTool()
}

async def test_tools_individually():
    """Test each tool individually to ensure they work."""
    print("🔧 TESTING INDIVIDUAL TOOLS")
    print("="*50)
    
    # Test Calculator
    calc = AVAILABLE_TOOLS["calculator"]
    print(f"\n📊 Testing {calc.name}:")
    result = await calc.execute("100000 - 120000 - 24000 - 15000")
    print(f"   Calculation: {result}")
    
    # Test Web Search
    search = AVAILABLE_TOOLS["web_search"]
    print(f"\n🔍 Testing {search.name}:")
    result = await search.execute("startup funding trends 2024", search_type="market_research")
    print(f"   Search results: {len(result.get('results', []))} results found")
    
    # Test Data Analysis
    analysis = AVAILABLE_TOOLS["data_analysis"]
    print(f"\n📈 Testing {analysis.name}:")
    result = await analysis.execute({"revenue": [50000, 55000, 60000]}, analysis_type="financial")
    print(f"   Analysis: {result.get('insights', [])}")

if __name__ == "__main__":
    asyncio.run(test_tools_individually())
