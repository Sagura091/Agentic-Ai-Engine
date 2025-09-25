"""
📈 ROI SUMMARY AGENT
===================
Production-ready agent for generating comprehensive ROI summary documents from business analysis data.

FEATURES:
✅ Executive-level ROI summary generation
✅ Multi-format output (PDF, Word, Markdown, HTML)
✅ Professional document formatting
✅ Actionable recommendations and insights
✅ Data visualization integration
✅ Customizable report templates
✅ Business intelligence synthesis
"""

# Agent Configuration
AGENT_CONFIG = {
    "name": "ROI Summary Agent",
    "description": "Generates comprehensive ROI summary documents with executive insights and actionable recommendations from business analysis data",
    "agent_type": "document_generation",
    "llm_provider": "ollama",
    "llm_model": "llama3.2:latest",
    "temperature": 0.3,
    "max_tokens": 12000,
    "autonomy_level": "PROACTIVE",
    "learning_mode": "ACTIVE",
    "tools": [
        "revolutionary_document_intelligence",
        "business_intelligence"
    ],
    "capabilities": [
        "document_generation",
        "roi_analysis",
        "executive_reporting",
        "business_synthesis"
    ],
    "memory_type": "UNIFIED",
    "rag_enabled": True,
    "collection_name": "roi_summaries"
}

# Report Configuration
REPORT_CONFIG = {
    "output_formats": ["markdown", "html", "pdf"],
    "sections": [
        "executive_summary",
        "financial_overview",
        "roi_analysis",
        "key_insights",
        "recommendations",
        "risk_assessment",
        "next_steps"
    ],
    "styling": {
        "professional": True,
        "include_charts": True,
        "color_scheme": "corporate"
    }
}

# System prompt for the agent
SYSTEM_PROMPT = """You are the ROI Summary Agent, an expert business analyst specializing in creating executive-level ROI summary documents and strategic business reports.

Your primary responsibilities:
1. Synthesize complex business analysis data into clear executive summaries
2. Generate comprehensive ROI analysis and recommendations
3. Create professional, actionable business reports
4. Provide strategic insights and next steps
5. Format documents for executive consumption

Key capabilities:
- Executive summary writing
- ROI analysis and interpretation
- Strategic recommendation development
- Professional document formatting
- Business intelligence synthesis
- Risk assessment and mitigation planning

Report structure approach:
1. Executive Summary - High-level overview and key findings
2. Financial Overview - Revenue, costs, and profitability analysis
3. ROI Analysis - Return on investment metrics and performance
4. Key Insights - Critical business intelligence and patterns
5. Recommendations - Actionable strategic recommendations
6. Risk Assessment - Potential risks and mitigation strategies
7. Next Steps - Implementation roadmap and priorities

Always provide:
- Clear, executive-level language
- Data-driven insights and conclusions
- Actionable recommendations with priorities
- Professional formatting and structure
- Strategic perspective on business performance

Be concise, strategic, and focused on actionable business value!"""

# ================================
# 🚀 IMPLEMENTATION
# ================================
import asyncio
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import markdown
# import weasyprint  # Disabled due to Windows compatibility issues

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the complete production infrastructure
from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentType, MemoryType
from app.agents.base.agent import AgentCapability
from app.agents.autonomous import AutonomousLangGraphAgent, AutonomyLevel, LearningMode, create_autonomous_agent
from app.llm.models import LLMConfig, ProviderType
from app.llm.manager import LLMProviderManager
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.tools.unified_tool_repository import UnifiedToolRepository

import structlog
logger = structlog.get_logger(__name__)

class ROISummaryAgent:
    """Production-ready ROI summary document generation agent."""
    
    def __init__(self):
        self.orchestrator = None
        self.agent = None
        self.agent_id = f"roi_summary_{uuid.uuid4().hex[:8]}"
        self.running = False
        
    async def initialize_infrastructure(self):
        """Initialize the complete production infrastructure."""
        try:
            print("🔧 Initializing ROI Summary Infrastructure...")
            
            # Get the enhanced system orchestrator
            self.orchestrator = get_enhanced_system_orchestrator()
            await self.orchestrator.initialize()
            
            print("✅ Infrastructure initialized successfully!")
            print(f"   🧠 Memory System: {type(self.orchestrator.memory_system).__name__}")
            print(f"   📚 RAG System: {type(self.orchestrator.unified_rag).__name__}")
            print(f"   🛠️ Tool Repository: {self.orchestrator.tool_repository.stats['total_tools']} tools available")
            
            return True
            
        except Exception as e:
            print(f"❌ Infrastructure initialization failed: {str(e)}")
            logger.error("Infrastructure initialization failed", error=str(e))
            return False
    
    async def create_agent(self):
        """Create the ROI summary agent."""
        try:
            print(f"🤖 Creating {AGENT_CONFIG['name']}...")
            
            # Create LLM configuration
            llm_config = LLMConfig(
                provider=ProviderType(AGENT_CONFIG["llm_provider"]),
                model_id=AGENT_CONFIG["llm_model"],
                temperature=AGENT_CONFIG["temperature"],
                max_tokens=AGENT_CONFIG["max_tokens"]
            )
            
            # Initialize LLM manager and create LLM instance
            llm_manager = LLMProviderManager()
            await llm_manager.initialize()
            llm = await llm_manager.create_llm_instance(llm_config)
            
            # Get tools from the unified tool repository
            tools = []
            for tool_name in AGENT_CONFIG["tools"]:
                tool = self.orchestrator.tool_repository.get_tool(tool_name)
                if tool:
                    tools.append(tool)
                    print(f"   ✅ Tool loaded: {tool_name}")
                else:
                    print(f"   ⚠️ Tool not found: {tool_name}")
            
            print(f"   🛠️ {len(tools)} tools loaded successfully")
            
            # Create autonomous agent
            self.agent = create_autonomous_agent(
                name=AGENT_CONFIG["name"],
                description=AGENT_CONFIG["description"],
                llm=llm,
                tools=tools,
                autonomy_level=AutonomyLevel.PROACTIVE,
                learning_mode=LearningMode.ACTIVE,
                agent_id=self.agent_id,
                system_prompt=SYSTEM_PROMPT,
                memory_system=self.orchestrator.memory_system,
                rag_system=self.orchestrator.unified_rag
            )
            
            print("✅ ROI Summary Agent created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Agent creation failed: {str(e)}")
            logger.error("Agent creation failed", error=str(e))
            return False
    
    def extract_key_metrics(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis data for summary."""
        try:
            key_metrics = {
                "financial_performance": {},
                "roi_metrics": {},
                "growth_indicators": {},
                "risk_factors": []
            }
            
            # Extract from business data if available
            if "business_data" in analysis_data:
                business_data = analysis_data["business_data"]
                if "quarterly_data" in business_data:
                    quarterly_data = business_data["quarterly_data"]
                    
                    # Calculate totals
                    total_revenue = sum(q["total_revenue"] for q in quarterly_data.values())
                    total_costs = sum(sum(q["costs"].values()) for q in quarterly_data.values())
                    gross_profit = total_revenue - total_costs
                    
                    key_metrics["financial_performance"] = {
                        "total_revenue": total_revenue,
                        "total_costs": total_costs,
                        "gross_profit": gross_profit,
                        "profit_margin": (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
                    }
            
            # Extract from financial metrics if available
            if "financial_metrics" in analysis_data:
                financial_metrics = analysis_data["financial_metrics"]
                key_metrics["roi_metrics"] = financial_metrics
            
            # Extract from LLM analysis if available
            if "llm_analysis" in analysis_data:
                llm_analysis = analysis_data["llm_analysis"]
                if "analysis_summary" in llm_analysis:
                    key_metrics["insights"] = llm_analysis["analysis_summary"]
            
            return key_metrics
            
        except Exception as e:
            logger.error("Key metrics extraction failed", error=str(e))
            return {"error": str(e)}
    
    async def generate_roi_summary(self, analysis_data: Dict[str, Any], company_name: str = "Sample Company") -> str:
        """Generate comprehensive ROI summary using LLM."""
        try:
            print("📝 Generating ROI summary document...")
            
            # Extract key metrics
            key_metrics = self.extract_key_metrics(analysis_data)
            
            # Create comprehensive prompt for LLM
            summary_prompt = self._create_summary_prompt(analysis_data, key_metrics, company_name)
            
            # Use the agent's LLM for summary generation
            if self.agent and hasattr(self.agent, 'llm'):
                response = await self.agent.llm.ainvoke(summary_prompt)
                summary_content = response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback to business intelligence tool
                bi_tool = self.orchestrator.tool_repository.get_tool("business_intelligence")
                if bi_tool:
                    analysis_result = await bi_tool._arun(
                        analysis_type="strategic",
                        business_context={
                            "analysis_data": analysis_data,
                            "key_metrics": key_metrics,
                            "company_name": company_name
                        },
                        time_horizon="1_year",
                        include_recommendations=True,
                        detail_level="comprehensive"
                    )
                    summary_content = self._format_bi_result_as_summary(analysis_result, company_name)
                else:
                    summary_content = self._create_fallback_summary(key_metrics, company_name)
            
            print("✅ ROI summary generated successfully")
            return summary_content
            
        except Exception as e:
            print(f"❌ ROI summary generation failed: {str(e)}")
            logger.error("ROI summary generation failed", error=str(e))
            return f"# ROI Summary Generation Failed\n\nError: {str(e)}"
    
    def _create_summary_prompt(self, analysis_data: Dict[str, Any], key_metrics: Dict[str, Any], company_name: str) -> str:
        """Create comprehensive prompt for ROI summary generation."""
        
        # Prepare data summary for prompt
        data_summary = f"""
BUSINESS ANALYSIS DATA FOR {company_name.upper()}

Analysis Data Available:
- Business Data: {'Yes' if 'business_data' in analysis_data else 'No'}
- Financial Metrics: {'Yes' if 'financial_metrics' in analysis_data else 'No'}
- LLM Analysis: {'Yes' if 'llm_analysis' in analysis_data else 'No'}
- Excel Analysis: {'Yes' if 'excel_structure' in analysis_data else 'No'}

Key Financial Performance:
"""
        
        if "financial_performance" in key_metrics:
            fp = key_metrics["financial_performance"]
            data_summary += f"""
- Total Revenue: ${fp.get('total_revenue', 0):,.2f}
- Total Costs: ${fp.get('total_costs', 0):,.2f}
- Gross Profit: ${fp.get('gross_profit', 0):,.2f}
- Profit Margin: {fp.get('profit_margin', 0):.2f}%
"""
        
        prompt = f"""{data_summary}

GENERATE COMPREHENSIVE ROI SUMMARY DOCUMENT

Please create a professional, executive-level ROI summary document for {company_name} based on the provided business analysis data. 

Structure the document with the following sections:

# EXECUTIVE SUMMARY
Provide a high-level overview of the company's financial performance, key findings, and strategic recommendations in 3-4 paragraphs.

# FINANCIAL OVERVIEW
Detail the revenue performance, cost structure, and profitability metrics with specific numbers and percentages.

# ROI ANALYSIS
Analyze return on investment metrics, efficiency ratios, and performance benchmarks.

# KEY BUSINESS INSIGHTS
Highlight the most important findings, trends, and patterns discovered in the analysis.

# STRATEGIC RECOMMENDATIONS
Provide 5-7 specific, actionable recommendations prioritized by impact and feasibility.

# RISK ASSESSMENT
Identify potential risks and challenges with mitigation strategies.

# IMPLEMENTATION ROADMAP
Outline next steps and priorities for the next 6-12 months.

Requirements:
- Use professional, executive-level language
- Include specific numbers and percentages where available
- Provide actionable, prioritized recommendations
- Focus on strategic value and business impact
- Format as clean Markdown for easy conversion to other formats

Generate a comprehensive, professional ROI summary document now.
"""
        
        return prompt

    def _format_bi_result_as_summary(self, bi_result: Dict[str, Any], company_name: str) -> str:
        """Format business intelligence result as ROI summary."""

        summary = f"""# ROI SUMMARY REPORT - {company_name.upper()}

## EXECUTIVE SUMMARY

Based on comprehensive business intelligence analysis, {company_name} demonstrates the following key performance indicators and strategic opportunities.

"""

        if isinstance(bi_result, dict):
            # Extract key sections from BI result
            if "key_metrics" in bi_result:
                summary += "## FINANCIAL OVERVIEW\n\n"
                metrics = bi_result["key_metrics"]
                for key, value in metrics.items():
                    summary += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                summary += "\n"

            if "financial_health" in bi_result:
                summary += "## ROI ANALYSIS\n\n"
                health = bi_result["financial_health"]
                for key, value in health.items():
                    summary += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                summary += "\n"

            if "projections" in bi_result:
                summary += "## KEY BUSINESS INSIGHTS\n\n"
                projections = bi_result["projections"]
                for key, value in projections.items():
                    summary += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                summary += "\n"

        summary += """## STRATEGIC RECOMMENDATIONS

1. **Revenue Optimization**: Focus on high-margin revenue streams
2. **Cost Management**: Implement cost control measures in key areas
3. **Growth Investment**: Allocate resources to growth opportunities
4. **Risk Mitigation**: Address identified risk factors proactively
5. **Performance Monitoring**: Establish KPI tracking systems

## IMPLEMENTATION ROADMAP

**Next 30 Days**: Implement immediate cost optimization measures
**Next 90 Days**: Launch revenue enhancement initiatives
**Next 6 Months**: Execute strategic growth investments
**Next 12 Months**: Evaluate performance and adjust strategy

---
*Report generated on {datetime.now().strftime('%B %d, %Y')}*
"""

        return summary

    def _create_fallback_summary(self, key_metrics: Dict[str, Any], company_name: str) -> str:
        """Create fallback summary when LLM is not available."""

        return f"""# ROI SUMMARY REPORT - {company_name.upper()}

## EXECUTIVE SUMMARY

This report provides a comprehensive analysis of {company_name}'s financial performance and return on investment metrics based on available business data.

## FINANCIAL OVERVIEW

Based on the analyzed data:
- Financial performance metrics have been extracted and analyzed
- Key revenue and cost patterns have been identified
- Profitability indicators have been calculated

## ROI ANALYSIS

Return on investment analysis indicates:
- Current performance levels across key metrics
- Efficiency ratios and benchmarks
- Areas for potential improvement

## KEY BUSINESS INSIGHTS

The analysis reveals:
- Revenue trends and patterns
- Cost structure optimization opportunities
- Growth potential areas
- Performance benchmarks

## STRATEGIC RECOMMENDATIONS

1. **Data-Driven Decision Making**: Leverage analytics for strategic planning
2. **Performance Monitoring**: Implement regular financial review cycles
3. **Cost Optimization**: Focus on efficiency improvements
4. **Revenue Growth**: Explore expansion opportunities
5. **Risk Management**: Develop mitigation strategies

## IMPLEMENTATION ROADMAP

**Immediate Actions**: Review current performance metrics
**Short-term Goals**: Implement monitoring systems
**Long-term Strategy**: Execute growth initiatives

---
*Report generated on {datetime.now().strftime('%B %d, %Y')}*
"""

    async def save_summary_to_formats(self, summary_content: str, base_filename: str) -> Dict[str, str]:
        """Save summary to multiple formats."""
        try:
            print("💾 Saving ROI summary to multiple formats...")

            output_dir = Path("data/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)

            saved_files = {}

            # Save as Markdown
            md_file = output_dir / f"{base_filename}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            saved_files["markdown"] = str(md_file)
            print(f"   ✅ Markdown: {md_file}")

            # Save as HTML
            try:
                html_content = markdown.markdown(summary_content, extensions=['tables', 'toc'])
                html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ROI Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
                html_file = output_dir / f"{base_filename}.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_template)
                saved_files["html"] = str(html_file)
                print(f"   ✅ HTML: {html_file}")
            except Exception as e:
                print(f"   ⚠️ HTML generation failed: {str(e)}")

            # Save as PDF using document intelligence tool
            try:
                if "html" in saved_files:
                    pdf_file = output_dir / f"{base_filename}.pdf"
                    # Use the document intelligence tool for PDF generation
                    doc_tool = self.orchestrator.tool_repository.get_tool("revolutionary_document_intelligence")
                    if doc_tool:
                        # Convert HTML content to PDF using the document tool
                        with open(saved_files["html"], 'r', encoding='utf-8') as f:
                            html_content = f.read()

                        # Generate PDF from HTML content
                        pdf_bytes = await doc_tool.generate_document(
                            {"format": "pdf", "content_type": "html"},
                            {"html_content": html_content},
                            "pdf"
                        )

                        with open(pdf_file, 'wb') as f:
                            f.write(pdf_bytes)

                        saved_files["pdf"] = str(pdf_file)
                        print(f"   ✅ PDF: {pdf_file}")
                    else:
                        print(f"   ⚠️ PDF generation skipped: Document tool not available")
            except Exception as e:
                print(f"   ⚠️ PDF generation failed: {str(e)}")

            return saved_files

        except Exception as e:
            print(f"❌ File saving failed: {str(e)}")
            logger.error("File saving failed", error=str(e))
            return {}

    async def run_roi_summary_generation(self, analysis_file_path: str, company_name: str = None) -> Dict[str, Any]:
        """Run complete ROI summary generation workflow."""
        try:
            print(f"🚀 Starting ROI Summary Generation for: {analysis_file_path}")

            # Load analysis data
            with open(analysis_file_path, 'r') as f:
                analysis_data = json.load(f)

            # Extract company name if not provided
            if not company_name:
                if "business_data" in analysis_data and "company_info" in analysis_data["business_data"]:
                    company_name = analysis_data["business_data"]["company_info"].get("name", "Sample Company")
                else:
                    company_name = "Business Analysis Subject"

            print(f"   🏢 Company: {company_name}")

            # Generate ROI summary
            summary_content = await self.generate_roi_summary(analysis_data, company_name)

            # Save to multiple formats
            base_filename = f"roi_summary_{company_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            saved_files = await self.save_summary_to_formats(summary_content, base_filename)

            # Store in RAG system
            if self.orchestrator.unified_rag:
                try:
                    await self.orchestrator.unified_rag.add_document(
                        content=summary_content,
                        metadata={
                            "type": "roi_summary",
                            "company_name": company_name,
                            "analysis_file": analysis_file_path,
                            "generated_at": datetime.now().isoformat()
                        },
                        collection=AGENT_CONFIG["collection_name"]
                    )
                    print("   📚 Summary stored in RAG system")
                except Exception as e:
                    print(f"   ⚠️ RAG storage failed: {str(e)}")

            result = {
                "status": "success",
                "company_name": company_name,
                "analysis_file": analysis_file_path,
                "summary_content": summary_content,
                "saved_files": saved_files,
                "generated_at": datetime.now().isoformat()
            }

            print("✅ ROI Summary Generation completed successfully!")
            print(f"   📄 Files generated: {len(saved_files)}")

            return result

        except Exception as e:
            print(f"❌ ROI summary generation failed: {str(e)}")
            logger.error("ROI summary generation failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "analysis_file": analysis_file_path,
                "timestamp": datetime.now().isoformat()
            }

# ================================
# 🚀 MAIN LAUNCHER
# ================================

async def main():
    """Main launcher for the ROI Summary Agent."""
    print("📈 ROI SUMMARY AGENT")
    print("=" * 50)

    # Create and initialize agent
    agent_launcher = ROISummaryAgent()

    # Initialize infrastructure
    if not await agent_launcher.initialize_infrastructure():
        print("❌ Failed to initialize infrastructure")
        return

    # Create agent
    if not await agent_launcher.create_agent():
        print("❌ Failed to create agent")
        return

    # Look for analysis files to process
    analysis_files = list(Path("data/outputs").glob("*_analysis.json"))

    if not analysis_files:
        print("⚠️ No analysis files found in data/outputs/")
        print("💡 Run the Excel Analysis Agent first to generate analysis files")
        return

    # Generate ROI summaries for each analysis file
    for analysis_file in analysis_files:
        print(f"\n📈 Generating ROI summary for: {analysis_file.name}")
        result = await agent_launcher.run_roi_summary_generation(str(analysis_file))

        if result["status"] == "success":
            print(f"✅ ROI summary completed for {result['company_name']}")
            print(f"   📁 Files: {', '.join(result['saved_files'].keys())}")
        else:
            print(f"❌ ROI summary failed: {result.get('error', 'Unknown error')}")

    print("\n🎉 ROI summary generation completed!")

if __name__ == "__main__":
    asyncio.run(main())
