#!/usr/bin/env python3
"""
Real-time RTX 5090 Research Test
Shows the full conversation and thinking process of the agent using the web research tool.
"""

import asyncio
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
from app.agents.factory import AgentBuilderFactory, AgentType, MemoryType
from app.llm.manager import LLMProviderManager
from app.llm.models import ProviderType


async def test_rtx5090_research():
    """Test the RTX 5090 research with full conversation visibility."""

    print("🚀 REAL-TIME RTX 5090 GRAPHICS CARD RESEARCH TEST")
    print("=" * 60)
    print()

    # Use the existing validator approach but with detailed output
    import subprocess
    import sys

    # The research task - Natural query without explicitly mentioning tools
    task = """I need comprehensive information about the NVIDIA RTX 5090 graphics card. Please find:

1. Official specifications and technical details
2. Expected release date and availability
3. Pricing information and market positioning
4. Performance benchmarks and comparisons
5. Key features and improvements over RTX 4090

Please research this thoroughly and provide detailed analysis with current information."""

    print("📋 RESEARCH TASK:")
    print(task)
    print()
    print("🧠 STARTING AGENT EXECUTION...")
    print("-" * 60)

    # Execute using the validator with real-time output
    try:
        cmd = [
            sys.executable,
            "scripts/comprehensive_backend_validator.py",
            "--create", "react",
            "--name", "RTX5090ResearchAgent",
            "--model", "phi4:latest",
            "--task", task
        ]

        print("🚀 Launching agent with web research capabilities...")
        print()

        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        print("� REAL-TIME AGENT OUTPUT:")
        print("=" * 40)

        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                # Highlight important parts
                if "THOUGHT:" in line or "ACTION:" in line or "OBSERVATION:" in line:
                    print(f"🧠 {line.strip()}")
                elif "web_research" in line.lower():
                    print(f"� {line.strip()}")
                elif "✅" in line or "❌" in line:
                    print(f"📊 {line.strip()}")
                elif "Response:" in line:
                    print(f"💬 {line.strip()}")
                else:
                    print(line.strip())

        process.wait()

        print("\n" + "=" * 60)
        print("🎉 RTX 5090 Research Test Complete!")
        print(f"📊 Process exit code: {process.returncode}")

    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_rtx5090_research())
