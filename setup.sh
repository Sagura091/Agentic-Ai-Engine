#!/bin/bash
# Bash Setup Script for Agentic AI System
# This script runs the complete system setup

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  AGENTIC AI SYSTEM SETUP${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check Python
echo -e "${BLUE}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}  Python is not installed or not in PATH${NC}"
    echo -e "${YELLOW}  Please install Python 3.11+ from https://www.python.org/${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}  $PYTHON_VERSION${NC}"

# Check if we're in the right directory
if [ ! -f "setup_system.py" ]; then
    echo -e "${RED}Error: setup_system.py not found${NC}"
    echo -e "${YELLOW}Please run this script from the project root directory${NC}"
    exit 1
fi

# Run the Python setup script
echo ""
echo -e "${BLUE}Running setup script...${NC}"
echo ""

if $PYTHON_CMD setup_system.py; then
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  SETUP SUCCESSFUL!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
    else
        echo ""
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}  SETUP COMPLETED WITH WARNINGS${NC}"
        echo -e "${YELLOW}========================================${NC}"
        echo ""
    fi
    
    exit $EXIT_CODE
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  SETUP FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "${WHITE}  1. Make sure Docker is running${NC}"
    echo -e "${WHITE}  2. Check that port 5432 is not in use${NC}"
    echo -e "${WHITE}  3. Verify Python dependencies are installed:${NC}"
    echo -e "${WHITE}     pip install -r requirements.txt${NC}"
    echo ""
    exit 1
fi

