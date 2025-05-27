#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔬 Running LLM Entropy Tests...${NC}"

# Get project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create and activate virtual environment if needed
if [ ! -d "${PROJECT_ROOT}/venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv "${PROJECT_ROOT}/venv"
fi

source "${PROJECT_ROOT}/venv/bin/activate"

# Install specific versions of pytest and plugins
echo -e "${BLUE}Installing test dependencies...${NC}"
pip install -q -U pip
pip install -q 'pytest>=7.0.0,<8.0.0' \
    'pytest-cov>=4.0.0,<5.0.0' \
    'pytest-xdist>=3.0.0,<4.0.0' \
    'pytest-mock>=3.10.0' \
    'coverage-badge>=1.1.0,<2.0.0'

# Create coverage directory
mkdir -p "${PROJECT_ROOT}/coverage"

# Run pytest with appropriate configuration
echo -e "${BLUE}Running tests...${NC}"
PYTHONPATH="${PROJECT_ROOT}" pytest \
    "${SCRIPT_DIR}/tests" \
    -v \
    --capture=no \
    --color=yes \
    --cov="${SCRIPT_DIR}" \
    --cov-report=term-missing \
    --cov-report=html:"${PROJECT_ROOT}/coverage" \
    --junitxml="${PROJECT_ROOT}/junit.xml"

# Store exit code and report status
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}"
else
    echo -e "\n${RED}❌ Some tests failed${NC}"
fi

exit $TEST_EXIT_CODE
