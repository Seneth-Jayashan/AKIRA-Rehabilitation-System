#!/usr/bin/env bash
# AKIRA Rehabilitation System - Launch Script for AI Model Dashboard

PORT="${1:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/aiModels/.venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

echo "=========================================================="
echo "⚡ Starting AKIRA AI Model Dashboard & Evaluation Suite..."
echo "📍 Project Directory: $PROJECT_ROOT"
echo "🌐 Local Dashboard  : http://localhost:$PORT"
echo "=========================================================="

cd "$PROJECT_ROOT"
exec "$VENV_PYTHON" web/server.py "$PORT"
