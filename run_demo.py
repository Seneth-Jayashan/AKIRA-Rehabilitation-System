"""Entrypoint to launch the AI Elbow Rehabilitation Interactive Demo.

Usage:
    python run_demo.py
    or
    python -m uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("Starting AI Elbow Rehabilitation Interactive Demo...")
    print("Open http://127.0.0.1:8000 in your browser.")
    uvicorn.run("demo.backend.app:app", host="127.0.0.1", port=8000, reload=False)
