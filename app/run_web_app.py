"""
========================================================================================
AKIRA KNEE REHABILITATION SYSTEM: WEB APPLICATION LAUNCHER (run_web_app.py)
========================================================================================
WHAT THIS FILE IS:
  The entry launcher script for the entire AKIRA Knee Rehabilitation system.
  It launches the FastAPI backend server on localhost (127.0.0.1:8000) and automatically
  opens the default web browser to the interactive clinical dashboard.

HOW TO RUN:
  From terminal:
      python run_web_app.py
  Or double click:
      run_web_app.bat

COMPONENTS INITIALIZED:
  1. Background Browser Launcher Thread: Opens http://127.0.0.1:8000 after 1.5s delay.
  2. Uvicorn Server: Starts ASGI server hosting the FastAPI backend and static web files.
========================================================================================
"""

import os
import sys
import time
import webbrowser
import threading
import uvicorn


def open_browser():
    """
    WHAT THIS DOES:
      Waits 1.5 seconds for the FastAPI server to initialize, then automatically
      opens the default web browser to the dashboard URL.
    """
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n[OK] Opening web browser at {url} ...")
    webbrowser.open(url)


if __name__ == "__main__":
    print("=" * 75)
    print("AKIRA Edge-AI Knee Rehabilitation Dashboard")
    print("Initializing server at http://127.0.0.1:8000 ...")
    print("Press CTRL+C anytime in this window to stop the server.")
    print("=" * 75)

    # Launch browser automatically in a background daemon thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Configure directory paths so the API can resolve local imports and datasets
    api_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api")
    sys.path.insert(0, api_dir)
    os.chdir(api_dir)

    # Start Uvicorn ASGI server running main:app
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
