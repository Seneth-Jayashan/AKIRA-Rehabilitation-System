"""
Launcher for AKIRA Edge-AI Knee Rehabilitation Web Interface
Automatically checks dependencies, launches the FastAPI server, and opens the browser.
"""

import os
import sys
import time
import webbrowser
import threading
import uvicorn

def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n[OK] Opening web browser at {url} ...")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 70)
    print("AKIRA Edge-AI Knee Rehabilitation Dashboard")
    print("Initializing server at http://127.0.0.1:8000 ...")
    print("Press CTRL+C anytime in this window to stop the server.")
    print("=" * 70)

    # Launch browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn server from the api/ folder (it also serves web/)
    api_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api")
    sys.path.insert(0, api_dir)
    os.chdir(api_dir)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
