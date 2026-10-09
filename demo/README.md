# AKIRA Local Web Demo

This is a local inference application demonstrating the frozen Multi-Task TCN model for the AKIRA project.

## Important Note
This application **does not** retrain the model, overwrite research artifacts, or make new clinical claims. It serves as a pure inference presentation layer wrapping the official `models/final/best_tcn.pth`.

## Setup Instructions

1. Ensure you are in the project root or the `demo/` folder.
2. Create and activate a virtual environment (if not already using the global project environment):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r demo/requirements.txt
   ```
4. Run the FastAPI backend:
   ```bash
   $env:PYTHONPATH="D:\me\project"
   uvicorn demo.backend.app:app --reload
   ```
5. Open your browser and navigate to:
   http://127.0.0.1:8000
