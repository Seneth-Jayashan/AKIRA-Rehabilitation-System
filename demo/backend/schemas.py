from pydantic import BaseModel
from typing import List, Dict, Optional

class PredictionResponse(BaseModel):
    elbow_prediction: List[float]
    forearm_prediction: List[float]
    exercise_prediction: str
    class_probabilities: Dict[str, float]
    rom_elbow: Optional[float]
    rom_forearm: Optional[float]
    repetition_count: Optional[int]
    velocity_elbow: Optional[float]
    
    # Ground truth (optional, if demo sample used)
    gt_elbow: Optional[List[float]] = None
    gt_forearm: Optional[List[float]] = None
    gt_exercise: Optional[str] = None
