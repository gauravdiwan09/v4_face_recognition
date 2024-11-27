from pydantic import BaseModel
from datetime import datetime

class Detection(BaseModel):
    suspect_name: str
    timestamp: datetime
    confidence: float
    image_path: str

class DetectionResponse(BaseModel):
    id: int
    suspect_name: str
    timestamp: datetime
    confidence: float
    
    class Config:
        from_attributes = True