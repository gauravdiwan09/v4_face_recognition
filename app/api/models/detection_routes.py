from fastapi import APIRouter, HTTPException
from typing import List
import os
from datetime import datetime
from app.core.config import settings
from app.api.models.detection import Detection, DetectionResponse

router = APIRouter()

@router.get("/detections/", response_model=List[DetectionResponse])
async def get_detections():
    try:
        detections = []
        processed_dir = settings.PROCESSED_FACES_DIR
        
        if os.path.exists(processed_dir):
            for filename in os.listdir(processed_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    # Parse filename for metadata
                    suspect_name = filename.split('_')[0]
                    timestamp_str = '_'.join(filename.split('_')[1:]).rstrip('.jpg')
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    
                    detections.append({
                        "id": len(detections) + 1,
                        "suspect_name": suspect_name,
                        "timestamp": timestamp,
                        "confidence": 0.0  # You might want to store this information
                    })
        
        return detections
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))