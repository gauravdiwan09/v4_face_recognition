from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import os
import base64
from app.core.config import settings
from app.api.models.suspect import SuspectCreate, SuspectResponse
import shutil

router = APIRouter()

@router.post("/suspects/", response_model=SuspectResponse)
async def create_suspect(suspect: SuspectCreate):
    try:
        # Decode base64 image
        image_data = base64.b64decode(suspect.image_data)
        
        # Create suspect directory if it doesn't exist
        suspect_dir = os.path.join(settings.SUSPECTS_DIR, suspect.name)
        os.makedirs(suspect_dir, exist_ok=True)
        
        # Save image
        image_path = os.path.join(suspect_dir, "face.jpg")
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        return {
            "id": 1,  # You might want to implement proper ID generation
            "name": suspect.name,
            "description": suspect.description,
            "created_at": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/suspects/", response_model=List[SuspectResponse])
async def get_suspects():
    try:
        suspects = []
        for suspect_name in os.listdir(settings.SUSPECTS_DIR):
            if os.path.isdir(os.path.join(settings.SUSPECTS_DIR, suspect_name)):
                suspects.append({
                    "id": len(suspects) + 1,
                    "name": suspect_name,
                    "description": None,
                    "created_at": datetime.now()  # You might want to get actual creation time
                })
        return suspects
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/suspects/{suspect_name}")
async def remove_suspect(suspect_name: str):
    """Remove a suspect by name"""
    try:
        # Get the suspect directory path
        suspect_dir = os.path.join("suspects", suspect_name)
        
        # Check if suspect exists
        if not os.path.exists(suspect_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Suspect '{suspect_name}' not found"
            )
        
        # Remove the suspect directory and all its contents
        shutil.rmtree(suspect_dir)
        
        # Force reload suspects to update the system
        # This assumes you have access to the suspect_manager
        # You might need to modify this based on your implementation
        if suspect_manager:
            suspect_manager.load_suspects()
        
        return {
            "message": f"Successfully removed suspect: {suspect_name}",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error removing suspect: {str(e)}"
        )