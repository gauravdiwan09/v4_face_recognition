from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SuspectBase(BaseModel):
    name: str
    description: Optional[str] = None

class SuspectCreate(SuspectBase):
    image_data: str  # Base64 encoded image

class SuspectResponse(SuspectBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True