import os
import shutil
from app.core.config import settings

class SuspectService:
    @staticmethod
    async def add_suspect(name: str, image_path: str, description: str = None):
        try:
            suspect_dir = os.path.join(settings.SUSPECTS_DIR, name)
            os.makedirs(suspect_dir, exist_ok=True)
            
            # Copy image to suspect directory
            dest_path = os.path.join(suspect_dir, "face.jpg")
            shutil.copy2(image_path, dest_path)
            
            return True
        except Exception as e:
            print(f"Error adding suspect: {str(e)}")
            return False

    @staticmethod
    async def remove_suspect(name: str):
        try:
            suspect_dir = os.path.join(settings.SUSPECTS_DIR, name)
            if os.path.exists(suspect_dir):
                shutil.rmtree(suspect_dir)
            return True
        except Exception as e:
            print(f"Error removing suspect: {str(e)}")
            return False