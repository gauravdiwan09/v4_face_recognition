import os
import shutil
from datetime import datetime
from app.core.config import settings

class DetectionService:
    @staticmethod
    async def save_detection(suspect_name: str, image_data: bytes, confidence: float):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{suspect_name}_{timestamp}.jpg"
            
            # Save to detected_faces directory
            file_path = os.path.join(settings.DETECTED_FACES_DIR, filename)
            with open(file_path, "wb") as f:
                f.write(image_data)
            
            return True
        except Exception as e:
            print(f"Error saving detection: {str(e)}")
            return False

    @staticmethod
    async def process_detections():
        try:
            if not os.path.exists(settings.PROCESSED_FACES_DIR):
                os.makedirs(settings.PROCESSED_FACES_DIR)
            
            for filename in os.listdir(settings.DETECTED_FACES_DIR):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(settings.DETECTED_FACES_DIR, filename)
                    dst_path = os.path.join(settings.PROCESSED_FACES_DIR, filename)
                    shutil.move(src_path, dst_path)
            
            return True
        except Exception as e:
            print(f"Error processing detections: {str(e)}")
            return False