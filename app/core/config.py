from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Face Recognition System"
    
    # Paths
    SUSPECTS_DIR: str = "suspects"
    DETECTED_FACES_DIR: str = "detected_faces"
    PROCESSED_FACES_DIR: str = "processed_faces"
    
    # Intervals (in seconds)
    SUSPECT_UPDATE_INTERVAL: int = 30
    DETECTION_PROCESS_INTERVAL: int = 30
    
    # Database settings (if you want to add database later)
    DATABASE_URL: str = "sqlite:///./face_recognition.db"

settings = Settings()