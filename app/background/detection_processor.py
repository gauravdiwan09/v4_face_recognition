import asyncio
import os
from datetime import datetime
from app.core.config import settings
from app.services.detection_service import DetectionService

class DetectionProcessor:
    def __init__(self, process_interval=settings.DETECTION_PROCESS_INTERVAL):
        self.process_interval = process_interval
        self.is_running = False
        self.task = None
        self.detection_service = DetectionService()

    async def process_loop(self):
        """Periodically processes detected faces"""
        while self.is_running:
            try:
                print(f"\n[{datetime.now()}] Processing detected faces...")
                await self.detection_service.process_detections()
                await asyncio.sleep(self.process_interval)
            except Exception as e:
                print(f"Error in detection process loop: {str(e)}")
                await asyncio.sleep(5)

    def start(self):
        """Start the background task"""
        self.is_running = True
        self.task = asyncio.create_task(self.process_loop())

    def stop(self):
        """Stop the background task"""
        self.is_running = False
        if self.task:
            self.task.cancel()