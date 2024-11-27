import asyncio
from datetime import datetime
from app.core.config import settings

class SuspectUpdateManager:
    def __init__(self, suspect_manager, update_interval=settings.SUSPECT_UPDATE_INTERVAL):
        self.suspect_manager = suspect_manager
        self.update_interval = update_interval
        self.is_running = False
        self.task = None

    async def update_loop(self):
        """Periodically checks and reloads suspects"""
        while self.is_running:
            try:
                print(f"\n[{datetime.now()}] Checking for suspect updates...")
                # Reload suspects from the suspects directory
                self.suspect_manager.load_suspects()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in suspect update loop: {str(e)}")
                await asyncio.sleep(5)

    def start(self):
        """Start the background task"""
        self.is_running = True
        self.task = asyncio.create_task(self.update_loop())

    def stop(self):
        """Stop the background task"""
        self.is_running = False
        if self.task:
            self.task.cancel()