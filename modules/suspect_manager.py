import os
import cv2
import numpy as np
from threading import Thread, Lock, Timer
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import threading
from queue import Queue
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor

class SuspectManager:
    def __init__(self, suspects_dir, face_encoder):
        self.suspects_dir = suspects_dir
        self.face_encoder = face_encoder
        self.suspects = {}
        self.lock = threading.Lock()
        self.load_queue = Queue()
        self.is_running = True
        self.observer = None
        
        # Cache directory setup
        self.cache_dir = os.path.join(suspects_dir, '.cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Load existing cache
        self.processed_files = self._load_cache()
        
        # Initialize thread pool
        self.max_workers = min(4, os.cpu_count() or 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start background loading thread
        self.loading_thread = Thread(target=self._background_loader, daemon=True)
        self.loading_thread.start()

    def _get_file_hash(self, file_path):
        """Calculate MD5 hash of file"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # Read in 64kb chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _load_cache(self):
        """Load processed files cache"""
        cache_file = os.path.join(self.cache_dir, 'processed_files.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save processed files cache"""
        cache_file = os.path.join(self.cache_dir, 'processed_files.json')
        with open(cache_file, 'w') as f:
            json.dump(self.processed_files, f)

    def _background_loader(self):
        while self.is_running:
            try:
                # Wait for load requests
                suspect_path = self.load_queue.get()
                if suspect_path == "STOP":
                    break
                    
                # Process single suspect
                self._process_suspect(suspect_path)
                
            except Exception as e:
                print(f"Error in background loader: {str(e)}")
            finally:
                self.load_queue.task_done()

    def queue_suspect_load(self, suspect_path):
        """Queue a suspect for background loading"""
        self.load_queue.put(suspect_path)

    def load_suspects(self):
        """Queue all suspects for loading"""
        if os.path.exists(self.suspects_dir):
            print("\nQueuing suspects for background loading...")
            suspect_count = 0
            for suspect in os.listdir(self.suspects_dir):
                if suspect != '.cache':  # Skip cache directory
                    suspect_path = os.path.join(self.suspects_dir, suspect)
                    if os.path.isdir(suspect_path):
                        self.queue_suspect_load(suspect_path)
                        suspect_count += 1
            print(f"Queued {suspect_count} suspects for loading")

    def _process_image(self, img_path, img_file, suspect_name):
        """Process a single image and return its embedding"""
        try:
            file_hash = self._get_file_hash(img_path)
            
            # Check cache first
            if suspect_name in self.processed_files and file_hash in self.processed_files[suspect_name]:
                return np.array(self.processed_files[suspect_name][file_hash]), file_hash, True
            
            # Process new image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_file}")
                return None, file_hash, False
                
            # Resize image if too large (optimize memory usage)
            max_dimension = 1024
            h, w = image.shape[:2]
            if h > max_dimension or w > max_dimension:
                scale = max_dimension / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            embedding = self.face_encoder.get_embedding(image)
            return embedding, file_hash, False
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            return None, None, False

    def _process_suspect(self, suspect_path):
        """Process a single suspect directory"""
        start_time = time.time()
        suspect_name = os.path.basename(suspect_path)
        if suspect_name.startswith('.'):
            return
            
        print(f"\nProcessing suspect: {suspect_name}")
        
        # Collect all images
        image_files = [f for f in os.listdir(suspect_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found for {suspect_name}")
            return
            
        print(f"Found {len(image_files)} images for {suspect_name}")
        
        # Process images in parallel
        embeddings = []
        new_files_processed = False
        futures = []
        
        for img_file in image_files:
            img_path = os.path.join(suspect_path, img_file)
            futures.append(self.thread_pool.submit(
                self._process_image, img_path, img_file, suspect_name
            ))
        
        # Collect results
        for future in futures:
            embedding, file_hash, from_cache = future.result()
            if embedding is not None:
                embeddings.append(embedding)
                if not from_cache:
                    new_files_processed = True
                    if suspect_name not in self.processed_files:
                        self.processed_files[suspect_name] = {}
                    self.processed_files[suspect_name][file_hash] = embedding.tolist()
        
        # Update suspect database
        if embeddings:
            with self.lock:
                self.suspects[suspect_name] = embeddings
                if new_files_processed:
                    self._save_cache()
            print(f"Successfully processed {len(embeddings)}/{len(image_files)} images for {suspect_name}")
        else:
            print(f"No valid face embeddings found for {suspect_name}")
            
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time for {suspect_name}: {processing_time:.2f} seconds")

    def remove_suspect(self, suspect_name):
        """Remove a suspect from the database"""
        with self.lock:
            if suspect_name in self.suspects:
                del self.suspects[suspect_name]
                if suspect_name in self.processed_files:
                    del self.processed_files[suspect_name]
                    self._save_cache()
                return True
        return False

    def identify_suspect(self, face_embedding, threshold=0.7):
        """Identify a suspect based on face embedding"""
        with self.lock:
            if not self.suspects:
                return None, 0.0
                
            best_match = None
            best_score = float('inf')
            
            for suspect, embeddings in self.suspects.items():
                similarities = np.dot(embeddings, face_embedding)
                norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(face_embedding)
                cosine_similarities = similarities / norms
                
                max_similarity = np.max(cosine_similarities)
                avg_similarity = np.mean(cosine_similarities)
                
                score = 1 - (0.7 * max_similarity + 0.3 * avg_similarity)
                
                if score < best_score and score < threshold:
                    best_score = score
                    best_match = suspect
                    
            return best_match, best_score

    def get_suspect_count(self):
        """Return the number of loaded suspects"""
        return len(self.suspects)

    def get_suspect_list(self):
        """Return list of suspect names"""
        return list(self.suspects.keys())

    def start_directory_watching(self):
        """Start watching the suspects directory for changes"""
        if not self.observer:
            self.observer = Observer()
            event_handler = SuspectDirectoryHandler(self)
            self.observer.schedule(event_handler, self.suspects_dir, recursive=True)
            self.observer.start()
            print("\nStarted watching suspects directory for changes")

    def stop_directory_watching(self):
        """Stop watching the suspects directory"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.is_running = False
        self.load_queue.put("STOP")
        if hasattr(self, 'loading_thread'):
            self.loading_thread.join()
        self.stop_directory_watching()
        self.thread_pool.shutdown()

class SuspectDirectoryHandler(FileSystemEventHandler):
    def __init__(self, suspect_manager):
        self.suspect_manager = suspect_manager
        self.reload_timer = None
        self.pending_changes = set()  # Track pending changes
        
    def on_any_event(self, event):
        if event.is_directory:
            return
            
        if event.event_type in ['created', 'modified', 'deleted']:
            path_parts = event.src_path.split(os.path.sep)
            if 'suspects' in path_parts:
                suspect_idx = path_parts.index('suspects') + 1
                if suspect_idx < len(path_parts):
                    suspect_name = path_parts[suspect_idx]
                    
                    if event.event_type == 'deleted':
                        suspect_path = os.path.join(self.suspect_manager.suspects_dir, suspect_name)
                        if not os.path.exists(suspect_path) or not any(
                            f.endswith(('.jpg', '.jpeg', '.png')) 
                            for f in os.listdir(suspect_path)
                        ):
                            print(f"\nRemoving suspect: {suspect_name}")
                            self.suspect_manager.remove_suspect(suspect_name)
                    else:
                        self.pending_changes.add(suspect_name)
                        
                        # Cancel existing timer if any
                        if self.reload_timer:
                            self.reload_timer.cancel()
                        
                        # Create new timer for batch processing
                        self.reload_timer = Timer(2.0, self._process_pending_changes)
                        self.reload_timer.start()
    
    def _process_pending_changes(self):
        """Process all pending changes at once"""
        try:
            if self.pending_changes:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Processing changes for {len(self.pending_changes)} suspects")
                
                for suspect_name in self.pending_changes:
                    suspect_path = os.path.join(self.suspect_manager.suspects_dir, suspect_name)
                    if os.path.exists(suspect_path):
                        self.suspect_manager.queue_suspect_load(suspect_path)
                
                self.pending_changes.clear()
        except Exception as e:
            print(f"Error during batch processing: {str(e)}")