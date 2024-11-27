import cv2
import numpy as np
from modules.face_detector import FaceDetector
from modules.face_encoder import FaceEncoder
from modules.suspect_manager import SuspectManager
import os
import time
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import threading
import asyncio
import shutil
from typing import Dict
from contextlib import asynccontextmanager
import torch
import torch.cuda
from queue import Queue
from threading import Thread

# Initialize FastAPI with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global recognition_running, recognition_thread, face_detector, face_encoder, suspect_manager
    
    print("\n=== Initializing Face Recognition System ===")
    
    # Initialize components
    face_detector = FaceDetector()
    face_encoder = FaceEncoder()
    suspect_manager = SuspectManager("suspects", face_encoder)
    
    # Start background tasks
    asyncio.create_task(process_detected_faces())
    
    # Start face recognition BEFORE loading suspects
    recognition_running = True
    recognition_thread = threading.Thread(target=run_face_recognition)
    recognition_thread.start()
    
    # Queue suspect loading AFTER webcam is running
    print("\nStarting suspect loading in background...")
    asyncio.create_task(load_suspects_async())
    
    print("\n=== Checking GPU Availability ===")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Configure CUDA for maximum performance
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
    else:
        print("No GPU detected, using CPU")
    
    yield
    
    # Shutdown
    recognition_running = False
    if recognition_thread:
        recognition_thread.join()

app = FastAPI(title="Face Recognition System", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
recognition_running = False
recognition_thread = None
face_detector = None
face_encoder = None
suspect_manager = None

async def load_suspects_async():
    """Separate async task for loading suspects"""
    await asyncio.sleep(2)  # Give webcam time to initialize
    suspect_manager.load_suspects()
    # Start watching for changes after initial load
    suspect_manager.start_directory_watching()

# Background task for processing detected faces
async def process_detected_faces():
    while True:
        try:
            detected_dir = "detected_faces"
            processed_dir = "processed_faces"
            
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
                
            for filename in os.listdir(detected_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(detected_dir, filename)
                    processed_path = os.path.join(processed_dir, filename)
                    
                    # Move file to processed directory
                    shutil.move(file_path, processed_path)
                    print(f"Processed and moved: {filename}")
                    
            await asyncio.sleep(30)  # Process every 30 seconds
        except Exception as e:
            print(f"Error in processing detected faces: {str(e)}")
            await asyncio.sleep(5)

def run_face_recognition():
    global recognition_running, face_detector, face_encoder, suspect_manager
    
    # Initialize video capture with RTSP stream
    rtsp_url = "rtsp://192.168.137.8:8080/h264.sdp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Optimize stream settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream: {rtsp_url}")
        return
    
    print(f"\nStarting video capture from: {rtsp_url}")
    
    # Create output directory
    output_dir = "detected_faces"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Variables for FPS and processing
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    last_detection_time = {}
    detection_cooldown = 1  # Reduced cooldown
    frame_count = 0
    process_every_n_frames = 2  # Process every 2nd frame
    
    # Create processing pools
    num_workers = 2
    frame_queues = [Queue(maxsize=1) for _ in range(num_workers)]
    result_queues = [Queue(maxsize=1) for _ in range(num_workers)]
    
    # Create CUDA streams for parallel processing
    if torch.cuda.is_available():
        cuda_streams = [torch.cuda.Stream() for _ in range(num_workers)]
        
    def process_frame_worker(worker_id):
        while recognition_running:
            try:
                frame = frame_queues[worker_id].get()
                if frame is None:
                    break
                
                if torch.cuda.is_available():
                    with torch.cuda.stream(cuda_streams[worker_id]):
                        # Detect faces
                        faces = face_detector.detect_faces(frame)
                        results = []
                        
                        for face in faces:
                            x, y, w, h = face['box']
                            face_img = frame[max(0, y-20):min(y+h+20, frame.shape[0]), 
                                          max(0, x-20):min(x+w+20, frame.shape[1])]
                            
                            if face_img.size == 0:
                                continue
                            
                            # Get embedding and identify
                            embedding = face_encoder.get_embedding(face_img)
                            if embedding is not None:
                                suspect_name, match_score = suspect_manager.identify_suspect(embedding)
                                results.append((face, suspect_name, match_score))
                        
                        # Put results in queue, dropping old results if necessary
                        if not result_queues[worker_id].empty():
                            result_queues[worker_id].get()
                        result_queues[worker_id].put((results, frame))
                        
            except Exception as e:
                print(f"Worker {worker_id} error: {str(e)}")
    
    # Start worker threads
    workers = [Thread(target=process_frame_worker, args=(i,)) for i in range(num_workers)]
    for worker in workers:
        worker.start()
    
    current_worker = 0
    while recognition_running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from stream.")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            continue
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Queue frame for processing
        if not frame_queues[current_worker].full():
            frame_queues[current_worker].put(frame.copy())
            current_worker = (current_worker + 1) % num_workers
        
        # Check for results from all workers
        display_frame = frame.copy()
        current_time = time.time()
        
        for i in range(num_workers):
            if not result_queues[i].empty():
                results, _ = result_queues[i].get()
                
                for face, suspect_name, match_score in results:
                    x, y, w, h = face['box']
                    
                    if suspect_name and match_score < 0.7:
                        if suspect_name not in last_detection_time or \
                           current_time - last_detection_time[suspect_name] >= detection_cooldown:
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            face_img = frame[max(0, y-20):min(y+h+20, frame.shape[0]), 
                                          max(0, x-20):min(x+w+20, frame.shape[1])]
                            output_path = os.path.join(output_dir, f"{suspect_name}_{timestamp}.jpg")
                            cv2.imwrite(output_path, face_img)
                            
                            last_detection_time[suspect_name] = current_time
                            print(f"\nSuspect Detected: {suspect_name}")
                            print(f"Match Score: {1-match_score:.2%}")
                            print(f"Image saved: {output_path}")
                        
                        color = (0, 0, 255)
                        label = f"{suspect_name} ({(1-match_score):.2%})"
                    else:
                        color = (255, 0, 0)
                        label = "Unknown"
                    
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(display_frame, label, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate and display FPS
        fps_counter += 1
        if current_time - fps_start_time > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = current_time
        
        cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition System', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    for q in frame_queues:
        q.put(None)
    for worker in workers:
        worker.join()
    cap.release()
    cv2.destroyAllWindows()

# API Routes
@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "recognition_active": recognition_running,
        "suspects_loaded": suspect_manager.get_suspect_count() if suspect_manager else 0
    }

@app.post("/reload-suspects")
async def reload_suspects():
    try:
        if suspect_manager:
            suspect_manager.load_suspects()
            return {"status": "success", "message": "Suspects queued for loading"}
        return {"status": "error", "message": "System not initialized"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/suspects")
async def get_suspects():
    try:
        if suspect_manager:
            suspects = suspect_manager.get_suspect_list()
            return {"status": "success", "suspects": suspects}
        return {"status": "error", "message": "System not initialized", "suspects": []}
    except Exception as e:
        return {"status": "error", "message": str(e), "suspects": []}

@app.get("/loading-status")
async def get_loading_status():
    try:
        if suspect_manager:
            queue_size = suspect_manager.load_queue.qsize()
            return {
                "status": "success",
                "queue_size": queue_size,
                "loaded_suspects": suspect_manager.get_suspect_count()
            }
        return {"status": "error", "message": "System not initialized"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)