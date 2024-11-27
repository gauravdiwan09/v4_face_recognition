from facenet_pytorch import MTCNN
import torch
import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue

class FaceDetector:
    def __init__(self):
        # Configure CUDA settings for better performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Face Detector using device: {self.device}")
        
        # Configure MTCNN for better GPU utilization
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.85],
            selection_method='probability',
            post_process=True
        )
        
        # Enable GPU memory caching
        if torch.cuda.is_available():
            self.detector = self.detector.to(self.device)
            torch.cuda.empty_cache()
            
            # Set optimal parameters for GPU processing
            self.batch_size = 4  # We'll handle batching manually
            self.margin = 40  # Increased margin for better face detection
        else:
            self.batch_size = 1
            self.margin = 20
        
        # Increase queue sizes for better throughput
        self.process_queue = Queue(maxsize=self.batch_size)
        self.result_queue = Queue(maxsize=self.batch_size)
        self.is_running = False
        self.lock = Lock()
        
        self.start_processing_thread()
    
    def start_processing_thread(self):
        """Start the background processing thread"""
        self.is_running = True
        self.processing_thread = Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
    
    def _process_frames(self):
        """Background thread for face detection"""
        while self.is_running:
            try:
                if not self.process_queue.empty():
                    frame = self.process_queue.get()
                    # Process the frame
                    enhanced_frame = self.enhance_image(frame)
                    faces = self._detect_faces(enhanced_frame)
                    # Put results in queue, overwriting old results if necessary
                    if not self.result_queue.empty():
                        self.result_queue.get()
                    self.result_queue.put((faces, frame))
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
    
    def detect_faces(self, image):
        """
        Queue frame for processing and return most recent results
        """
        # Queue new frame for processing, dropping old frame if necessary
        if not self.process_queue.full():
            self.process_queue.put(image)
        
        # Return most recent results if available
        if not self.result_queue.empty():
            faces, _ = self.result_queue.get()
            return faces
        return []
    
    def _detect_faces(self, image):
        """Internal method for face detection"""
        try:
            if torch.cuda.is_available():
                # Process multiple frames in parallel
                with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Optimize image tensor creation
                    image_tensor = torch.from_numpy(image).to(self.device, non_blocking=True).float()
                    if image_tensor.ndimension() == 3:
                        image_tensor = image_tensor.unsqueeze(0)
                    
                    # Batch process detection
                    with torch.no_grad():
                        boxes, probs = self.detector.detect(image)
                    
                    if boxes is None:
                        return []
                    
                    # Process results in parallel
                    faces = []
                    for box, prob in zip(boxes, probs):
                        if prob >= 0.85:
                            x1, y1, x2, y2 = box
                            w = x2 - x1
                            h = y2 - y1
                            
                            # Add margin to face box
                            x1 = max(0, x1 - self.margin)
                            y1 = max(0, y1 - self.margin)
                            x2 = min(image.shape[1], x2 + self.margin)
                            y2 = min(image.shape[0], y2 + self.margin)
                            
                            w = x2 - x1
                            h = y2 - y1
                            
                            faces.append({
                                'box': [int(x1), int(y1), int(w), int(h)],
                                'confidence': float(prob)
                            })
                    
                    return faces
            else:
                # CPU fallback
                return self._detect_faces_cpu(image)
            
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return []

    def _detect_faces_cpu(self, image):
        """Fallback method for CPU processing"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes, probs = self.detector.detect(image)
        
        if boxes is None:
            return []
        
        faces = []
        for box, prob in zip(boxes, probs):
            if prob >= 0.85:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                faces.append({
                    'box': [int(x1), int(y1), int(w), int(h)],
                    'confidence': float(prob)
                })
        
        return faces

    def enhance_image(self, image):
        """Enhanced image preprocessing"""
        # Apply CLAHE for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Denoise while preserving edges
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()