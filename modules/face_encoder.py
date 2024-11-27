import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
import onnxruntime

class FaceEncoder:
    def __init__(self):
        if torch.cuda.is_available():
            # Configure CUDA for better performance
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            print(f"Face Encoder using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Configure providers with optimized CUDA settings
        providers = []
        if torch.cuda.is_available():
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB GPU memory limit
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
                'cudnn_conv_use_max_workspace': True,
            }))
        providers.append('CPUExecutionProvider')
        
        # Initialize InsightFace with optimized settings
        self.app = FaceAnalysis(
            providers=providers,
            allowed_modules=['detection', 'recognition'],
            max_batch_size=4  # Increase batch processing
        )
        
        # Prepare with optimized settings
        self.app.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1,
            det_size=(640, 640),
            det_thresh=0.5
        )
        
    def get_embedding(self, face_image):
        """Get face embeddings for multiple faces"""
        try:
            if face_image is None:
                return None
            
            # Process image regardless of GPU availability
            # Pre-process and pad image
            padding = 40
            h, w = face_image.shape[:2]
            padded_image = cv2.copyMakeBorder(
                face_image,
                padding, padding, padding, padding,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            
            # Optimize image size
            min_size = 320
            if h < min_size or w < min_size:
                scale = min_size / min(h, w)
                new_size = (int(w * scale), int(h * scale))
                padded_image = cv2.resize(padded_image, new_size, interpolation=cv2.INTER_CUBIC)
            
            # Pre-process image
            face_image = self.preprocess_image(padded_image)
            
            # Use GPU acceleration if available
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    with torch.cuda.stream(torch.cuda.Stream()):
                        faces = self.app.get(face_image)
                        if not faces:
                            return None
                        
                        embedding = faces[0].embedding
                        torch.cuda.current_stream().synchronize()
                        return embedding
            else:
                # Direct CPU processing
                faces = self.app.get(face_image)
                if not faces:
                    return None
                return faces[0].embedding
                
        except Exception as e:
            print(f"Error in face encoding: {str(e)}")
            return None

    def preprocess_image(self, image):
        """
        Preprocess image to improve quality
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image