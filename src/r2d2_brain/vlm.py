import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json
import base64
import io
import argparse
import cv2
import numpy as np
from qwen_vl_utils import process_vision_info
from flask import Flask, request, jsonify
import time
import hashlib

# --- Global Variables ---
start_time = time.time()
result_cache = {}
MAX_CACHE_SIZE = 10

# --- The Enhanced Prompt for Structured JSON Output ---
JSON_PROMPT = """You are an expert AI vision system for an autonomous robot. Analyze this image and provide a structured JSON with:

1. "objects": A list of detected items. For each object include:
   - "name": Brief identifier (e.g., "person", "chair")
   - "description": Brief physical description (e.g., "tall person wearing red")
   - "box_2d": Estimated pixel coordinates [x_min, y_min, x_max, y_max]
   - "importance": Integer from 1-5 indicating relevance to a navigating robot (5=highest)

2. "scene_inference": One sentence summarizing what the robot is seeing

3. "navigation_advice": Brief suggestion about how the robot should navigate this scene

Output ONLY valid JSON with these fields and no other text.
"""

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global Model Variables ---
model = None
processor = None
device = "cuda"

def load_model_and_processor():
    """Load the Qwen model and processor once at startup."""
    global model, processor
    if model is None:
        print("Loading Qwen2-VL-7B-Instruct model and processor...")
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        print("Model and processor loaded successfully.")

@app.route('/status', methods=['GET'])
def status():
    """Return VLM server status information."""
    return jsonify({
        'status': 'running',
        'model': 'Qwen2-VL-7B-Instruct',
        'uptime_seconds': int(time.time() - start_time),
        'cache_size': len(result_cache),
        'max_cache_size': MAX_CACHE_SIZE
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """API endpoint that receives, decompresses, and analyzes an image frame."""
    global result_cache
    
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # 1. Decode the base64 string to get the raw JPEG bytes
        image_bytes = base64.b64decode(request.json['image'])
        
        # Generate image hash for caching
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Check cache for this image
        if image_hash in result_cache:
            print(f"Cache hit for image hash {image_hash}")
            return jsonify(result_cache[image_hash])
        
        # 2. Convert raw bytes to a NumPy array for OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # 3. Decode the image using OpenCV
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            return jsonify({'error': 'Failed to decode compressed image on server'}), 400
            
        # 4. Convert from OpenCV's BGR format to PIL's RGB format for the model
        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        # --- Model Inference ---
        processing_start = time.time()
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": JSON_PROMPT}]}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # --- Parse and Return JSON ---
        json_start = output_text.find('{')
        json_end = output_text.rfind('}') + 1
        processing_time = time.time() - processing_start
        
        if json_start != -1 and json_end != -1:
            json_string = output_text[json_start:json_end]
            result = json.loads(json_string)
            
            # Add processing metadata
            result['processing_time'] = processing_time
            
            # Store in cache
            result_cache[image_hash] = result
            
            # Maintain cache size
            if len(result_cache) > MAX_CACHE_SIZE:
                oldest_key = next(iter(result_cache))
                result_cache.pop(oldest_key)
            
            print(f"Processed frame in {processing_time:.2f}s: {result.get('scene_inference', '')}")
            return jsonify(result)
        else:
            error_response = {
                'error': 'Model did not return a valid JSON object', 
                'raw_output': output_text,
                'processing_time': processing_time
            }
            return jsonify(error_response), 500

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    load_model_and_processor()
    app.run(host=args.host, port=args.port)