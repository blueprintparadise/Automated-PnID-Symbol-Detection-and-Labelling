#!/usr/bin/env python3
"""
P&ID Symbol Detection Gradio App v2
Enhanced with comprehensive error logging and port 8080
"""

import gradio as gr
import cv2
import numpy as np
import os
import sys
import logging
import traceback
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import math
import time
import tempfile
import shutil
import atexit
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradio_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define the FCN model class
class FCN(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(FCN, self).__init__()
        logger.info("Initializing FCN model...")
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(20736, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(128)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(64)
        )
        self.fc4 = nn.Linear(64, 7)

    def forward(self, x):
        try:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = F.softmax(self.fc4(x), dim=1)
            return x
        except Exception as e:
            logger.error(f"Error in FCN forward pass: {e}")
            logger.error(traceback.format_exc())
            raise

# Global variables
model = None
device = None
data_transform = None

def load_model():
    """Load the FCN model with comprehensive error handling"""
    global model, device, data_transform
    
    try:
        if model is not None:
            logger.info("Model already loaded, returning existing model")
            return model
        
        logger.info("Starting model loading process...")
        model = FCN()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Setting model up on {device}")
        
        model_location = "./main_driver/model_Inverted"
        
        if not os.path.exists(model_location):
            error_msg = f"Model file not found at {model_location}"
            logger.error(error_msg)
            return None
        
        logger.info(f"Loading model from {model_location}")
        
        if torch.cuda.is_available():
            model.cuda()
            model.load_state_dict(torch.load(model_location))
            logger.info("Model loaded on GPU")
        else:
            model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))
            logger.info("Model loaded on CPU")
        
        data_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(np.expand_dims(np.array(x), axis=0)).float())
        ])
        
        model.eval()
        logger.info("Model loaded successfully and set to evaluation mode!")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return None

def recolor_crop(crop):
    """Convert image to black and white with error handling"""
    try:
        result = crop.copy()
        for i in range(len(result)):
            for j in range(len(result[0])):
                if result[i][j] >= 120:
                    result[i][j] = 225
                else:
                    result[i][j] = 0
        return result
    except Exception as e:
        logger.error(f"Error in recolor_crop: {e}")
        raise

def invert(x):
    """Invert colors for model processing with error handling"""
    try:
        result = x.copy()
        for i in range(len(result)):
            for j in range(len(result[0])):
                if result[i][j] > 200:
                    result[i][j] = 0
                else:
                    result[i][j] = 1
        return result
    except Exception as e:
        logger.error(f"Error in invert function: {e}")
        raise

def get_distance(x, y):
    """Calculate distance between two centroids"""
    try:
        xmid1, ymid1 = x[0], x[1]
        xmid2, ymid2 = y[0], y[1]
        return math.sqrt((ymid2 - ymid1)**2 + (xmid2 - xmid1)**2)
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return float('inf')

def process_pnid_image(image_path, x_start=400, x_end=5600, y_start=250, y_end=5000):
    """Main processing function with comprehensive error handling"""
    
    try:
        logger.info(f"Starting processing of image: {image_path}")
        
        # Load model
        model = load_model()
        if model is None:
            logger.error("Failed to load model")
            return None, None
        
        # Load image
        logger.info("Loading image...")
        img = cv2.imread(image_path, 0)
        if img is None:
            error_msg = f"Could not load image from {image_path}"
            logger.error(error_msg)
            return None, None
        
        logger.info(f"Image loaded successfully. Size: {img.shape}")
        
        # Get ROI
        m, n = img.shape
        x_end = min(x_end, n)
        y_end = min(y_end, m)
        
        logger.info(f"Processing ROI: x={x_start}-{x_end}, y={y_start}-{y_end}")
        
        ready_img = img[y_start:y_end, x_start:x_end]
        
        # Recolor image
        logger.info("Recoloring image...")
        colorized = recolor_crop(ready_img)
        main_img = colorized.copy()
        
        # Get dimensions of ROI
        roi_m, roi_n = ready_img.shape
        logger.info(f"ROI dimensions: {roi_m} x {roi_n}")
        
        # Object detection loop
        objects_info = {}
        object_id = 0
        
        logger.info("Starting object detection...")
        total_windows = len(range(0, roi_m-150, 75)) * len(range(0, roi_n-150, 75))
        logger.info(f"Processing {total_windows} windows...")
        
        for i in tqdm(range(0, roi_m-150, 75), desc="Processing image"):
            for j in range(0, roi_n-150, 75):
                try:
                    x_min = j
                    x_max = j + 150
                    y_min = i
                    y_max = i + 150
                    
                    bounding_box = [x_min, x_max, y_min, y_max]
                    xmid = x_min + ((x_max - x_min) // 2)
                    ymid = y_min + ((y_max - y_min) // 2)
                    centroid = [xmid, ymid]
                    
                    temp = main_img[y_min:y_max, x_min:x_max]
                    window = temp.copy()
                    
                    black = np.count_nonzero(window == 0)
                    total = 150 * 150
                    percentage_black = (black / total) * 100
                    
                    if percentage_black > 10:
                        window = invert(window)
                        im_pil = Image.fromarray(window)
                        im = data_transform(im_pil)
                        im_a = im.numpy()
                        im_a = np.expand_dims(im_a, axis=0)
                        t = torch.tensor(im_a)
                        
                        with torch.no_grad():
                            t = t.to(device)
                            score = model(t)
                            _, predictions = torch.max(score, 1)
                            class_id = predictions.item()
                            
                            if class_id != 6:  # Not background
                                temp_dict = {
                                    "class_id": int(class_id),
                                    "bounding_box": bounding_box,
                                    "centroid": centroid
                                }
                                objects_info[object_id] = temp_dict
                                object_id += 1
                                
                except Exception as e:
                    logger.warning(f"Error processing window at ({i}, {j}): {e}")
                    continue
        
        logger.info(f"Found {len(objects_info)} potential objects")
        
        if len(objects_info) == 0:
            logger.warning("No objects detected!")
            return colorized, []
        
        # Group similar objects
        logger.info("Grouping similar objects...")
        list_of_objects = list(objects_info.keys())
        groups = []
        
        for i in range(len(objects_info)):
            try:
                curr_object = objects_info[list_of_objects[i]]
                curr_centroid = curr_object['centroid']
                curr_class = curr_object['class_id']
                
                temp_grp = [i]
                
                for j in range(len(objects_info)):
                    compare_object = objects_info[list_of_objects[j]]
                    compare_centroid = compare_object['centroid']
                    compare_class = compare_object['class_id']
                    
                    if get_distance(curr_centroid, compare_centroid) < 80 and curr_class == compare_class:
                        temp_grp.append(j)
                
                temp_grp.sort()
                groups.append(temp_grp)
                
            except Exception as e:
                logger.warning(f"Error grouping object {i}: {e}")
                continue
        
        # Process groups
        logger.info("Processing groups...")
        groups_dict = {}
        been_done = {}
        
        for i in groups:
            try:
                temp = i
                if temp[0] not in been_done.keys():
                    been_done[temp[0]] = temp[0]
                
                main_key = been_done[temp[0]]
                
                if main_key not in groups_dict.keys():
                    groups_dict[main_key] = []
                
                for j in temp:
                    been_done[j] = main_key
                
                groups_dict[main_key].extend(temp)
                groups_dict[main_key] = list(set(groups_dict[main_key]))
                groups_dict[main_key].sort()
                
            except Exception as e:
                logger.warning(f"Error processing group: {e}")
                continue
        
        # Create final bounding boxes
        logger.info("Creating final bounding boxes...")
        final_info = {}
        draw_img = colorized.copy()
        
        for i in groups_dict.keys():
            try:
                object_id = i
                
                x_min_list = []
                y_min_list = []
                x_max_list = []
                y_max_list = []
                
                for j in groups_dict[i]:
                    xmin, xmax, ymin, ymax = objects_info[j]['bounding_box']
                    x_min_list.append(xmin)
                    x_max_list.append(xmax)
                    y_min_list.append(ymin)
                    y_max_list.append(ymax)
                
                xmin = min(x_min_list)
                ymin = min(y_min_list)
                xmax = max(x_max_list)
                ymax = max(y_max_list)
                
                xmid = xmin + ((xmax - xmin) // 2)
                ymid = ymin + ((ymax - ymin) // 2)
                
                xmin = xmid - 80
                ymin = ymid - 80
                xmax = xmid + 80
                ymax = ymid + 80
                
                # Ensure bounding box is within image bounds
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(roi_n, xmax)
                ymax = min(roi_m, ymax)
                
                class_id = objects_info[i]['class_id']
                
                temp = {
                    "class_id": class_id,
                    "bbox": [xmin, xmax, ymin, ymax],
                    "centroid": [xmid, ymid]
                }
                
                # Draw bounding box
                draw_img = cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.putText(draw_img, str(i), (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                final_info[i] = temp
                
            except Exception as e:
                logger.warning(f"Error creating bounding box for object {i}: {e}")
                continue
        
        logger.info(f"Final detection: {len(final_info)} objects")
        
        # Create CSV data
        component_names = {
            0: "Valve",
            1: "Control Valve", 
            2: "Circular Component",
            3: "Spectacle Blind",
            4: "Inline Mixer/Filter",
            5: "Instrument"
        }
        
        csv_data = []
        for obj_id, obj_info in final_info.items():
            try:
                class_id = obj_info['class_id']
                bbox = obj_info['bbox']
                component_name = component_names.get(class_id, "Unknown")
                
                # Generate meaningful item labels based on component type
                item_label_prefixes = {
                    0: "V",      # Valve -> V001, V002, etc.
                    1: "CV",     # Control Valve -> CV001, CV002, etc. 
                    2: "CC",     # Circular Component -> CC001, CC002, etc.
                    3: "SB",     # Spectacle Blind -> SB001, SB002, etc.
                    4: "MF",     # Inline Mixer/Filter -> MF001, MF002, etc.
                    5: "I"       # Instrument -> I001, I002, etc.
                }
                
                prefix = item_label_prefixes.get(class_id, "UNK")
                item_label = f"{prefix}{obj_id:03d}"  # Format as 3-digit number (001, 002, etc.)
                
                csv_data.append({
                    "Object-ID": obj_id,
                    "Class-ID": class_id,
                    "Component Name": component_name,
                    "Item Label": item_label,
                    "Location (xmin,xmax,ymin,ymax)": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                })
            except Exception as e:
                logger.warning(f"Error creating CSV entry for object {obj_id}: {e}")
                continue
        
        logger.info("Processing completed successfully")
        return draw_img, csv_data
        
    except Exception as e:
        logger.error(f"Critical error in process_pnid_image: {e}")
        logger.error(traceback.format_exc())
        return None, None

def gradio_process_image(image):
    """Gradio interface function with enhanced error handling"""
    tmp_file_path = None
    csv_path = None
    
    try:
        logger.info("Processing image through Gradio interface...")
        
        if image is None:
            logger.error("No image provided")
            return create_error_image("No image provided"), None
        
        # Save uploaded image temporarily with proper cleanup
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tmp_file_path = tmp_file.name
        tmp_file.close()
        
        logger.info(f"Saving temporary image to {tmp_file_path}")
        image.save(tmp_file_path)
        
        # Process the image
        start_time = time.time()
        result_img, csv_data = process_pnid_image(tmp_file_path)
        end_time = time.time()
        
        logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
        
        if result_img is None:
            logger.error("Image processing failed")
            return create_error_image("Image processing failed"), None
        
        # Convert result image to RGB for display
        if len(result_img.shape) == 2:  # Grayscale
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
        else:  # Already RGB
            result_img_rgb = result_img
        
        # Create CSV file with proper handling
        if csv_data:
            csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', newline='')
            csv_path = csv_file.name
            csv_file.close()
            
            logger.info(f"Saving CSV to {csv_path}")
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"CSV saved with {len(csv_data)} objects")
        
        return result_img_rgb, csv_path
        
    except Exception as e:
        logger.error(f"Error in gradio_process_image: {e}")
        logger.error(traceback.format_exc())
        return create_error_image(f"Error: {str(e)[:50]}..."), None
        
    finally:
        # Clean up temporary image file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                logger.info(f"Cleaned up temporary file: {tmp_file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary file {tmp_file_path}: {e}")

def create_error_image(error_message):
    """Create an error image with the error message"""
    try:
        error_img = np.zeros((400, 800, 3), dtype=np.uint8)
        # Split long messages into multiple lines
        words = error_message.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) < 50:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        y_start = 150
        for i, line in enumerate(lines[:5]):  # Max 5 lines
            cv2.putText(error_img, line, (10, y_start + i*40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return error_img
    except Exception as e:
        logger.error(f"Error creating error image: {e}")
        # Return a simple red image if even error image creation fails
        return np.full((400, 800, 3), [0, 0, 255], dtype=np.uint8)

def create_interface():
    """Create the Gradio interface"""
    try:
        logger.info("Creating Gradio interface...")
        
        with gr.Blocks(
            title="P&ID Symbol Detection v2",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            """
        ) as demo:
            
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1>🔧 P&ID Symbol Detection and Labelling v2</h1>
                <p>Upload a P&ID diagram to detect and classify symbols automatically</p>
                <p><em>Enhanced with comprehensive error logging • Port 8080</em></p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📤 Upload P&ID Image", 
                        type="pil",
                        height=400
                    )
                    
                    gr.Examples(
                        examples=["./main_driver/3.jpg"],
                        inputs=image_input,
                        label="📋 Try Sample P&ID Image"
                    )
                    
                    process_btn = gr.Button(
                        "🚀 Process P&ID", 
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.HTML("""
                    <div style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
                        <h4 style="color: #2c3e50 !important; margin-bottom: 10px;">📊 Detectable Symbol Types:</h4>
                        <ul style="margin: 0; padding-left: 20px; list-style-type: disc;">
                            <li style="color: #333333 !important; margin-bottom: 5px;">Valves</li>
                            <li style="color: #333333 !important; margin-bottom: 5px;">Control Valves</li>
                            <li style="color: #333333 !important; margin-bottom: 5px;">Circular Components</li>
                            <li style="color: #333333 !important; margin-bottom: 5px;">Spectacle Blinds</li>
                            <li style="color: #333333 !important; margin-bottom: 5px;">Inline Mixers/Filters</li>
                            <li style="color: #333333 !important; margin-bottom: 5px;">Instruments</li>
                        </ul>
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    image_output = gr.Image(
                        label="📋 Detection Results", 
                        height=400
                    )
                    
                    csv_output = gr.File(
                        label="💾 Download Results (CSV)",
                        visible=True
                    )
                    
                    gr.HTML("""
                    <div style="margin-top: 10px; padding: 10px; background-color: #e8f5e8; border-radius: 5px; border: 1px solid #c3e6c3;">
                        <h4 style="color: #2c3e50 !important; margin-bottom: 10px;">✅ Output Information:</h4>
                        <p style="color: #333333 !important; margin-bottom: 8px;"><strong style="color: #2c3e50 !important;">Image:</strong> Shows bounding boxes around detected symbols</p>
                        <p style="color: #333333 !important; margin-bottom: 8px;"><strong style="color: #2c3e50 !important;">CSV:</strong> Contains object IDs, classifications, and coordinates</p>
                        <p style="color: #333333 !important; margin-bottom: 0;"><strong style="color: #2c3e50 !important;">Log:</strong> Check gradio_app.log for detailed processing logs</p>
                    </div>
                    """)
            
            # Event handlers
            process_btn.click(
                fn=gradio_process_image,
                inputs=[image_input],
                outputs=[image_output, csv_output],
                show_progress=True
            )
        
        logger.info("Gradio interface created successfully")
        return demo
        
    except Exception as e:
        logger.error(f"Error creating Gradio interface: {e}")
        logger.error(traceback.format_exc())
        raise

# Cleanup function
def cleanup_temp_files():
    """Clean up any temporary files on exit"""
    try:
        logger.info("Cleaning up temporary files...")
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith('tmp') and (filename.endswith('.jpg') or filename.endswith('.csv')):
                try:
                    file_path = os.path.join(temp_dir, filename)
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up {filename}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_temp_files)

if __name__ == "__main__":
    try:
        logger.info("=== Starting P&ID Symbol Detection Gradio App v2 ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check if model exists
        model_path = "./main_driver/model_Inverted"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            print("ERROR: Model file not found at ./main_driver/model_Inverted")
            print("Please ensure the model file is in the correct location.")
            exit(1)
        
        logger.info("Model file found, proceeding with startup...")
        
        # Load model
        logger.info("Pre-loading model...")
        model = load_model()
        if model is None:
            logger.error("Failed to load model during startup")
            exit(1)
        
        # Create interface
        logger.info("Creating Gradio interface...")
        demo = create_interface()
        
        # Launch the app
        logger.info("Launching Gradio app on port 8080...")
        demo.launch(
            server_port=8080,
            share=True,  # Temporarily disable sharing
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        cleanup_temp_files()
        gc.collect()
        exit(0)
        
    except Exception as e:
        logger.error(f"Critical error starting the application: {e}")
        logger.error(traceback.format_exc())
        cleanup_temp_files()
        exit(1) 