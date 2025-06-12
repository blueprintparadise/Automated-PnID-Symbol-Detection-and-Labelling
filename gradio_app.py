import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pylab import *
from skimage import data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import math
import json
import time
import tempfile
import shutil
import atexit
import gc
from imutils.object_detection import non_max_suppression

# Define the FCN model class (same as in notebook)
class FCN(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(FCN, self).__init__()
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(self.fc4(x), dim=1)
        return x

# Initialize model globally
model = None
device = None
data_transform = None

def load_model():
    global model, device, data_transform
    
    if model is not None:
        return model
    
    model = FCN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Setting model up on {device}")
    
    model_location = "./main_driver/model_Inverted"
    
    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load(model_location))
    else:
        model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))
    
    data_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(np.expand_dims(np.array(x), axis=0)).float())
    ])
    
    model.eval()
    return model

def recolor_crop(crop):
    """Convert image to black and white"""
    result = crop.copy()
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j] >= 120:
                result[i][j] = 225
            else:
                result[i][j] = 0
    return result

def invert(x):
    """Invert colors for model processing"""
    result = x.copy()
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j] > 200:
                result[i][j] = 0
            else:
                result[i][j] = 1
    return result

def get_distance(x, y):
    """Calculate distance between two centroids"""
    xmid1, ymid1 = x[0], x[1]
    xmid2, ymid2 = y[0], y[1]
    return math.sqrt((ymid2 - ymid1)**2 + (xmid2 - xmid1)**2)

def process_pnid_image(image_path, x_start=400, x_end=5600, y_start=250, y_end=5000):
    """Main processing function"""
    
    # Load model
    model = load_model()
    
    # Load image
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError("Could not load image")
    
    # Get ROI
    m, n = img.shape
    x_end = min(x_end, n)
    y_end = min(y_end, m)
    
    ready_img = img[y_start:y_end, x_start:x_end]
    
    # Recolor image
    colorized = recolor_crop(ready_img)
    main_img = colorized.copy()
    
    # Get dimensions of ROI
    roi_m, roi_n = ready_img.shape
    
    # Object detection loop
    objects_info = {}
    object_id = 0
    
    for i in tqdm(range(0, roi_m-150, 75), desc="Processing image"):
        for j in range(0, roi_n-150, 75):
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
    
    # Group similar objects
    list_of_objects = list(objects_info.keys())
    groups = []
    
    for i in range(len(objects_info)):
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
    
    # Process groups
    groups_dict = {}
    been_done = {}
    
    for i in groups:
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
    
    # Create final bounding boxes
    final_info = {}
    draw_img = colorized.copy()
    
    for i in groups_dict.keys():
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
        
        draw_img = cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (0, 225, 0), 5)
        cv2.putText(draw_img, str(i), (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 225, 0), 3)
        
        final_info[i] = temp
    
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
        class_id = obj_info['class_id']
        bbox = obj_info['bbox']
        component_name = component_names.get(class_id, "Unknown")
        
        csv_data.append({
            "Object-ID": obj_id,
            "Class-ID": class_id,
            "Component Name": component_name,
            "Item Label": "Undefined",  # Simplified for now
            "Location (xmin,xmax,ymin,ymax)": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        })
    
    return draw_img, csv_data

def gradio_process_image(image):
    """Gradio interface function"""
    tmp_file_path = None
    csv_path = None
    
    try:
        # Save uploaded image temporarily with proper cleanup
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tmp_file_path = tmp_file.name
        tmp_file.close()  # Close file handle before saving
        
        # Save the image
        image.save(tmp_file_path)
        
        # Process the image
        result_img, csv_data = process_pnid_image(tmp_file_path)
        
        # Convert result image to RGB for display
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
        
        # Create CSV file with proper handling
        csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', newline='')
        csv_path = csv_file.name
        csv_file.close()  # Close file handle before writing
        
        # Write CSV data
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        return result_img_rgb, csv_path
        
    except Exception as e:
        # Create error image
        error_img = np.zeros((400, 600, 3), dtype=np.uint8)
        error_text = f"Error: {str(e)[:50]}..."  # Truncate long error messages
        cv2.putText(error_img, error_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return error_img, None
        
    finally:
        # Clean up temporary image file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass  # Ignore cleanup errors

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="PnID Symbol Detection and Labelling") as demo:
        gr.Markdown("""
        # PnID Symbol Detection and Labelling System
        
        Upload a P&ID (Piping and Instrumentation Diagram) image to detect and label various symbols.
        
        **Supported symbols:**
        - Valves
        - Control Valves
        - Instruments
        - Spectacle Blinds
        - Circular Components
        - Inline Mixers/Filters
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload P&ID Image")
                process_btn = gr.Button("Process P&ID", variant="primary")
                
            with gr.Column():
                output_image = gr.Image(label="Detected Objects")
                output_csv = gr.File(label="Download Results (CSV)")
        
        process_btn.click(
            fn=gradio_process_image,
            inputs=[input_image],
            outputs=[output_image, output_csv]
        )
        
        gr.Markdown("""
        ### Instructions:
        1. Upload a P&ID diagram image (JPG, PNG, etc.)
        2. Click "Process P&ID" to detect symbols
        3. View the annotated image with bounding boxes
        4. Download the CSV file with detailed results
        
        **Note:** Processing may take a few minutes depending on image size.
        """)
    
    return demo

# Cleanup function
def cleanup_temp_files():
    """Clean up any temporary files on exit"""
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith('tmp') and (filename.endswith('.jpg') or filename.endswith('.csv')):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except:
                pass

# Register cleanup function
atexit.register(cleanup_temp_files)

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists("./main_driver/model_Inverted"):
        print("ERROR: Model file not found at ./main_driver/model_Inverted")
        print("Please ensure the model file is in the correct location.")
        exit(1)
    
    print("Loading model...")
    try:
        load_model()
        print("Model loaded successfully!")
        
        demo = create_interface()
        print("Starting Gradio interface...")
        
        # Launch with improved stability settings
        demo.launch(

            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False,
            inbrowser=False,  # Don't auto-open browser
            prevent_thread_lock=True  # Prevent threading issues
        )
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        cleanup_temp_files()
        gc.collect()  # Force garbage collection
        exit(0)
    except Exception as e:
        print(f"Error starting the application: {e}")
        print("Try restarting the application or check the troubleshooting guide.")
        cleanup_temp_files()
        exit(1) 