#!/usr/bin/env python3
"""
Simple P&ID Image Processing Script
Processes a single image and saves the output
"""

import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import math
import time

# Define the FCN model class (same as in gradio_app.py)
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

# Global variables
model = None
device = None
data_transform = None

def load_model():
    global model, device, data_transform
    
    if model is not None:
        return model
    
    print("Loading model...")
    model = FCN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Setting model up on {device}")
    
    model_location = "./main_driver/model_Inverted"
    
    if not os.path.exists(model_location):
        print(f"ERROR: Model file not found at {model_location}")
        return None
    
    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load(model_location))
    else:
        model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))
    
    data_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(np.expand_dims(np.array(x), axis=0)).float())
    ])
    
    model.eval()
    print("Model loaded successfully!")
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
    
    print(f"Processing image: {image_path}")
    
    # Load model
    model = load_model()
    if model is None:
        return None, None
    
    # Load image
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"ERROR: Could not load image from {image_path}")
        return None, None
    
    print(f"Image loaded successfully. Size: {img.shape}")
    
    # Get ROI
    m, n = img.shape
    x_end = min(x_end, n)
    y_end = min(y_end, m)
    
    print(f"Processing ROI: x={x_start}-{x_end}, y={y_start}-{y_end}")
    
    ready_img = img[y_start:y_end, x_start:x_end]
    
    # Recolor image
    colorized = recolor_crop(ready_img)
    main_img = colorized.copy()
    
    # Get dimensions of ROI
    roi_m, roi_n = ready_img.shape
    print(f"ROI dimensions: {roi_m} x {roi_n}")
    
    # Object detection loop
    objects_info = {}
    object_id = 0
    
    print("Starting object detection...")
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
    
    print(f"Found {len(objects_info)} potential objects")
    
    if len(objects_info) == 0:
        print("No objects detected!")
        return colorized, []
    
    # Group similar objects
    list_of_objects = list(objects_info.keys())
    groups = []
    
    print("Grouping similar objects...")
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
    
    print("Creating final bounding boxes...")
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
        
        # Draw bounding box
        draw_img = cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(draw_img, str(i), (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        final_info[i] = temp
    
    print(f"Final detection: {len(final_info)} objects")
    
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

def main():
    print("=== P&ID Symbol Detection Script ===")
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path>")
        print("Example: python process_image.py 3.png")
        return
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file '{image_path}' not found!")
        return
    
    # Process the image
    start_time = time.time()
    result_img, csv_data = process_pnid_image(image_path)
    end_time = time.time()
    
    if result_img is None:
        print("Failed to process image!")
        return
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save annotated image
    output_image_path = f"{base_name}_detected.png"
    cv2.imwrite(output_image_path, result_img)
    print(f"✅ Annotated image saved: {output_image_path}")
    
    # Save CSV
    if csv_data:
        output_csv_path = f"{base_name}_results.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(output_csv_path, index=False)
        print(f"✅ Results CSV saved: {output_csv_path}")
        print(f"📊 Detected {len(csv_data)} objects:")
        for item in csv_data:
            print(f"   - {item['Component Name']} (ID: {item['Object-ID']})")
    else:
        print("⚠️  No objects detected in the image")
    
    print("🎉 Processing complete!")

if __name__ == "__main__":
    main() 