# PnID Symbol Detection and Labelling - Gradio Interface

This is a web-based interface for the Automated PnID Symbol Detection and Labelling system using Gradio.

## Features

- **Web-based Interface**: Upload P&ID images through a user-friendly web interface
- **Real-time Processing**: Process images and get results immediately
- **CSV Export**: Download detection results as CSV files
- **Visual Output**: View annotated images with bounding boxes around detected symbols

## Supported Symbol Types

The system can detect and classify the following P&ID symbols:
- **Valves** (Class 0)
- **Control Valves** (Class 1) 
- **Circular Components** (Class 2)
- **Spectacle Blinds** (Class 3)
- **Inline Mixers/Filters** (Class 4)
- **Instruments** (Class 5)

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r gradio_requirements.txt
```

### 4. Run the Application
```bash
python gradio_app.py
```

The application will start and provide you with:
- **Local URL**: http://127.0.0.1:7860 (for local access)
- **Public URL**: A shareable link for remote access (expires in 1 week)

## How to Use

1. **Upload Image**: Click on the upload area and select your P&ID diagram image (JPG, PNG, etc.)

2. **Process**: Click the "Process P&ID" button to start detection

3. **View Results**: 
   - The processed image will show bounding boxes around detected symbols
   - Each symbol is labeled with an Object ID number

4. **Download CSV**: Click the download button to get a CSV file containing:
   - Object ID
   - Class ID  
   - Component Name
   - Item Label
   - Bounding box coordinates (xmin, xmax, ymin, ymax)

## Processing Details

- The system processes images in 150x150 pixel crops with 50% overlap
- Default ROI (Region of Interest) is set to avoid text areas: x=400-5600, y=250-5000
- Processing time depends on image size (typically 1-3 minutes for standard P&ID diagrams)
- The system runs on CPU by default but will use GPU if available

## Output Format

The CSV file contains the following columns:
- **Object-ID**: Unique identifier for each detected symbol
- **Class-ID**: Numerical class (0-5) representing symbol type
- **Component Name**: Human-readable name of the component
- **Item Label**: Text label associated with the symbol (simplified in current version)
- **Location**: Bounding box coordinates in format "xmin,xmax,ymin,ymax"

## Troubleshooting

### Model Not Found Error
If you get a "Model file not found" error:
- Ensure the model file `model_Inverted` exists in the `./main_driver/` directory
- The model file should be approximately 21MB in size

### Memory Issues
If you encounter memory issues:
- Try processing smaller images
- Close other applications to free up RAM
- Consider using a machine with more memory

### Slow Processing
If processing is very slow:
- Ensure you're using a modern CPU
- Consider reducing the image size before processing
- GPU acceleration will significantly speed up processing if available

## Technical Notes

- Built with PyTorch for deep learning inference
- Uses OpenCV for image processing
- Gradio provides the web interface
- Model architecture: Custom FCN (Fully Convolutional Network)
- Input image format: Grayscale, normalized
- Processing approach: Sliding window with overlap

## Limitations

- OCR functionality is simplified in this version (shows "Undefined" for most labels)
- Best results with high-quality, clear P&ID diagrams
- Processing time scales with image size
- Requires the pre-trained model file to be present

## Future Enhancements

- Full OCR integration with EAST text detection
- Batch processing for multiple images
- Advanced filtering and post-processing options
- Export to additional formats (Excel, PDF)
- Real-time preview during processing 