#!/usr/bin/env python3
"""
Simple launcher for the PnID Gradio App v2
Enhanced with error logging and port 8080
"""

import subprocess
import sys
import os
import time

def main():
    print("=== PnID Symbol Detection Gradio App v2 Launcher ===")
    print("🚀 Enhanced with comprehensive error logging")
    print("🌐 Running on port 8080")
    print("-" * 55)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected!")
        print("Please activate the virtual environment first:")
        print("   .\\venv\\Scripts\\activate")
        print("   python start_gradio_v2.py")
        return
    
    # Check if model exists
    if not os.path.exists("./main_driver/model_Inverted"):
        print("❌ ERROR: Model file not found at ./main_driver/model_Inverted")
        print("Please ensure the model file is in the correct location.")
        return
    
    print("🚀 Starting Gradio App v2...")
    print("📍 Access the app at: http://127.0.0.1:8080 or http://0.0.0.0:8080")
    print("📋 Logs will be saved to: gradio_app.log")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 55)
    
    try:
        # Run the gradio app v2
        result = subprocess.run([sys.executable, "gradio_app_v2.py"], 
                              check=False, 
                              cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"❌ App exited with code: {result.returncode}")
        else:
            print("✅ App shut down normally")
            
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("📄 Check gradio_app.log for detailed logs")
    print("👋 Goodbye!")

if __name__ == "__main__":
    main() 