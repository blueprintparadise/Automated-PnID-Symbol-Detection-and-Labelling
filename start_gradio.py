#!/usr/bin/env python3
"""
Simple launcher for the PnID Gradio App
This avoids Windows batch file issues
"""

import subprocess
import sys
import os
import time

def main():
    print("=== PnID Symbol Detection Gradio App Launcher ===")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected!")
        print("Please activate the virtual environment first:")
        print("   .\\venv\\Scripts\\activate")
        print("   python start_gradio.py")
        return
    
    # Check if model exists
    if not os.path.exists("./main_driver/model_Inverted"):
        print("❌ ERROR: Model file not found at ./main_driver/model_Inverted")
        print("Please ensure the model file is in the correct location.")
        return
    
    print("🚀 Starting Gradio App...")
    print("📍 Access the app at: http://127.0.0.1:7860")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the gradio app
        result = subprocess.run([sys.executable, "gradio_app.py"], 
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
    
    print("👋 Goodbye!")

if __name__ == "__main__":
    main() 