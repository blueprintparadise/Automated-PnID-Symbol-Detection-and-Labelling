@echo off
echo Activating virtual environment...
call .\venv\Scripts\activate

echo Starting PnID Detection Gradio App...
python gradio_app.py

pause 