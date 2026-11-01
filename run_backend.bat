@echo off
echo ========================================================
echo   EmotionSense AI - Starting FastAPI Backend Engine
echo   PyTorch Vision Transformer + Occlusion Attention
echo ========================================================
cd /d "%~dp0backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
