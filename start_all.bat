@echo off
echo ===================================================================
echo   EmotionSense AI – Smart Facial Emotion Recognition System
echo   Starting both Backend (FastAPI :8000) and Frontend (Vite :5173)...
echo ===================================================================
start "EmotionSense AI - Backend API" cmd /k "run_backend.bat"
timeout /t 3 /nobreak >nul
start "EmotionSense AI - Frontend UI" cmd /k "run_frontend.bat"
echo Applications launched! Access the dashboard at http://localhost:5173
pause
