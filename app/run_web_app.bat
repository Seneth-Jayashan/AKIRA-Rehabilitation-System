@echo off
title AKIRA Knee Rehabilitation Edge-AI Dashboard
cd /d "%~dp0"
echo ======================================================================
echo Starting AKIRA Edge-AI Knee Rehabilitation Web Interface...
echo URL: http://127.0.0.1:8000
echo ======================================================================
python run_web_app.py
pause
