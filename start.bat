@echo off
title HH GOA 2026 VOICE RAG

cd /d "%~dp0"

call .venv\Scripts\activate

echo ==========================================
echo        HH GOA 2026 VOICE RAG
echo ==========================================
echo.
echo Starting backend...
echo Please wait...
echo.

python backend\main.py

pause