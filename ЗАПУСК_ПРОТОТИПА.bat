@echo off
title Digital Agronomist Prototype
cd /d "C:\Users\metla\Desktop\digital-agronomist-robotrek"
python app.py
if errorlevel 1 (
    cd /d "C:\Users\metla\Desktop"
    python app.py
)
pause
