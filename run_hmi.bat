@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run src\hmi_app.py
) else if exist "venv\Scripts\streamlit.exe" (
    "venv\Scripts\streamlit.exe" run src\hmi_app.py
) else (
    python -m streamlit run src\hmi_app.py
)
