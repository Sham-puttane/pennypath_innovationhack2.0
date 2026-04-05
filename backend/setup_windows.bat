@echo off
REM ═══════════════════════════════════════════════════════════════════
REM PennyPath — Windows Quick Fix Script
REM Run this from the backend/ directory
REM ═══════════════════════════════════════════════════════════════════

echo.
echo ══════════════════════════════════════════════════════════════
echo  PennyPath — Setup Fix for Windows
echo ══════════════════════════════════════════════════════════════
echo.

REM --- Fix 1: GraphRAG on Windows ---
echo [FIX 1] Testing GraphRAG...
python -m graphrag --help >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   OK: graphrag found via python -m graphrag
) else (
    echo   GraphRAG not found. Installing...
    pip install graphrag
)

echo.
echo [STEP 1] Initialize GraphRAG...
python -m graphrag init --root .
echo   Done. Now edit settings.yaml with your Gemini API key.

echo.
echo [STEP 2] After editing settings.yaml, run indexing:
echo   python -m graphrag index --root .
echo.
echo   This takes 45-60 minutes. Build frontend while it runs.
echo.

pause
