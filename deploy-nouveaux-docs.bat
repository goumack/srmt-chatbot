@echo off
echo.
echo ===============================================
echo  DEPLOIEMENT LEXFIN AVEC NOUVEAUX DOCUMENTS
echo ===============================================
echo.

echo 📁 Verification des nouveaux documents...
dir documents\*.pdf /O:D | findstr /C:"2024" /C:"2025"

echo.
echo 📝 Fichiers a deployer:
git status --short

echo.
set /p confirm="❓ Continuer le deploiement? (O/N): "
if /i "%confirm%" NEQ "O" (
    echo ❌ Deploiement annule
    pause
    exit /b 1
)

echo.
echo 📦 Lancement du script PowerShell optimise...
powershell.exe -ExecutionPolicy Bypass -File "deploy-with-reindex.ps1"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Deploiement termine avec succes!
    echo 🌐 URL: https://srmt-documind-srmt-chat.apps.ocp.heritage.africa
) else (
    echo.
    echo ❌ Erreur lors du deploiement
)

echo.
pause