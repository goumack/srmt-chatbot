@echo off
:: Script pour réindexer les documents LexFin
echo ================================
echo LexFin - Reindexation Documents
echo ================================

:: Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

:: Naviguer vers le dossier du script
cd /d "%~dp0"

:: Lancer le script de réindexation
echo Lancement de l'outil de reindexation...
python reindex_documents.py

pause