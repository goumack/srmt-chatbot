@echo off
echo Configuration OpenAI pour LexFin
echo =================================
echo.

if "%1"=="" (
    echo Usage: setup_openai.bat YOUR_OPENAI_API_KEY
    echo.
    echo Exemple: setup_openai.bat sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    echo.
    echo Pour obtenir votre clé API:
    echo 1. Allez sur https://platform.openai.com/api-keys
    echo 2. Connectez-vous et créez une nouvelle clé API
    echo 3. Copiez la clé et utilisez ce script
    pause
    exit /b 1
)

echo Définition des variables d'environnement...
set OPENAI_API_KEY=%1
set AI_PROVIDER=openai

echo ✅ OPENAI_API_KEY définie
echo ✅ AI_PROVIDER défini sur 'openai'
echo.
echo Variables d'environnement configurées pour cette session.
echo.
echo Test de la configuration...
python test_openai_integration.py

echo.
echo Pour rendre la configuration permanente:
echo 1. Ouvrez les Paramètres système avancés
echo 2. Variables d'environnement
echo 3. Ajoutez OPENAI_API_KEY et AI_PROVIDER

pause