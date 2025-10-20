# Script de test local avant déploiement
# Usage: .\test-local.ps1

Write-Host "🧪 Test Local SRMT-DOCUMIND" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

$choice = Read-Host "Choisissez le mode de test:`n1. Test Python direct`n2. Test Docker`n3. Les deux`nVotre choix (1/2/3)"

if ($choice -eq "1" -or $choice -eq "3") {
    Write-Host ""
    Write-Host "🐍 Test Python direct..." -ForegroundColor Yellow
    python "boutton memoire nouveau .py"
}

if ($choice -eq "2" -or $choice -eq "3") {
    Write-Host ""
    Write-Host "🐳 Test Docker..." -ForegroundColor Yellow
    
    Write-Host "  - Construction de l'image..." -ForegroundColor Gray
    docker-compose build
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Image construite avec succès" -ForegroundColor Green
        Write-Host ""
        Write-Host "  - Démarrage du conteneur..." -ForegroundColor Gray
        docker-compose up -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Conteneur démarré avec succès" -ForegroundColor Green
            Write-Host ""
            Write-Host "  🌐 Application accessible sur: http://localhost:8505" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "  📝 Commandes utiles:" -ForegroundColor Yellow
            Write-Host "    - Voir les logs: docker-compose logs -f" -ForegroundColor Gray
            Write-Host "    - Arrêter: docker-compose down" -ForegroundColor Gray
            Write-Host "    - Redémarrer: docker-compose restart" -ForegroundColor Gray
        } else {
            Write-Host "  ❌ Erreur lors du démarrage du conteneur" -ForegroundColor Red
        }
    } else {
        Write-Host "  ❌ Erreur lors de la construction de l'image" -ForegroundColor Red
    }
}
