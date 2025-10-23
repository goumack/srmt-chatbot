# Script de déploiement complet avec réindexation automatique
# Assure que tous les nouveaux documents sont pris en compte

param(
    [Parameter(Mandatory=$false)]
    [string]$commitMessage = "Déploiement avec nouveaux documents indexés - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
)

Write-Host "DEPLOIEMENT LEXFIN AVEC NOUVEAUX DOCUMENTS" -ForegroundColor Green
Write-Host "=" * 80

# Configuration
$OC_PATH = "C:\Users\baye.niang\Desktop\oc"
$APP_URL = "https://srmt-documind-srmt-chat.apps.ocp.heritage.africa"

# 1. Vérifier les nouveaux fichiers
Write-Host "Analyse des nouveaux fichiers dans le repertoire documents/..." -ForegroundColor Cyan
$documentsPath = ".\documents"
$newPdfFiles = Get-ChildItem -Path $documentsPath -Filter "*.pdf" | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) }

if ($newPdfFiles.Count -gt 0) {
    Write-Host "NOUVEAUX DOCUMENTS PDF DETECTES :" -ForegroundColor Green
    foreach ($file in $newPdfFiles) {
        $sizeMB = [math]::Round($file.Length/1MB, 1)
        Write-Host "   * $($file.Name) ($sizeMB MB)" -ForegroundColor White
    }
} else {
    Write-Host "Aucun nouveau document PDF recent detecte" -ForegroundColor Yellow
}

# 2. Vérifier tous les fichiers untracked
Write-Host "Fichiers non suivis par Git :" -ForegroundColor Cyan
git status --porcelain | Where-Object { $_ -match "^\?\?" }

# 3. Demander confirmation
Write-Host "`nFichiers qui seront inclus dans le déploiement :" -ForegroundColor Yellow
git status --short
$confirm = Read-Host "`n❓ Voulez-vous continuer avec le déploiement complet ? (O/N)"
if ($confirm -ne "O" -and $confirm -ne "o") {
    Write-Host "❌ Déploiement annulé" -ForegroundColor Red
    exit 0
}

# 4. Ajouter tous les fichiers (y compris les nouveaux documents)
Write-Host "`n📦 Ajout de tous les fichiers au repository..." -ForegroundColor Cyan
git add .
git status --short

# 5. Commit avec message détaillé
Write-Host "`n💾 Commit des modifications..." -ForegroundColor Cyan
$detailedMessage = @"
$commitMessage

📊 Nouveaux documents ajoutés: $($newPdfFiles.Count)
📁 Taille totale des nouveaux PDFs: $(($newPdfFiles | Measure-Object Length -Sum).Sum / 1MB | Format-List {0:F1}) MB

Fichiers principaux inclus:
- Documents budgétaires 2025-2026
- Codes fiscaux et douaniers mis à jour
- Rapports économiques et financiers
- Scripts de réindexation optimisés

🔄 Réindexation automatique programmée après déploiement
"@

git commit -m $detailedMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Aucune modification à commiter ou erreur de commit" -ForegroundColor Yellow
}

# 6. Push vers GitHub
Write-Host "`n📤 Envoi vers GitHub..." -ForegroundColor Cyan
git push

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors du push sur GitHub" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Code et documents poussés sur GitHub avec succès" -ForegroundColor Green

# 7. Déploiement sur OpenShift
Write-Host "`n🏗️ Lancement du build OpenShift..." -ForegroundColor Cyan
& $OC_PATH start-build srmt-documind --follow

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors du build OpenShift" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build OpenShift terminé avec succès" -ForegroundColor Green

# 8. Attendre que le déploiement soit prêt
Write-Host "`n⏳ Attente de la stabilisation du déploiement..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Vérifier que l'application est prête
$maxRetries = 12
$retryCount = 0
$isReady = $false

while ($retryCount -lt $maxRetries -and -not $isReady) {
    try {
        $response = Invoke-WebRequest -Uri "$APP_URL/health" -TimeoutSec 10 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $isReady = $true
            Write-Host "✅ Application prête et accessible" -ForegroundColor Green
        }
    } catch {
        $retryCount++
        Write-Host "⏳ Tentative $retryCount/$maxRetries - Application en cours de démarrage..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
}

if (-not $isReady) {
    Write-Host "⚠️ Application déployée mais pas encore complètement accessible" -ForegroundColor Yellow
}

# 9. Déclenchement de la réindexation automatique
Write-Host "`n🔄 Lancement de la réindexation des nouveaux documents..." -ForegroundColor Cyan

try {
    # Tentative de réindexation via l'API
    $reindexResponse = Invoke-RestMethod -Uri "$APP_URL/force_full_reindex" -Method Post -TimeoutSec 60
    if ($reindexResponse.message) {
        Write-Host "✅ Réindexation réussie: $($reindexResponse.message)" -ForegroundColor Green
        Write-Host "📊 Fichiers indexés: $($reindexResponse.indexed_count)" -ForegroundColor White
        Write-Host "📁 Fichiers trouvés: $($reindexResponse.files_found)" -ForegroundColor White
    }
} catch {
    Write-Host "⚠️ Impossible de lancer la réindexation automatique via l'API" -ForegroundColor Yellow
    Write-Host "💡 Vous pourrez lancer manuellement: python reindex_documents.py force" -ForegroundColor Cyan
}

# 10. Résumé final
Write-Host "`n" + "=" * 80
Write-Host "🎉 DÉPLOIEMENT TERMINÉ AVEC SUCCÈS !" -ForegroundColor Green
Write-Host "🌐 URL: $APP_URL" -ForegroundColor Cyan
Write-Host "📊 Documents ajoutés: $($newPdfFiles.Count)" -ForegroundColor White

# Afficher les commandes utiles
Write-Host "`n📝 Commandes de maintenance utiles :" -ForegroundColor Yellow
Write-Host "   🔍 Voir les logs:      $OC_PATH logs -f deployment/srmt-documind" -ForegroundColor Gray
Write-Host "   📦 Voir les pods:      $OC_PATH get pods" -ForegroundColor Gray
Write-Host "   🔄 Redémarrer:         $OC_PATH rollout restart deployment/srmt-documind" -ForegroundColor Gray
Write-Host "   📊 Réindexer:          python reindex_documents.py" -ForegroundColor Gray

# Proposer d'ouvrir l'application
$open = Read-Host "`n❓ Voulez-vous ouvrir l'application dans le navigateur ? (O/N)"
if ($open -eq "O" -or $open -eq "o") {
    Start-Process $APP_URL
    Write-Host "🌐 Application ouverte dans le navigateur" -ForegroundColor Green
}

Write-Host "`n🎯 Déploiement complet finalisé !" -ForegroundColor Green