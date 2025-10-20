# Script de déploiement automatique sur OpenShift avec améliorations du prompt
# Usage: .\deploy-with-improvements.ps1

Write-Host "🚀 DÉPLOIEMENT DES AMÉLIORATIONS SUR OPENSHIFT" -ForegroundColor Green
Write-Host "=" * 70

# 1. Vérifier les modifications
Write-Host "`n📝 Modifications détectées:" -ForegroundColor Cyan
git status --short

# 2. Demander confirmation
$confirm = Read-Host "`n❓ Voulez-vous déployer ces modifications sur OpenShift ? (O/N)"
if ($confirm -ne "O" -and $confirm -ne "o") {
    Write-Host "❌ Déploiement annulé" -ForegroundColor Red
    exit 0
}

# 3. Commit et push
Write-Host "`n📦 Commit et push des modifications..." -ForegroundColor Cyan
git add "boutton memoire nouveau .py"
git commit -m "Amélioration majeure du prompt système pour meilleure compréhension des questions

- Augmentation des résultats de recherche (3 → 10 documents)
- Ajout d'instructions détaillées pour l'analyse de la question
- Consignes explicites pour éviter les confusions entre différents types de taux
- Validation de la pertinence des articles trouvés
- Méthodologie en 4 étapes pour répondre précisément aux questions
"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors du commit" -ForegroundColor Red
    exit 1
}

git push
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors du push" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Code poussé sur GitHub" -ForegroundColor Green

# 4. Déclencher le build OpenShift
Write-Host "`n🏗️ Déclenchement du build OpenShift..." -ForegroundColor Cyan
C:\Users\baye.niang\Desktop\oc start-build srmt-documind --follow

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors du build" -ForegroundColor Red
    exit 1
}

# 5. Vérifier le déploiement
Write-Host "`n🔍 Vérification du déploiement..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

$pods = C:\Users\baye.niang\Desktop\oc get pods -l deployment=srmt-documind --no-headers
Write-Host "📦 Pods actifs:"
Write-Host $pods

# Récupérer le nom du dernier pod
$latestPod = (C:\Users\baye.niang\Desktop\oc get pods -l deployment=srmt-documind --sort-by=.metadata.creationTimestamp --no-headers | Select-Object -Last 1).Split()[0]

if ($latestPod) {
    Write-Host "`n📋 Logs du pod $latestPod :" -ForegroundColor Cyan
    C:\Users\baye.niang\Desktop\oc logs $latestPod --tail=30
    
    Write-Host "`n✅ Déploiement terminé avec succès !" -ForegroundColor Green
    Write-Host "🌐 Application accessible à: https://srmt-documind-srmt-chat.apps.ocp.heritage.africa" -ForegroundColor Cyan
    
    # Demander si on veut ouvrir l'application
    $open = Read-Host "`n❓ Voulez-vous ouvrir l'application dans le navigateur ? (O/N)"
    if ($open -eq "O" -or $open -eq "o") {
        Start-Process "https://srmt-documind-srmt-chat.apps.ocp.heritage.africa"
    }
} else {
    Write-Host "⚠️ Aucun pod trouvé" -ForegroundColor Yellow
}

Write-Host "`n" + "=" * 70
Write-Host "🎉 Déploiement terminé !" -ForegroundColor Green
