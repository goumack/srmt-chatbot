# Script de déploiement automatique pour LexFin sur OpenShift
# Usage: .\deploy.ps1 "Message de commit"

param(
    [Parameter(Mandatory=$false)]
    [string]$commitMessage = "Update: modifications de l'application LexFin"
)

Write-Host "🚀 Déploiement LexFin sur OpenShift" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Chemin vers oc
$OC_PATH = "C:\Users\baye.niang\Desktop\oc"

# 1. Vérifier les modifications
Write-Host "📝 Vérification des modifications..." -ForegroundColor Yellow
git status

Write-Host ""
$confirm = Read-Host "Voulez-vous continuer avec le déploiement? (o/n)"
if ($confirm -ne "o" -and $confirm -ne "O") {
    Write-Host "❌ Déploiement annulé" -ForegroundColor Red
    exit
}

# 2. Add, Commit, Push
Write-Host ""
Write-Host "📤 Envoi des modifications sur GitHub..." -ForegroundColor Yellow
git add .
git commit -m $commitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Aucune modification à commiter ou erreur de commit" -ForegroundColor Yellow
}

git push

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors du push sur GitHub" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Code poussé sur GitHub avec succès" -ForegroundColor Green

# 3. Lancer le build sur OpenShift
Write-Host ""
Write-Host "🔨 Lancement du build sur OpenShift..." -ForegroundColor Yellow
& $OC_PATH start-build srmt-documind --follow

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors du build OpenShift" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build terminé avec succès" -ForegroundColor Green

# 4. Vérifier le déploiement
Write-Host ""
Write-Host "🔍 Vérification du déploiement..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

& $OC_PATH get pods | Select-String "srmt-documind" | Where-Object { $_ -notmatch "build" }

Write-Host ""
Write-Host "📊 Statut de l'application:" -ForegroundColor Yellow
& $OC_PATH get route srmt-documind

Write-Host ""
Write-Host "✅ Déploiement LexFin terminé!" -ForegroundColor Green
Write-Host "🌐 URL: https://srmt-documind-srmt-chat.apps.ocp.heritage.africa" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Commandes utiles:" -ForegroundColor Yellow
Write-Host "  - Voir les logs: $OC_PATH logs -f deployment/srmt-documind" -ForegroundColor Gray
Write-Host "  - Voir les pods: $OC_PATH get pods" -ForegroundColor Gray
Write-Host "  - Redémarrer: $OC_PATH rollout restart deployment/srmt-documind" -ForegroundColor Gray
