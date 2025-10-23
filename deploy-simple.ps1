# Script de deploiement complet avec reindexation automatique
# Assure que tous les nouveaux documents sont pris en compte

param(
    [Parameter(Mandatory=$false)]
    [string]$commitMessage = "Deploiement avec nouveaux documents indexes - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
)

Write-Host "DEPLOIEMENT LEXFIN AVEC NOUVEAUX DOCUMENTS" -ForegroundColor Green
Write-Host "=" * 80

# Configuration
$OC_PATH = "C:\Users\baye.niang\Desktop\oc"
$APP_URL = "https://srmt-documind-srmt-chat.apps.ocp.heritage.africa"

# 1. Verifier les nouveaux fichiers
Write-Host ""
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

# 2. Verifier tous les fichiers untracked
Write-Host ""
Write-Host "Fichiers non suivis par Git :" -ForegroundColor Cyan
git status --porcelain | Where-Object { $_ -match "^\?\?" }

# 3. Demander confirmation
Write-Host ""
Write-Host "Fichiers qui seront inclus dans le deploiement :" -ForegroundColor Yellow
git status --short
$confirm = Read-Host "Voulez-vous continuer avec le deploiement complet ? (O/N)"
if ($confirm -ne "O" -and $confirm -ne "o") {
    Write-Host "Deploiement annule" -ForegroundColor Red
    exit 0
}

# 4. Ajouter tous les fichiers (y compris les nouveaux documents)
Write-Host ""
Write-Host "Ajout de tous les fichiers au repository..." -ForegroundColor Cyan
git add .
git status --short

# 5. Commit avec message detaille
Write-Host ""
Write-Host "Commit des modifications..." -ForegroundColor Cyan
$detailedMessage = @"
$commitMessage

Nouveaux documents ajoutes: $($newPdfFiles.Count)
Taille totale des nouveaux PDFs: $([math]::Round(($newPdfFiles | Measure-Object Length -Sum).Sum / 1MB, 1)) MB

Fichiers principaux inclus:
- Documents budgetaires 2025-2026
- Codes fiscaux et douaniers mis a jour
- Rapports economiques et financiers
- Scripts de reindexation optimises

Reindexation automatique programmee apres deploiement
"@

git commit -m $detailedMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "Aucune modification a commiter ou erreur de commit" -ForegroundColor Yellow
}

# 6. Push vers GitHub
Write-Host ""
Write-Host "Envoi vers GitHub..." -ForegroundColor Cyan
git push

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erreur lors du push sur GitHub" -ForegroundColor Red
    exit 1
}

Write-Host "Code et documents pousses sur GitHub avec succes" -ForegroundColor Green

# 7. Deploiement sur OpenShift
Write-Host ""
Write-Host "Lancement du build OpenShift..." -ForegroundColor Cyan
& $OC_PATH start-build srmt-documind --follow

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erreur lors du build OpenShift" -ForegroundColor Red
    exit 1
}

Write-Host "Build OpenShift termine avec succes" -ForegroundColor Green

# 8. Attendre que le deploiement soit pret
Write-Host ""
Write-Host "Attente de la stabilisation du deploiement..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Verifier que l'application est prete
$maxRetries = 12
$retryCount = 0
$isReady = $false

while ($retryCount -lt $maxRetries -and -not $isReady) {
    try {
        $response = Invoke-WebRequest -Uri "$APP_URL/health" -TimeoutSec 10 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $isReady = $true
            Write-Host "Application prete et accessible" -ForegroundColor Green
        }
    } catch {
        $retryCount++
        Write-Host "Tentative $retryCount/$maxRetries - Application en cours de demarrage..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
}

if (-not $isReady) {
    Write-Host "Application deployee mais pas encore completement accessible" -ForegroundColor Yellow
}

# 9. Declenchement de la reindexation automatique
Write-Host ""
Write-Host "Lancement de la reindexation des nouveaux documents..." -ForegroundColor Cyan

try {
    # Tentative de reindexation via l'API
    $reindexResponse = Invoke-RestMethod -Uri "$APP_URL/force_full_reindex" -Method Post -TimeoutSec 60
    if ($reindexResponse.message) {
        Write-Host "Reindexation reussie: $($reindexResponse.message)" -ForegroundColor Green
        Write-Host "Fichiers indexes: $($reindexResponse.indexed_count)" -ForegroundColor White
        Write-Host "Fichiers trouves: $($reindexResponse.files_found)" -ForegroundColor White
    }
} catch {
    Write-Host "Impossible de lancer la reindexation automatique via l'API" -ForegroundColor Yellow
    Write-Host "Vous pourrez lancer manuellement: python reindex_documents.py force" -ForegroundColor Cyan
}

# 10. Resume final
Write-Host ""
Write-Host "=" * 80
Write-Host "DEPLOIEMENT TERMINE AVEC SUCCES !" -ForegroundColor Green
Write-Host "URL: $APP_URL" -ForegroundColor Cyan
Write-Host "Documents ajoutes: $($newPdfFiles.Count)" -ForegroundColor White

# Afficher les commandes utiles
Write-Host ""
Write-Host "Commandes de maintenance utiles :" -ForegroundColor Yellow
Write-Host "   Voir les logs:      $OC_PATH logs -f deployment/srmt-documind" -ForegroundColor Gray
Write-Host "   Voir les pods:      $OC_PATH get pods" -ForegroundColor Gray
Write-Host "   Redemarrer:         $OC_PATH rollout restart deployment/srmt-documind" -ForegroundColor Gray
Write-Host "   Reindexer:          python reindex_documents.py" -ForegroundColor Gray

# Proposer d'ouvrir l'application
$open = Read-Host "Voulez-vous ouvrir l'application dans le navigateur ? (O/N)"
if ($open -eq "O" -or $open -eq "o") {
    Start-Process $APP_URL
    Write-Host "Application ouverte dans le navigateur" -ForegroundColor Green
}

Write-Host ""
Write-Host "Deploiement complet finalise !" -ForegroundColor Green