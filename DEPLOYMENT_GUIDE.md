# Guide de Mise à Jour et Déploiement SRMT-DOCUMIND

## 🔄 Procédures de Mise à Jour

### 1️⃣ Modifications du Code (Python, HTML, CSS, JS)

```powershell
# Étape 1: Modifier vos fichiers
# (Éditez dans VS Code)

# Étape 2: Tester localement (recommandé)
.\test-local.ps1

# Étape 3: Déployer automatiquement
.\deploy.ps1 "Description de vos modifications"
```

**OU manuellement:**

```powershell
# 1. Commit et push
git add .
git commit -m "Update: vos modifications"
git push

# 2. Rebuild sur OpenShift
C:\Users\baye.niang\Desktop\oc start-build srmt-documind --follow

# 3. Vérifier
C:\Users\baye.niang\Desktop\oc get pods
C:\Users\baye.niang\Desktop\oc logs -f deployment/srmt-documind
```

### 2️⃣ Modifications du Dockerfile

```powershell
# 1. Modifier le Dockerfile

# 2. Tester localement
docker-compose build
docker-compose up

# 3. Si OK, déployer
.\deploy.ps1 "Update: Dockerfile modifié"
```

### 3️⃣ Ajout/Modification de Dépendances

```powershell
# 1. Modifier requirements.txt
# Ajouter ou modifier les versions des packages

# 2. Tester localement
pip install -r requirements.txt
python "boutton memoire nouveau .py"

# 3. Déployer
.\deploy.ps1 "Update: nouvelles dépendances"
```

### 4️⃣ Modifications de Configuration OpenShift

```powershell
# Pour modifier les ressources (CPU, RAM, réplicas, etc.)

# 1. Modifier openshift/deployment.yaml

# 2. Appliquer les changements
C:\Users\baye.niang\Desktop\oc apply -f openshift/deployment.yaml

# 3. Sauvegarder dans Git
git add openshift/
git commit -m "Update: configuration OpenShift"
git push
```

### 5️⃣ Modification des Variables d'Environnement

```powershell
# Option A: Via ConfigMap
# 1. Modifier openshift/configmap.yaml
# 2. Appliquer
C:\Users\baye.niang\Desktop\oc apply -f openshift/configmap.yaml

# 3. Redémarrer le deployment
C:\Users\baye.niang\Desktop\oc rollout restart deployment/srmt-documind

# Option B: Directement via CLI
C:\Users\baye.niang\Desktop\oc set env deployment/srmt-documind \
  OLLAMA_BASE_URL=https://nouvelle-url.com \
  OLLAMA_CHAT_MODEL=mistral:7b
```

### 6️⃣ Redémarrage Simple (sans rebuild)

```powershell
# Redémarrer l'application sans rebuild
C:\Users\baye.niang\Desktop\oc rollout restart deployment/srmt-documind

# Suivre le redémarrage
C:\Users\baye.niang\Desktop\oc rollout status deployment/srmt-documind
```

## 📊 Workflow Recommandé

```
┌─────────────────────┐
│  Modifier le code   │
│   (VS Code)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Tester localement  │
│  .\test-local.ps1   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Déployer auto     │
│   .\deploy.ps1      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Vérifier sur       │
│  OpenShift          │
└─────────────────────┘
```

## 🛠️ Commandes Utiles

### Gestion des Builds

```powershell
# Voir l'historique des builds
C:\Users\baye.niang\Desktop\oc get builds

# Voir les logs d'un build spécifique
C:\Users\baye.niang\Desktop\oc logs build/srmt-documind-2

# Annuler un build en cours
C:\Users\baye.niang\Desktop\oc cancel-build srmt-documind-3

# Supprimer un build échoué
C:\Users\baye.niang\Desktop\oc delete build srmt-documind-3
```

### Gestion des Pods

```powershell
# Lister les pods
C:\Users\baye.niang\Desktop\oc get pods

# Voir les logs en temps réel
C:\Users\baye.niang\Desktop\oc logs -f deployment/srmt-documind

# Voir les logs des 100 dernières lignes
C:\Users\baye.niang\Desktop\oc logs --tail=100 deployment/srmt-documind

# Accéder au shell du pod
C:\Users\baye.niang\Desktop\oc rsh deployment/srmt-documind

# Exécuter une commande dans le pod
C:\Users\baye.niang\Desktop\oc exec deployment/srmt-documind -- ls -la /opt/app-root/src
```

### Gestion du Deployment

```powershell
# Scaler l'application
C:\Users\baye.niang\Desktop\oc scale deployment/srmt-documind --replicas=3

# Voir l'historique des déploiements
C:\Users\baye.niang\Desktop\oc rollout history deployment/srmt-documind

# Rollback vers la version précédente
C:\Users\baye.niang\Desktop\oc rollout undo deployment/srmt-documind

# Rollback vers une version spécifique
C:\Users\baye.niang\Desktop\oc rollout undo deployment/srmt-documind --to-revision=2
```

### Gestion des Données

```powershell
# Voir les PVC (volumes persistants)
C:\Users\baye.niang\Desktop\oc get pvc

# Sauvegarder les données ChromaDB
C:\Users\baye.niang\Desktop\oc rsync deployment/srmt-documind:/opt/app-root/src/chroma_db ./backup/chroma_db

# Restaurer les données
C:\Users\baye.niang\Desktop\oc rsync ./backup/chroma_db/ deployment/srmt-documind:/opt/app-root/src/chroma_db
```

### Surveillance

```powershell
# Voir le statut général
C:\Users\baye.niang\Desktop\oc status

# Voir toutes les ressources
C:\Users\baye.niang\Desktop\oc get all

# Voir les événements
C:\Users\baye.niang\Desktop\oc get events --sort-by='.lastTimestamp'

# Voir les métriques (CPU, RAM)
C:\Users\baye.niang\Desktop\oc adm top pods
```

## 🚨 Dépannage

### Le pod ne démarre pas

```powershell
# 1. Voir les détails du pod
C:\Users\baye.niang\Desktop\oc describe pod <nom-du-pod>

# 2. Voir les logs
C:\Users\baye.niang\Desktop\oc logs <nom-du-pod>

# 3. Voir les événements
C:\Users\baye.niang\Desktop\oc get events | Select-String "srmt-documind"
```

### Le build échoue

```powershell
# 1. Voir les logs du build
C:\Users\baye.niang\Desktop\oc logs -f build/srmt-documind-X

# 2. Vérifier le Dockerfile localement
docker build -t test .

# 3. Relancer le build
C:\Users\baye.niang\Desktop\oc start-build srmt-documind --follow
```

### L'application ne répond pas

```powershell
# 1. Vérifier que le pod est Running
C:\Users\baye.niang\Desktop\oc get pods

# 2. Tester depuis l'intérieur du pod
C:\Users\baye.niang\Desktop\oc exec deployment/srmt-documind -- curl -s http://localhost:8505

# 3. Vérifier les endpoints
C:\Users\baye.niang\Desktop\oc get endpoints srmt-documind

# 4. Redémarrer
C:\Users\baye.niang\Desktop\oc rollout restart deployment/srmt-documind
```

## 📝 Checklist Avant Déploiement

- [ ] Code testé localement
- [ ] Pas d'erreurs de syntaxe
- [ ] Dependencies à jour dans requirements.txt
- [ ] Dockerfile valide (si modifié)
- [ ] Variables d'environnement correctes
- [ ] Commit message descriptif
- [ ] Backup des données importantes

## 🎯 Bonnes Pratiques

1. **Toujours tester localement** avant de déployer
2. **Commiter régulièrement** avec des messages clairs
3. **Suivre les logs** pendant le build
4. **Sauvegarder les données** avant les grosses modifications
5. **Documenter** les changements importants
6. **Utiliser les scripts** deploy.ps1 et test-local.ps1 pour la cohérence

## 📚 Ressources

- Repository GitHub: https://github.com/goumack/srmt-chatbot
- URL Production: https://srmt-documind-srmt-chat.apps.ocp.heritage.africa
- Documentation Docker: README_DOCKER.md
- Documentation OpenShift: README_OPENSHIFT.md

---

**Version:** 1.0  
**Dernière mise à jour:** Octobre 2025
