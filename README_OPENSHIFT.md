# 🚀 Déploiement SRMT-DOCUMIND sur OpenShift

Guide complet pour déployer l'Assistant IA Fiscal et Douanier sur Red Hat OpenShift.

## 📋 Prérequis

- **OpenShift CLI (oc)** installé
- Accès à un cluster OpenShift
- Compte avec permissions suffisantes
- Git configuré

## 🔐 Connexion à OpenShift

```bash
# Se connecter à votre cluster OpenShift
oc login https://your-openshift-cluster-url --token=your-token

# Vérifier la connexion
oc whoami
oc version
```

## 📦 Méthode 1 : Déploiement avec les fichiers YAML

### 1. Créer un nouveau projet

```bash
# Créer un projet OpenShift
oc new-project srmt-chat

# Vérifier le projet actif
oc project
```

### 2. Déployer les ressources

```bash
# Appliquer tous les fichiers de configuration
oc apply -f openshift/configmap.yaml
oc apply -f openshift/pvc-chroma.yaml
oc apply -f openshift/pvc-documents.yaml
oc apply -f openshift/imagestream.yaml
oc apply -f openshift/buildconfig.yaml
oc apply -f openshift/deployment.yaml
oc apply -f openshift/service.yaml
oc apply -f openshift/route.yaml
```

Ou appliquer tous les fichiers en une seule commande :

```bash
oc apply -k openshift/
```

### 3. Construire l'image

```bash
# Démarrer le build depuis le Dockerfile
oc start-build srmt-documind --from-dir=. --follow

# Vérifier le statut du build
oc get builds
oc logs -f build/srmt-documind-1
```

### 4. Vérifier le déploiement

```bash
# Vérifier les pods
oc get pods

# Vérifier les services
oc get svc

# Vérifier les routes
oc get route

# Obtenir l'URL de l'application
oc get route srmt-documind -o jsonpath='{.spec.host}'
```

## 📦 Méthode 2 : Déploiement depuis Docker Hub

Si vous avez déjà une image Docker :

```bash
# Tag de l'image locale
docker tag srmt-documind:latest your-registry/srmt-documind:latest

# Push vers un registry
docker push your-registry/srmt-documind:latest

# Créer le deployment depuis l'image
oc new-app your-registry/srmt-documind:latest --name=srmt-documind

# Exposer le service
oc expose svc/srmt-documind
```

## 📦 Méthode 3 : Déploiement depuis GitHub (Source-to-Image)

```bash
# Créer une nouvelle application depuis GitHub
oc new-app python:3.10~https://github.com/goumack/ALEX-Assistant-IA.git \
  --name=srmt-documind \
  --strategy=docker

# Exposer le service
oc expose svc/srmt-documind

# Suivre les logs du build
oc logs -f bc/srmt-documind
```

## 🔧 Configuration Post-Déploiement

### Configurer les variables d'environnement

```bash
# Mettre à jour la ConfigMap
oc edit configmap srmt-config

# Ou via la ligne de commande
oc set env deployment/srmt-documind \
  OLLAMA_BASE_URL=https://your-ollama-url.com \
  OLLAMA_CHAT_MODEL=mistral:7b
```

### Ajuster les ressources

```bash
# Augmenter les limites de ressources
oc set resources deployment/srmt-documind \
  --limits=cpu=2,memory=4Gi \
  --requests=cpu=500m,memory=1Gi
```

### Scaler l'application

```bash
# Augmenter le nombre de réplicas
oc scale deployment/srmt-documind --replicas=3

# Vérifier les réplicas
oc get deployment srmt-documind
```

## 📊 Surveillance et Logs

### Voir les logs

```bash
# Logs en temps réel
oc logs -f deployment/srmt-documind

# Logs d'un pod spécifique
oc logs -f pod/srmt-documind-xxxxx

# Logs des 100 dernières lignes
oc logs --tail=100 deployment/srmt-documind
```

### Vérifier l'état

```bash
# État des pods
oc get pods -w

# Détails d'un pod
oc describe pod srmt-documind-xxxxx

# Événements du projet
oc get events --sort-by='.lastTimestamp'
```

### Accéder au conteneur

```bash
# Ouvrir un shell dans le pod
oc rsh deployment/srmt-documind

# Exécuter une commande
oc exec deployment/srmt-documind -- ls -la /app
```

## 🔄 Mise à Jour de l'Application

### Méthode 1 : Rebuild depuis le source

```bash
# Déclencher un nouveau build
oc start-build srmt-documind --follow

# Le deployment sera mis à jour automatiquement
```

### Méthode 2 : Mettre à jour l'image

```bash
# Tagger une nouvelle version
oc tag srmt-documind:latest srmt-documind:v2

# Rollout de la nouvelle version
oc rollout latest deployment/srmt-documind

# Vérifier le statut du rollout
oc rollout status deployment/srmt-documind
```

### Rollback

```bash
# Voir l'historique des déploiements
oc rollout history deployment/srmt-documind

# Rollback vers la version précédente
oc rollout undo deployment/srmt-documind

# Rollback vers une version spécifique
oc rollout undo deployment/srmt-documind --to-revision=2
```

## 💾 Gestion des Volumes

### Vérifier les PVC

```bash
# Lister les PVC
oc get pvc

# Détails d'un PVC
oc describe pvc srmt-chroma-pvc
```

### Sauvegarder les données

```bash
# Copier les données depuis le pod
oc rsync deployment/srmt-documind:/app/chroma_db ./backup/chroma_db

# Copier les documents
oc rsync deployment/srmt-documind:/app/documents ./backup/documents
```

### Restaurer les données

```bash
# Copier les données vers le pod
oc rsync ./backup/chroma_db/ deployment/srmt-documind:/app/chroma_db

# Copier les documents
oc rsync ./backup/documents/ deployment/srmt-documind:/app/documents
```

## 🌐 Configuration du Réseau

### Vérifier la route

```bash
# Obtenir l'URL publique
oc get route srmt-documind

# Tester l'accès
curl https://$(oc get route srmt-documind -o jsonpath='{.spec.host}')
```

### Configurer HTTPS personnalisé

```bash
# Créer un secret TLS
oc create secret tls srmt-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem

# Mettre à jour la route
oc patch route srmt-documind -p '{"spec":{"tls":{"certificate":"...","key":"..."}}}'
```

## 🔒 Sécurité

### Configurer les limites de sécurité

```bash
# Créer un SecurityContextConstraints (SCC)
oc adm policy add-scc-to-user anyuid -z default

# Ou créer un SCC personnalisé
oc apply -f openshift/scc.yaml
```

### Secrets

```bash
# Créer un secret
oc create secret generic srmt-secrets \
  --from-literal=api-key=your-api-key

# Utiliser le secret dans le deployment
oc set env deployment/srmt-documind \
  --from=secret/srmt-secrets
```

## 📈 Auto-Scaling

```bash
# Configurer l'autoscaling horizontal
oc autoscale deployment/srmt-documind \
  --min=1 --max=5 \
  --cpu-percent=80

# Vérifier l'HPA
oc get hpa
```

## 🐛 Dépannage

### Pod ne démarre pas

```bash
# Vérifier les événements
oc describe pod srmt-documind-xxxxx

# Vérifier les logs
oc logs srmt-documind-xxxxx

# Vérifier les ressources
oc get events --sort-by='.lastTimestamp'
```

### Problèmes de build

```bash
# Voir les logs du build
oc logs -f build/srmt-documind-1

# Annuler un build
oc cancel-build srmt-documind-1

# Supprimer un build échoué
oc delete build srmt-documind-1
```

### Problèmes de réseau

```bash
# Tester la connectivité depuis le pod
oc exec deployment/srmt-documind -- curl localhost:8505

# Vérifier les endpoints
oc get endpoints srmt-documind
```

## 🗑️ Nettoyage

```bash
# Supprimer toutes les ressources
oc delete all -l app=srmt-documind

# Supprimer les PVC (ATTENTION: supprime les données)
oc delete pvc srmt-chroma-pvc srmt-documents-pvc

# Supprimer le projet complet
oc delete project srmt-chat
```

## 📚 Ressources Supplémentaires

- [Documentation OpenShift](https://docs.openshift.com/)
- [OpenShift CLI Reference](https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## 🎯 Configuration Production

Pour un environnement de production :

1. **Augmenter les réplicas** : `replicas: 3`
2. **Configurer l'autoscaling** : HPA avec métriques CPU/RAM
3. **Ajouter des health checks** : Liveness et Readiness probes
4. **Configurer les limites de ressources** : CPU et mémoire
5. **Utiliser des secrets** : Pour les informations sensibles
6. **Configurer des backups** : Automatiser les sauvegardes des PVC
7. **Monitoring** : Prometheus, Grafana
8. **Logging** : EFK Stack (Elasticsearch, Fluentd, Kibana)

## ⚙️ Variables d'Environnement Importantes

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `OLLAMA_BASE_URL` | URL du serveur Ollama | Via ConfigMap |
| `OLLAMA_CHAT_MODEL` | Modèle de chat | `mistral:7b` |
| `OLLAMA_EMBEDDING_MODEL` | Modèle d'embedding | `nomic-embed-text` |
| `CHROMA_PERSIST_DIRECTORY` | Répertoire ChromaDB | `/app/chroma_db` |
| `WATCH_DIRECTORY` | Répertoire surveillé | `/app/documents` |

---

**Version:** 1.0  
**Dernière mise à jour:** Octobre 2025  
**Support:** OpenShift 4.x+
