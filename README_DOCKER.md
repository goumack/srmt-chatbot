# 🐳 SRMT-DOCUMIND avec Docker

Guide complet pour déployer l'Assistant IA Fiscal et Douanier avec Docker Desktop.

## 📋 Prérequis

- **Docker Desktop** installé et en cours d'exécution
- **Git** (optionnel, pour cloner le projet)
- Au moins **4 GB de RAM** disponible pour Docker

## 🚀 Installation Rapide

### 1. Préparer l'environnement

```powershell
# Vérifier que Docker est installé et fonctionne
docker --version
docker-compose --version
```

### 2. Configuration (Optionnelle)

Si vous souhaitez personnaliser la configuration :

```powershell
# Copier le fichier d'exemple
Copy-Item .env.example .env

# Éditer le fichier .env avec vos paramètres
notepad .env
```

### 3. Construire et démarrer l'application

```powershell
# Construire l'image Docker
docker-compose build

# Démarrer l'application
docker-compose up -d
```

### 4. Accéder à l'application

Ouvrez votre navigateur et accédez à :
```
http://localhost:8505
```

## 🔧 Commandes Docker Utiles

### Démarrage et Arrêt

```powershell
# Démarrer l'application
docker-compose up -d

# Arrêter l'application
docker-compose down

# Redémarrer l'application
docker-compose restart
```

### Gestion des Logs

```powershell
# Voir les logs en temps réel
docker-compose logs -f

# Voir les derniers logs
docker-compose logs --tail=100

# Voir les logs d'un service spécifique
docker-compose logs -f srmt-chat
```

### Gestion des Données

```powershell
# Sauvegarder la base de données ChromaDB
docker-compose exec srmt-chat tar czf /tmp/chroma_backup.tar.gz /app/chroma_db

# Supprimer les volumes (ATTENTION: supprime les données)
docker-compose down -v
```

### Maintenance

```powershell
# Reconstruire l'image après modification du code
docker-compose build --no-cache

# Nettoyer les images inutilisées
docker system prune -a

# Vérifier l'état du conteneur
docker-compose ps

# Entrer dans le conteneur pour debug
docker-compose exec srmt-chat /bin/bash
```

## 📁 Structure des Volumes

Les données persistantes sont stockées dans les dossiers suivants :

- `./chroma_db` : Base de données vectorielle ChromaDB
- `./documents` : Documents indexés (surveiller automatiquement)
- `./static` : Fichiers statiques (CSS, JS, images)
- `./templates` : Templates HTML

## 🔄 Mise à Jour de l'Application

```powershell
# 1. Arrêter l'application
docker-compose down

# 2. Mettre à jour le code (si depuis Git)
git pull

# 3. Reconstruire l'image
docker-compose build

# 4. Redémarrer
docker-compose up -d
```

## 🐛 Dépannage

### Le conteneur ne démarre pas

```powershell
# Vérifier les logs d'erreur
docker-compose logs srmt-chat

# Vérifier que le port 8505 n'est pas déjà utilisé
netstat -ano | findstr :8505
```

### Réinitialiser complètement

```powershell
# Arrêter et supprimer tout
docker-compose down -v

# Nettoyer les images
docker system prune -a

# Reconstruire depuis zéro
docker-compose build --no-cache
docker-compose up -d
```

### Problème de mémoire

Si Docker manque de mémoire, augmentez la RAM allouée :
1. Ouvrez Docker Desktop
2. Allez dans **Settings** → **Resources**
3. Augmentez la **Memory** à au moins 4 GB
4. Cliquez sur **Apply & Restart**

## 🌐 Variables d'Environnement

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `OLLAMA_BASE_URL` | URL du serveur Ollama | `https://ollamaaccel-chatbotaccel.apps.senum.heritage.africa` |
| `OLLAMA_CHAT_MODEL` | Modèle de chat | `mistral:7b` |
| `OLLAMA_EMBEDDING_MODEL` | Modèle d'embedding | `nomic-embed-text` |
| `CHROMA_PERSIST_DIRECTORY` | Répertoire ChromaDB | `/app/chroma_db` |
| `WATCH_DIRECTORY` | Répertoire surveillé | `/app/documents` |

## 🔒 Sécurité

Pour un déploiement en production :

1. **Ne pas exposer sur 0.0.0.0** : Modifiez le fichier pour utiliser un reverse proxy
2. **Utiliser HTTPS** : Configurez un certificat SSL/TLS
3. **Variables d'environnement** : Ne committez jamais le fichier `.env`
4. **Limiter les ressources** : Ajoutez des limites CPU/RAM dans docker-compose.yml

## 📊 Monitoring

### Vérifier les ressources utilisées

```powershell
# Stats en temps réel
docker stats srmt-documind

# Utilisation disque
docker system df
```

## 🎯 Production

Pour un déploiement en production, considérez :

1. **Utiliser un orchestrateur** : Kubernetes, Docker Swarm
2. **Ajouter un reverse proxy** : Nginx, Traefik
3. **Sauvegardes automatiques** : Script de backup des volumes
4. **Monitoring** : Prometheus, Grafana
5. **Logs centralisés** : ELK Stack, Loki

## ℹ️ Support

Pour toute question ou problème :
- Consultez les logs : `docker-compose logs -f`
- Vérifiez la documentation Docker : https://docs.docker.com
- Contactez l'équipe de support

---

**Version:** 1.0  
**Dernière mise à jour:** Octobre 2025
