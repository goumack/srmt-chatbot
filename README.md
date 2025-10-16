# 🇸🇳 SRMT-DOCUMIND - Assistant IA Fiscal et Douanier Sénégal

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![OpenShift](https://img.shields.io/badge/OpenShift-Compatible-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Assistant IA intelligent spécialisé dans le Code des Impôts et le Code des Douanes du Sénégal**

[Fonctionnalités](#-fonctionnalités) •
[Installation](#-installation-rapide) •
[Docker](#-déploiement-docker) •
[OpenShift](#-déploiement-openshift) •
[Documentation](#-documentation)

</div>

---

## 📋 Description

**SRMT-DOCUMIND** est un assistant IA conversationnel spécialisé dans le domaine fiscal et douanier sénégalais. Il utilise une approche RAG (Retrieval-Augmented Generation) stricte pour fournir des réponses précises basées uniquement sur les documents officiels indexés.

### 🎯 Fonctionnalités Principales

- ✅ **Mode RAG Strict** - Réponses basées uniquement sur les documents indexés
- 🔍 **Recherche Sémantique** - Utilisation de ChromaDB pour une recherche vectorielle performante
- 📚 **Indexation Automatique** - Surveillance des nouveaux documents en temps réel
- 🤖 **IA Ollama** - Intégration avec Mistral 7B pour des réponses pertinentes
- 📊 **Multi-formats** - Support PDF, DOCX, CSV, Excel, TXT, MD, JSON
- 🏛️ **Spécialisation** - Code des Impôts & Code des Douanes du Sénégal
- 🌐 **Interface Web** - Interface utilisateur intuitive
- 🐳 **Docker Ready** - Déploiement facile avec Docker
- ☁️ **OpenShift Compatible** - Déploiement cloud native

## 🚀 Installation Rapide

### Prérequis

- Python 3.10+
- Git
- Docker Desktop (optionnel)
- OpenShift CLI (pour déploiement cloud)

### Installation Locale

```bash
# Cloner le repository
git clone https://github.com/goumack/ALEX-Assistant-IA.git
cd ALEX-Assistant-IA

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement (optionnel)
cp .env.example .env
# Éditer .env selon vos besoins

# Lancer l'application
python "boutton memoire nouveau .py"
```

L'application sera accessible sur : **http://localhost:8505**

## 🐳 Déploiement Docker

### Docker Compose (Recommandé)

```bash
# Construire l'image
docker-compose build

# Démarrer l'application
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter l'application
docker-compose down
```

### Docker Simple

```bash
# Construire l'image
docker build -t srmt-documind .

# Lancer le conteneur
docker run -d -p 8505:8505 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/documents:/app/documents \
  --name srmt-documind \
  srmt-documind
```

📖 **Documentation complète** : [README_DOCKER.md](README_DOCKER.md)

## ☁️ Déploiement OpenShift

### Déploiement Rapide

```bash
# Se connecter à OpenShift
oc login https://your-cluster-url --token=your-token

# Créer un projet
oc new-project srmt-chat

# Déployer depuis GitHub
oc new-app python:3.10~https://github.com/goumack/ALEX-Assistant-IA.git \
  --name=srmt-documind \
  --strategy=docker

# Exposer le service
oc expose svc/srmt-documind

# Obtenir l'URL
oc get route srmt-documind
```

### Déploiement avec Kustomize

```bash
# Déployer toutes les ressources
oc apply -k openshift/

# Vérifier le déploiement
oc get pods
oc get route
```

📖 **Documentation complète** : [README_OPENSHIFT.md](README_OPENSHIFT.md)

## 📁 Structure du Projet

```
SRMT CHAT/
├── boutton memoire nouveau .py   # Application principale Flask
├── requirements.txt              # Dépendances Python
├── Dockerfile                    # Configuration Docker
├── docker-compose.yml            # Orchestration Docker
├── .gitignore                    # Fichiers à ignorer
├── .env.example                  # Template de configuration
├── chroma_db/                    # Base de données vectorielle
├── documents/                    # Documents à indexer
├── static/                       # Fichiers statiques (CSS, JS)
├── templates/                    # Templates HTML
├── openshift/                    # Configuration OpenShift
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── route.yaml
│   ├── configmap.yaml
│   └── ...
└── backup/                       # Fichiers de backup

```

## ⚙️ Configuration

### Variables d'Environnement

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `OLLAMA_BASE_URL` | URL du serveur Ollama | `https://ollamaaccel-chatbotaccel.apps.senum.heritage.africa` |
| `OLLAMA_CHAT_MODEL` | Modèle de chat | `mistral:7b` |
| `OLLAMA_EMBEDDING_MODEL` | Modèle d'embedding | `nomic-embed-text` |
| `CHROMA_PERSIST_DIRECTORY` | Répertoire ChromaDB | `./chroma_db` |
| `WATCH_DIRECTORY` | Répertoire surveillé | `./documents` |

### Formats de Documents Supportés

- 📄 PDF (`.pdf`)
- 📝 Word (`.docx`, `.odt`)
- 📊 Excel (`.xlsx`, `.xls`)
- 📋 CSV (`.csv`)
- 📃 Texte (`.txt`, `.md`)
- 🔧 JSON (`.json`)

## 🛠️ Utilisation

### Ajouter des Documents

1. Placer vos documents dans le dossier `documents/`
2. L'application les indexera automatiquement
3. Vérifier l'indexation dans les logs

### Poser des Questions

1. Accéder à l'interface web : http://localhost:8505
2. Poser une question fiscale ou douanière
3. L'IA recherchera dans les documents indexés
4. Recevoir une réponse précise avec les sources

### API Endpoints

```bash
# Chat
POST /chat
{
  "message": "Quelle est la TVA au Sénégal ?"
}

# Indexer un document
POST /add_document
{
  "content": "...",
  "source": "nom_du_document"
}

# Statistiques
GET /stats
```

## 📊 Surveillance et Logs

```bash
# Logs locaux
tail -f logs/app.log

# Logs Docker
docker-compose logs -f

# Logs OpenShift
oc logs -f deployment/srmt-documind
```

## 🔧 Développement

### Installer les dépendances de développement

```bash
pip install -r requirements.txt
```

### Lancer en mode debug

```python
# Modifier dans boutton memoire nouveau .py
app.run(host="0.0.0.0", port=8505, debug=True)
```

### Tests

```bash
# Exécuter les tests (à venir)
pytest tests/
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📖 Documentation

- [Guide Docker](README_DOCKER.md) - Déploiement avec Docker
- [Guide OpenShift](README_OPENSHIFT.md) - Déploiement sur OpenShift
- [API Documentation](docs/API.md) - Documentation de l'API (à venir)

## 🐛 Problèmes Connus

- La première indexation peut prendre du temps selon la taille des documents
- Nécessite une connexion au serveur Ollama configuré

## 📝 Roadmap

- [ ] Tests unitaires et d'intégration
- [ ] Interface d'administration
- [ ] Support de langues supplémentaires
- [ ] API REST complète
- [ ] Documentation Swagger/OpenAPI
- [ ] Métriques et monitoring Prometheus
- [ ] CI/CD avec GitHub Actions

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- **Baye Niang** - *Développeur Principal* - [@goumack](https://github.com/goumack)

## 🙏 Remerciements

- Direction Générale des Impôts et des Domaines du Sénégal
- Direction Générale des Douanes du Sénégal
- Communauté Ollama
- Équipe ChromaDB

## 📞 Support

Pour toute question ou support :
- 📧 Email : support@example.com
- 🐛 Issues : [GitHub Issues](https://github.com/goumack/ALEX-Assistant-IA/issues)

---

<div align="center">

**Développé avec ❤️ pour le Sénégal 🇸🇳**

</div>
