# Dockerfile pour SRMT-DOCUMIND
# Assistant IA Spécialisé Fiscal et Douanier

FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier tout le code de l'application
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p chroma_db documents static templates

# Exposer le port de l'application
EXPOSE 8505

# Commande de démarrage
CMD ["python", "boutton memoire nouveau .py"]
