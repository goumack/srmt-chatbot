# Dockerfile pour SRMT-DOCUMIND
# Assistant IA Spécialisé Fiscal et Douanier

FROM registry.access.redhat.com/ubi9/python-311

# Basculer en root pour installer les dépendances
USER 0

# Définir le répertoire de travail
WORKDIR /opt/app-root/src

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Installer les dépendances système nécessaires et SQLite >= 3.35.0
RUN yum update -y && \
    yum install -y gcc gcc-c++ make wget && \
    cd /tmp && \
    wget https://www.sqlite.org/2024/sqlite-autoconf-3460100.tar.gz && \
    tar xzf sqlite-autoconf-3460100.tar.gz && \
    cd sqlite-autoconf-3460100 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/sqlite-autoconf-3460100* && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/sqlite3.conf && \
    ldconfig && \
    yum clean all

# Configurer les variables d'environnement pour SQLite
ENV LD_LIBRARY_PATH=/usr/local/lib

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier tout le code de l'application
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p chroma_db documents static templates && \
    chown -R 1001:0 /opt/app-root/src && \
    chmod -R g=u /opt/app-root/src

# Revenir à l'utilisateur non-root
USER 1001

# Exposer le port de l'application
EXPOSE 8505

# Commande de démarrage
CMD ["python", "boutton memoire nouveau .py"]
