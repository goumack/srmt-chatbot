#!/usr/bin/env python3
"""
Script pour forcer la réindexation complète avec les vrais numéros de page
"""
import os
import shutil
from pathlib import Path

def force_reindex():
    """Supprime la base de données ChromaDB pour forcer une réindexation"""
    print("🔄 Début de la réindexation forcée...")
    
    # Chemins vers les bases de données
    chroma_paths = [
        "./chroma_db",
        "./chroma_db_fiscal", 
        "./chroma_db_douanes"
    ]
    
    for path in chroma_paths:
        if os.path.exists(path):
            print(f"🗑️ Suppression de {path}...")
            try:
                shutil.rmtree(path)
                print(f"✅ {path} supprimé avec succès")
            except Exception as e:
                print(f"❌ Erreur lors de la suppression de {path}: {e}")
        else:
            print(f"ℹ️ {path} n'existe pas")
    
    print("✅ Réindexation forcée terminée")
    print("📝 Redémarrez l'application pour réindexer avec les vrais numéros de page")

if __name__ == "__main__":
    force_reindex()