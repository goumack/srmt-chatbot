#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug spécifique TVA vs TAF
Analyser exactement ce que reçoit le modèle et pourquoi il confond
"""

import chromadb
import re
from chromadb.config import Settings

def debug_tva_vs_taf():
    """Analyser la confusion TVA vs TAF"""
    try:
        # Configuration ChromaDB
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        collection = client.get_collection("alex_pro_docs")
        
        # Récupérer tous les documents
        all_docs = collection.get()
        
        print("🔍 ANALYSE CRITIQUE TVA vs TAF")
        print("="*60)
        
        # Rechercher spécifiquement Article 369 et Article 404
        for i, doc in enumerate(all_docs['documents']):
            if "Article 369" in doc or "taux de la TVA est fixé" in doc:
                print(f"\n✅ ARTICLE 369 TROUVÉ - Document {i}:")
                lines = doc.split('\n')
                for line_num, line in enumerate(lines):
                    if 'TVA' in line or '18%' in line or '369' in line:
                        print(f"   >>> {line.strip()}")
                        if line_num > 0:
                            print(f"       Avant: {lines[line_num-1].strip()}")
                        if line_num < len(lines) - 1:
                            print(f"       Après: {lines[line_num+1].strip()}")
                print("-" * 40)
                        
            if "Article 404" in doc or "taxe sur les activités financières" in doc:
                print(f"\n❌ ARTICLE 404 TROUVÉ (CONFUSION!) - Document {i}:")
                lines = doc.split('\n')
                for line_num, line in enumerate(lines):
                    if 'financières' in line or '17%' in line or '404' in line:
                        print(f"   >>> {line.strip()}")
                        if line_num > 0:
                            print(f"       Avant: {lines[line_num-1].strip()}")
                        if line_num < len(lines) - 1:
                            print(f"       Après: {lines[line_num+1].strip()}")
                print("-" * 40)
        
        print("\n📊 SYNTHÈSE DU PROBLÈME:")
        print("✅ TVA (Taxe sur la Valeur Ajoutée) = 18% (Article 369)")
        print("❌ TAF (Taxe sur les Activités Financières) = 17% (Article 404)")
        print("🚨 LE MODÈLE CONFOND CES DEUX TAXES DIFFÉRENTES!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tva_vs_taf()