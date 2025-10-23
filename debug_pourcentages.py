#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug des pourcentages dans la base ChromaDB
Recherche tous les pourcentages pour trouver d'où vient le 17%
"""

import chromadb
import re
from chromadb.config import Settings

def debug_pourcentages():
    """Recherche tous les pourcentages dans la collection"""
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
        
        print(f"🔍 Analyse de {len(all_docs['documents'])} documents...")
        
        # Chercher tous les pourcentages
        pourcentages_trouvés = set()
        
        for i, doc in enumerate(all_docs['documents']):
            # Regex pour trouver les pourcentages
            percentages = re.findall(r'\b(\d{1,3})\s*%', doc)
            
            for pct in percentages:
                pourcentages_trouvés.add(pct)
                if pct in ['17', '18']:
                    print(f"\n📄 Document {i} contient {pct}%:")
                    # Extraire contexte autour du pourcentage
                    lines = doc.split('\n')
                    for line_num, line in enumerate(lines):
                        if f'{pct}%' in line:
                            print(f"   Ligne {line_num}: {line.strip()}")
                            # Contexte avant et après
                            if line_num > 0:
                                print(f"   Avant:  {lines[line_num-1].strip()}")
                            if line_num < len(lines) - 1:
                                print(f"   Après:  {lines[line_num+1].strip()}")
                            print("   " + "="*50)
        
        print(f"\n📊 Tous les pourcentages trouvés: {sorted(pourcentages_trouvés)}")
        
        # Recherche spécifique pour TVA
        print("\n🔎 Recherche spécifique 'TVA'...")
        tva_results = collection.query(
            query_texts=["TVA taux pourcentage"],
            n_results=10
        )
        
        for i, doc in enumerate(tva_results['documents'][0]):
            print(f"\n📄 Résultat TVA {i+1}:")
            # Chercher les pourcentages dans ce document
            percentages = re.findall(r'\b(\d{1,3})\s*%', doc)
            if percentages:
                print(f"   Pourcentages: {percentages}")
            
            # Montrer le contexte autour de TVA
            lines = doc.split('\n')
            for line in lines:
                if 'tva' in line.lower() or 'TVA' in line:
                    print(f"   TVA: {line.strip()}")
        
        print("\n✅ Analyse terminée")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pourcentages()