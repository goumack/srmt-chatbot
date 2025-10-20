#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recherche du TAUX de la taxe sur les activités financières"""

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("alex_pro_docs")

print("=" * 80)
print("🔍 RECHERCHE DU TAUX : Taxe sur les activités financières")
print("=" * 80)

# Récupérer tous les documents
all_docs = collection.get(include=['metadatas', 'documents'])

# Recherche : "activités financières" + "taux" OU "%"
print("\n🔎 Recherche de chunks avec 'activités financières' + ('taux' OU '%')...")
matches = []

for i, doc in enumerate(all_docs['documents']):
    doc_lower = doc.lower()
    
    if ('activités financières' in doc_lower or 'activite financiere' in doc_lower):
        # Vérifier si contient aussi "taux" ou un pourcentage
        if 'taux' in doc_lower or '%' in doc or 'pour cent' in doc_lower or 'pourcent' in doc_lower:
            metadata = all_docs['metadatas'][i]
            matches.append({
                'file': metadata.get('file_name', 'N/A'),
                'page': metadata.get('page', 'N/A'),
                'article': metadata.get('article_ref', 'N/A'),
                'content': doc
            })

print(f"✅ Trouvé {len(matches)} chunks avec taux/pourcentage")

if matches:
    for idx, m in enumerate(matches, 1):
        print(f"\n{'='*80}")
        print(f"📄 MATCH #{idx}")
        print(f"   Fichier : {m['file']}")
        print(f"   Article : {m['article']}")
        print(f"\n📖 CONTENU COMPLET :")
        print(m['content'])
        print("="*80)
else:
    print("\n❌ Aucun chunk avec taux trouvé !")
    
    # Recherche alternative : juste le mot "taux" dans les articles TAF
    print("\n🔎 Recherchons tous les articles mentionnant 'activités financières'...")
    
    for i, doc in enumerate(all_docs['documents']):
        doc_lower = doc.lower()
        if 'activités financières' in doc_lower or 'activite financiere' in doc_lower:
            metadata = all_docs['metadatas'][i]
            print(f"\n   Article : {metadata.get('article_ref', 'N/A')}")
            print(f"   Extrait : {doc[:150]}...")

print("\n" + "=" * 80)
