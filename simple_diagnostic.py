#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple diagnostic ChromaDB sans importer l'application"""

import chromadb
from chromadb.config import Settings

# Connexion directe à ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("alex_pro_docs")  # Nom correct de la collection

print("=" * 80)
print("🔍 RECHERCHE : Taxe sur les activités financières")
print("=" * 80)

# Récupérer tous les documents
all_docs = collection.get(include=['metadatas', 'documents'])
print(f"\n📊 Total chunks : {len(all_docs['ids'])}")

# Recherche textuelle
print("\n🔎 Recherche textuelle de 'activités financières'...")
matches = []

for i, doc in enumerate(all_docs['documents']):
    if 'activités financières' in doc.lower() or 'activite financiere' in doc.lower():
        metadata = all_docs['metadatas'][i]
        matches.append({
            'file': metadata.get('file_name', 'N/A'),
            'page': metadata.get('page', 'N/A'),
            'article': metadata.get('article_ref', 'N/A'),
            'content': doc
        })

print(f"✅ Trouvé {len(matches)} chunks")

if matches:
    for idx, m in enumerate(matches[:5], 1):
        print(f"\n📄 MATCH #{idx}")
        print(f"   Fichier : {m['file']}")
        print(f"   Page    : {m['page']}")
        print(f"   Article : {m['article']}")
        print(f"   Contenu : {m['content'][:400]}...")
        print("-" * 80)
else:
    print("\n❌ AUCUN chunk trouvé !")
    print("\n🔎 Recherchons 'financier' ou 'financière'...")
    
    count = 0
    for i, doc in enumerate(all_docs['documents']):
        if 'financier' in doc.lower():
            count += 1
            if count <= 3:
                metadata = all_docs['metadatas'][i]
                print(f"\n   Fichier : {metadata.get('file_name', 'N/A')}")
                print(f"   Page    : {metadata.get('page', 'N/A')}")
                print(f"   Extrait : {doc[:200]}...")
    
    print(f"\n   Total 'financier' : {count} chunks")

print("\n" + "=" * 80)
