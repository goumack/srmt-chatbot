#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic : Rechercher "taxe sur les activités financières" dans ChromaDB
"""

import chromadb
from chromadb.config import Settings

# Connexion à ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("documents")

print("=" * 80)
print("🔍 DIAGNOSTIC : Taxe sur les activités financières")
print("=" * 80)

# Récupérer TOUS les documents
all_docs = collection.get(include=['metadatas', 'documents'])

print(f"\n📊 Total de chunks indexés : {len(all_docs['ids'])}")

# Recherche textuelle directe
keywords = ["activités financières", "taxe financière", "activité financière"]

print(f"\n🔎 Recherche textuelle des mots-clés : {keywords}")
print("-" * 80)

matches = []
for i, doc in enumerate(all_docs['documents']):
    doc_lower = doc.lower()
    metadata = all_docs['metadatas'][i]
    
    for keyword in keywords:
        if keyword.lower() in doc_lower:
            matches.append({
                'index': i,
                'id': all_docs['ids'][i],
                'keyword': keyword,
                'file': metadata.get('file_name', 'N/A'),
                'page': metadata.get('page', 'N/A'),
                'article_ref': metadata.get('article_ref', 'N/A'),
                'content': doc[:500]  # Premier 500 caractères
            })
            break  # Un seul match par document

print(f"\n✅ Trouvé {len(matches)} chunks contenant les mots-clés")

if matches:
    print("\n" + "=" * 80)
    print("📄 CHUNKS TROUVÉS:")
    print("=" * 80)
    
    for idx, match in enumerate(matches[:10], 1):  # Afficher les 10 premiers
        print(f"\n🔹 MATCH #{idx}")
        print(f"   Fichier    : {match['file']}")
        print(f"   Page       : {match['page']}")
        print(f"   Article    : {match['article_ref']}")
        print(f"   Mot-clé    : {match['keyword']}")
        print(f"   Contenu (début) :")
        print(f"   {match['content'][:300]}...")
        print("-" * 80)
else:
    print("\n❌ AUCUN chunk ne contient ces mots-clés !")
    print("\n🔎 Recherchons 'financière' seul...")
    
    for i, doc in enumerate(all_docs['documents']):
        if 'financière' in doc.lower():
            metadata = all_docs['metadatas'][i]
            print(f"\n📄 Trouvé dans :")
            print(f"   Fichier : {metadata.get('file_name', 'N/A')}")
            print(f"   Page    : {metadata.get('page', 'N/A')}")
            print(f"   Article : {metadata.get('article_ref', 'N/A')}")
            print(f"   Extrait : {doc[:300]}...")
            break

# Recherche sémantique avec embedding
print("\n" + "=" * 80)
print("🧠 RECHERCHE SÉMANTIQUE (via embedding)")
print("=" * 80)

query = "taux de la taxe sur les activités financières"
results = collection.query(
    query_texts=[query],
    n_results=10,
    include=['metadatas', 'documents', 'distances']
)

print(f"\nRequête : {query}")
print(f"Résultats : {len(results['ids'][0])}")

for idx, doc_id in enumerate(results['ids'][0], 1):
    metadata = results['metadatas'][0][idx-1]
    document = results['documents'][0][idx-1]
    distance = results['distances'][0][idx-1]
    
    print(f"\n🔹 RÉSULTAT #{idx} (distance: {distance:.4f})")
    print(f"   Fichier : {metadata.get('file_name', 'N/A')}")
    print(f"   Page    : {metadata.get('page', 'N/A')}")
    print(f"   Article : {metadata.get('article_ref', 'N/A')}")
    print(f"   Extrait : {document[:250]}...")

print("\n" + "=" * 80)
print("✅ Diagnostic terminé")
print("=" * 80)
