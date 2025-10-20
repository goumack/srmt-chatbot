#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Afficher tous les articles 400-410 (section TAF)"""

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("alex_pro_docs")
all_docs = collection.get(include=['metadatas', 'documents'])

print("=" * 80)
print("📋 TOUS LES ARTICLES 400-410 (Section TAF)")
print("=" * 80)

# Chercher tous les articles de 400 à 410
for article_num in range(400, 411):
    article_name = f"Article {article_num}"
    
    for i, doc in enumerate(all_docs['documents']):
        metadata = all_docs['metadatas'][i]
        article_ref = metadata.get('article_ref', '')
        
        if article_ref and article_name in article_ref:
            print(f"\n{'='*80}")
            print(f"📄 {article_ref}")
            print(f"{'='*80}")
            print(doc)
            print("")

print("\n" + "=" * 80)
print("✅ Fin de la liste")
print("=" * 80)
