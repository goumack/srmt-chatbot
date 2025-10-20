#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Liste toutes les collections ChromaDB"""

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

print("=" * 80)
print("📋 Collections dans ChromaDB")
print("=" * 80)

collections = client.list_collections()

if collections:
    print(f"\n✅ Trouvé {len(collections)} collection(s) :")
    for col in collections:
        print(f"\n   📁 Nom : {col.name}")
        print(f"      ID  : {col.id}")
        count = col.count()
        print(f"      Docs: {count} chunks")
else:
    print("\n❌ Aucune collection trouvée !")

print("\n" + "=" * 80)
