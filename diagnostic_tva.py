"""
Diagnostic: Recherche de l'article sur le taux de TVA dans ChromaDB
"""
import chromadb
import os
from pathlib import Path

# Configuration ChromaDB
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"

print("🔍 DIAGNOSTIC - Recherche de l'article sur le taux de TVA")
print("=" * 70)

# Initialiser ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

# Récupérer la collection
try:
    collection = client.get_collection(name="srmt_documents")
    print(f"✅ Collection trouvée: {collection.count()} documents")
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# Termes de recherche pour le taux de TVA
search_terms = [
    "taux de la TVA",
    "TVA 18%",
    "taux TVA 18",
    "Article 369",
    "taxe sur la valeur ajoutée 18",
]

print("\n📚 Recherche dans ChromaDB...")
print("-" * 70)

for term in search_terms:
    print(f"\n🔎 Recherche: '{term}'")
    
    results = collection.query(
        query_texts=[term],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    if results['documents'][0]:
        print(f"   ✅ {len(results['documents'][0])} résultats trouvés")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            file_name = metadata.get('file_name', 'N/A')
            page = metadata.get('page', 'N/A')
            
            print(f"\n   [{i}] Distance: {distance:.4f}")
            print(f"       Fichier: {file_name}")
            print(f"       Page: {page}")
            print(f"       Extrait: {doc[:200]}...")
            
            # Vérifier si ça contient "18%"
            if "18" in doc and ("%" in doc or "pour cent" in doc):
                print(f"       ⭐ CONTIENT '18%' !")
    else:
        print(f"   ❌ Aucun résultat")

# Recherche directe dans tous les documents
print("\n\n📊 Analyse complète de la collection...")
print("-" * 70)

# Récupérer tous les documents
all_docs = collection.get(
    include=["documents", "metadatas"]
)

print(f"\n✅ Total: {len(all_docs['documents'])} chunks indexés")

# Chercher ceux qui contiennent "18" et "TVA"
tva_18_docs = []
for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
    doc_lower = doc.lower()
    if ("18" in doc or "dix-huit" in doc_lower) and ("tva" in doc_lower or "taxe sur la valeur" in doc_lower):
        tva_18_docs.append({
            'index': i,
            'doc': doc,
            'metadata': metadata
        })

if tva_18_docs:
    print(f"\n🎯 {len(tva_18_docs)} documents contenant '18' ET 'TVA':")
    
    for item in tva_18_docs[:10]:  # Afficher les 10 premiers
        print(f"\n   📄 {item['metadata'].get('file_name', 'N/A')} - Page {item['metadata'].get('page', 'N/A')}")
        print(f"      {item['doc'][:300]}...")
else:
    print("\n⚠️ AUCUN document ne contient '18' ET 'TVA' ensemble !")
    print("   Cela signifie que l'article sur le taux de TVA n'est peut-être pas indexé correctement.")

print("\n" + "=" * 70)
print("✅ Diagnostic terminé")
