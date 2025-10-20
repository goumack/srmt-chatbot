import chromadb

# Connexion à ChromaDB
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection('alex_pro_docs')

# Recherche Article 408
print("\n" + "="*80)
print("RECHERCHE: Article 408 - Base imposable cigarettes")
print("="*80)

queries = [
    "Article 408 base imposable cigarette",
    "cigarette base imposable taxe spécifique",
    "Arrêté 019479 cigarette"
]

for query in queries:
    print(f"\n🔍 Query: {query}")
    print("-" * 80)
    
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    for i in range(min(3, len(results['documents'][0]))):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        
        print(f"\n📄 Résultat {i+1} (distance: {dist:.4f}):")
        print(f"   Fichier: {meta.get('source', 'N/A')}")
        print(f"   Page: {meta.get('page', 'N/A')}")
        print(f"   Contenu: {doc[:400]}...")
        print("-" * 80)

print("\n✅ Recherche terminée")
