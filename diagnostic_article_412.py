"""
Script de diagnostic pour vérifier l'Article 412 dans ChromaDB
"""
import chromadb
from chromadb.config import Settings

# Connexion à ChromaDB
client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(name="alex_pro_docs")

print("=" * 80)
print("🔍 DIAGNOSTIC ARTICLE 412 DANS CHROMADB")
print("=" * 80)

# Récupérer TOUS les documents
all_docs = collection.get(include=['documents', 'metadatas'])

print(f"\n📊 Total documents dans ChromaDB: {len(all_docs['documents'])}")

# Chercher tous les Article 412
articles_412 = []
for idx, metadata in enumerate(all_docs['metadatas']):
    article_ref = metadata.get('article_ref', '')
    if '412' in article_ref:
        articles_412.append({
            'index': idx,
            'article_ref': article_ref,
            'file_name': metadata.get('file_name', ''),
            'page': metadata.get('page_start', '?'),
            'content': all_docs['documents'][idx]
        })

print(f"\n🔍 Nombre d'Article 412 trouvés: {len(articles_412)}")
print("=" * 80)

for i, article in enumerate(articles_412, 1):
    print(f"\n📄 ARTICLE 412 #{i}")
    print(f"Source: {article['file_name']}")
    print(f"Référence: {article['article_ref']}")
    print(f"Page: {article['page']}")
    print(f"\n📖 CONTENU (500 premiers caractères):")
    print(f"{article['content'][:500]}")
    print("-" * 80)
    
    # Vérifier si contient "vinaigrerie"
    if 'vinaigrerie' in article['content'].lower():
        print("✅ CONTIENT 'vinaigrerie'")
    else:
        print("❌ NE CONTIENT PAS 'vinaigrerie'")
    
    # Vérifier si contient "exonéré"
    if 'exonér' in article['content'].lower():
        print("✅ CONTIENT 'exonéré'")
    else:
        print("❌ NE CONTIENT PAS 'exonéré'")
    
    # Vérifier si contient "répressives" (Code Douanes)
    if 'répressives' in article['content'].lower() or 'repressive' in article['content'].lower():
        print("⚠️ CONTIENT 'répressives' (Code des Douanes - pénalités)")
    
    print("=" * 80)

# Chercher explicitement "vinaigrerie" dans tous les documents
print("\n\n🔍 RECHERCHE GLOBALE DE 'vinaigrerie' DANS TOUS LES DOCUMENTS")
print("=" * 80)

vinaigrerie_docs = []
for idx, doc in enumerate(all_docs['documents']):
    if 'vinaigrerie' in doc.lower():
        vinaigrerie_docs.append({
            'index': idx,
            'article_ref': all_docs['metadatas'][idx].get('article_ref', ''),
            'file_name': all_docs['metadatas'][idx].get('file_name', ''),
            'content': doc
        })

print(f"\n📊 Documents contenant 'vinaigrerie': {len(vinaigrerie_docs)}")

for i, doc in enumerate(vinaigrerie_docs, 1):
    print(f"\n📄 DOCUMENT #{i}")
    print(f"Source: {doc['file_name']}")
    print(f"Référence: {doc['article_ref']}")
    print(f"\n📖 CONTENU (500 premiers caractères):")
    print(f"{doc['content'][:500]}")
    print("=" * 80)

print(f"\n\n✅ Diagnostic terminé !")
