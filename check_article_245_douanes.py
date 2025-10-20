#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chromadb
import sys
import os
import requests
import json

def generate_embedding(text):
    """Génère un embedding avec le même modèle que l'application"""
    url = "https://ollamaaccel-chatbotaccel.apps.senum.heritage.africa/api/embeddings"
    payload = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"❌ Erreur embedding: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Erreur embedding: {e}")
        return None

def check_article_245():
    try:
        # Connexion à la base ChromaDB
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="alex_pro_docs")
        
        # Recherche spécifique Article 245 + transbordement + douanes
        print("=== RECHERCHE: Article 245 Code des Douanes ===")
        query_text = "Article 245 transbordement douanes conditions application régime"
        embedding = generate_embedding(query_text)
        
        if embedding:
            results1 = collection.query(
                query_embeddings=[embedding],
                n_results=5
            )
        else:
            print("❌ Impossible de générer l'embedding")
            return
        
        print("\n--- Résultats pour 'transbordement' ---")
        for i, doc in enumerate(results1['documents'][0]):
            meta = results1['metadatas'][0][i]
            distance = results1['distances'][0][i]
            print(f"\nDocument {i+1} (Distance: {distance:.3f})")
            print(f"Source: {meta.get('source', 'N/A')}")
            print(f"Page: {meta.get('page', 'N/A')}")
            print("Contenu:")
            # Chercher spécifiquement l'Article 245
            if "Article 245" in doc:
                start = doc.find("Article 245")
                end = doc.find("Article 246", start) if doc.find("Article 246", start) != -1 else start + 500
                article_content = doc[start:end].strip()
                print(article_content[:800] + "..." if len(article_content) > 800 else article_content)
            else:
                print(doc[:300] + "..." if len(doc) > 300 else doc)
            print("-" * 80)
        
        # Recherche alternative plus large
        print("\n=== RECHERCHE ALTERNATIVE: Code Douanes Article 245 ===")
        query_text2 = "Code des Douanes Article 245"
        embedding2 = generate_embedding(query_text2)
        
        if embedding2:
            results2 = collection.query(
                query_embeddings=[embedding2],
                n_results=5
            )
        else:
            print("❌ Impossible de générer l'embedding alternatif")
            return
        
        for i, doc in enumerate(results2['documents'][0]):
            meta = results2['metadatas'][0][i]
            distance = results2['distances'][0][i]
            source = meta.get('source', 'N/A')
            
            # Chercher uniquement dans les documents du Code des Douanes
            if "douane" in source.lower():
                print(f"\nDocument Douanes {i+1} (Distance: {distance:.3f})")
                print(f"Source: {source}")
                if "Article 245" in doc:
                    start = doc.find("Article 245")
                    end = doc.find("Article 246", start) if doc.find("Article 246", start) != -1 else start + 500
                    article_content = doc[start:end].strip()
                    print("ARTICLE 245 CODE DES DOUANES:")
                    print(article_content)
                    print("-" * 80)
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_article_245()