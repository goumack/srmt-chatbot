#!/usr/bin/env python3
"""
Script pour réindexer les documents via l'API
"""
import requests
import json
import sys
import time

def check_server_status():
    """Vérifie si le serveur est en ligne"""
    # Essayer les ports communs
    ports = [8505, 5000, 8080]
    for port in ports:
        try:
            # Essayer health d'abord, sinon status
            for endpoint in ['/health', '/status', '/']:
                try:
                    response = requests.get(f'http://localhost:{port}{endpoint}', timeout=5)
                    if response.status_code == 200:
                        return port
                except:
                    continue
        except:
            continue
    return None

def reindex_smart(port=8505):
    """Réindexation intelligente (respecte le cache)"""
    print("🔄 Lancement de la réindexation intelligente...")
    try:
        response = requests.post(f'http://localhost:{port}/reindex', timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            print(f"📊 Fichiers indexés: {result['indexed_count']}")
            print(f"📁 Fichiers trouvés: {result['files_found']}")
            if 'files_list' in result:
                print(f"📄 Exemples: {', '.join(result['files_list'])}")
        else:
            print(f"❌ Erreur: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")

def reindex_force(port=8505):
    """Réindexation complète (efface tout le cache)"""
    print("🔄 Lancement de la réindexation COMPLÈTE (effacement du cache)...")
    try:
        response = requests.post(f'http://localhost:{port}/force_full_reindex', timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            print(f"📊 Fichiers indexés: {result['indexed_count']}")
            print(f"📁 Fichiers trouvés: {result['files_found']}")
            print(f"🗑️ Cache vidé: {result.get('cache_cleared', False)}")
        else:
            print(f"❌ Erreur: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")

def get_status(port=8505):
    """Affiche le statut du système"""
    try:
        response = requests.get(f'http://localhost:{port}/status', timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("📊 Statut du système:")
            print(f"   Fichiers indexés: {result.get('indexed_files_count', 'N/A')}")
            print(f"   Serveur: {result.get('status', 'N/A')}")
            print(f"   Version: {result.get('version', 'N/A')}")
        else:
            print(f"❌ Impossible d'obtenir le statut: {response.status_code}")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")

def main():
    """Menu principal"""
    print("🤖 LexFin - Outil de réindexation des documents")
    print("=" * 50)
    
    # Vérifier si le serveur est en ligne
    port = check_server_status()
    if not port:
        print("❌ Le serveur LexFin n'est pas accessible")
        print("   Ports testés: 8505, 5000, 8080")
        print("   Assurez-vous que l'application est démarrée")
        return
    
    print(f"✅ Serveur LexFin détecté sur le port {port}")
    
    if len(sys.argv) > 1:
        # Mode ligne de commande
        command = sys.argv[1].lower()
        if command == 'smart':
            reindex_smart(port)
        elif command == 'force':
            reindex_force(port)
        elif command == 'status':
            get_status(port)
        else:
            print("❌ Commande inconnue. Utilisez: smart, force, ou status")
    else:
        # Mode interactif
        while True:
            print("\nOptions disponibles:")
            print("1. Réindexation intelligente (recommandée)")
            print("2. Réindexation complète (efface tout)")
            print("3. Voir le statut")
            print("4. Quitter")
            
            choice = input("\nVotre choix (1-4): ").strip()
            
            if choice == '1':
                reindex_smart(port)
            elif choice == '2':
                reindex_force(port)
            elif choice == '3':
                get_status(port)
            elif choice == '4':
                print("👋 Au revoir!")
                break
            else:
                print("❌ Choix invalide")

if __name__ == "__main__":
    main()