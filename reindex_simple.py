#!/usr/bin/env python3
"""
Script simple de réindexation des documents LexFin
Mode standalone : utilise directement les fonctions sans serveur web
"""
import os
import shutil
import sys
from pathlib import Path

def force_reindex_standalone():
    """Force une réindexation complète en supprimant la base ChromaDB"""
    print("🔄 Début de la réindexation forcée (mode standalone)...")
    
    # Chemins vers les bases de données
    chroma_paths = [
        "./chroma_db",
        "./chroma_db_fiscal", 
        "./chroma_db_douanes"
    ]
    
    for path in chroma_paths:
        if os.path.exists(path):
            print(f"🗑️ Suppression de {path}...")
            try:
                shutil.rmtree(path)
                print(f"✅ {path} supprimé avec succès")
            except Exception as e:
                print(f"❌ Erreur lors de la suppression de {path}: {e}")
        else:
            print(f"ℹ️ {path} n'existe pas")
    
    print("✅ Réindexation forcée terminée")
    print("📝 Les documents seront réindexés automatiquement au prochain démarrage")
    
    return True

def check_documents_folder():
    """Vérifie le dossier documents et affiche les informations"""
    docs_folder = Path("./documents")
    
    if not docs_folder.exists():
        print("❌ Le dossier 'documents' n'existe pas")
        return False
    
    files = list(docs_folder.glob("*.pdf"))
    print(f"📁 Dossier documents: {len(files)} fichiers PDF trouvés")
    
    for file in files[:10]:  # Affiche les 10 premiers
        size = file.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"   📄 {file.name} ({size_mb:.1f} MB)")
    
    if len(files) > 10:
        print(f"   ... et {len(files) - 10} autres fichiers")
    
    return True

def check_chroma_status():
    """Vérifie l'état de la base ChromaDB"""
    chroma_path = Path("./chroma_db")
    
    if not chroma_path.exists():
        print("📊 Base ChromaDB: ❌ Non initialisée")
        return False
    
    # Vérifier la taille
    try:
        total_size = sum(f.stat().st_size for f in chroma_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"📊 Base ChromaDB: ✅ Initialisée ({size_mb:.1f} MB)")
        
        # Compter les fichiers
        db_files = list(chroma_path.rglob('*'))
        print(f"   📁 {len(db_files)} fichiers dans la base")
        return True
    except Exception as e:
        print(f"📊 Base ChromaDB: ⚠️ Erreur d'accès: {e}")
        return False

def show_reindex_help():
    """Affiche l'aide pour la réindexation"""
    print("""
🔧 GUIDE DE RÉINDEXATION LEXFIN 
================================

📖 Quand réindexer ?
   • Après ajout de nouveaux documents PDF
   • Si les recherches donnent des résultats incohérents  
   • Après modification des documents existants
   • Si la base ChromaDB semble corrompue

⚡ Options disponibles :
   1. Force reindex : Supprime complètement la base et force la réindexation
   2. Check status : Vérifie l'état des documents et de la base

🚀 Étapes après force reindex :
   1. Exécuter ce script avec l'option 'force'
   2. Redémarrer l'application LexFin
   3. L'indexation se fera automatiquement au démarrage

⚠️ ATTENTION : Force reindex supprime toute la base vectorielle !
""")

def main():
    """Menu principal"""
    print("🤖 LexFin - Outil de réindexation des documents (Mode Standalone)")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # Mode ligne de commande
        command = sys.argv[1].lower()
        if command == 'force':
            if check_documents_folder():
                force_reindex_standalone()
        elif command == 'status':
            check_documents_folder()
            check_chroma_status()
        elif command == 'help':
            show_reindex_help()
        else:
            print("❌ Commande inconnue. Utilisez: force, status, ou help")
    else:
        # Mode interactif
        while True:
            print("\nOptions disponibles:")
            print("1. Force reindex (supprime tout et force la réindexation)")
            print("2. Vérifier le statut")
            print("3. Aide / Documentation")
            print("4. Quitter")
            
            choice = input("\nVotre choix (1-4): ").strip()
            
            if choice == '1':
                print("\n⚠️ ATTENTION: Cette action va supprimer complètement la base vectorielle!")
                confirm = input("Êtes-vous sûr ? (tapez 'OUI' pour confirmer): ").strip()
                if confirm.upper() == 'OUI':
                    if check_documents_folder():
                        force_reindex_standalone()
                else:
                    print("❌ Opération annulée")
            elif choice == '2':
                print("\n📊 VÉRIFICATION DU STATUT")
                print("=" * 30)
                check_documents_folder()
                check_chroma_status()
            elif choice == '3':
                show_reindex_help()
            elif choice == '4':
                print("👋 Au revoir!")
                break
            else:
                print("❌ Choix invalide")

if __name__ == "__main__":
    main()