"""
SRMT-DOCUMIND - Assistant IA Spécialisé Fiscal et Douanier
Assistant IA intelligent pour les contribuables sénégalais
Spécialisé dans le Code des Impôts et le Code des Douanes du Sénégal
"""
import os
from flask import Flask, render_template_string, request, jsonify
import requests
import json
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import logging
import threading
import time
import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger la configuration
load_dotenv()

class SrmtDocumindConfig:
    """Configuration SRMT-DOCUMIND - Assistant Fiscal et Douanier"""
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollamaaccel-chatbotaccel.apps.senum.heritage.africa")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral:7b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    WATCH_DIRECTORY = os.getenv("WATCH_DIRECTORY", "./documents")  # Répertoire à surveiller
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md', '.json', '.csv', '.odt', '.xlsx', '.xls']

class DocumentWatcherHandler(FileSystemEventHandler):
    """Gestionnaire de surveillance automatique en arrière-plan"""
    
    def __init__(self, srmt_client):
        self.srmt_client = srmt_client
        self.processing_queue = []
        self.last_processed = {}
        super().__init__()
    
    def on_created(self, event):
        """Nouveau fichier créé - Indexation automatique en arrière-plan"""
        if not event.is_directory:
            file_path = event.src_path
            
        # Ignorer les fichiers temporaires
        if Path(file_path).name.startswith(('~$', '.')):
            return
            
        logger.info(f"📁 [AUTO] Nouveau fichier détecté: {Path(file_path).name}")
        
        # Traitement asynchrone en arrière-plan
        import threading
        def delayed_process():
            try:
                time.sleep(2)  # Attendre que le fichier soit complètement écrit
                if self.srmt_client.is_supported_file(file_path):
                    self.srmt_client.process_new_file_background(file_path)
                    logger.info(f" [AUTO] Fichier indexé automatiquement: {Path(file_path).name}")
                else:
                    logger.debug(f"⏭ [AUTO] Fichier ignoré (format non supporté): {Path(file_path).name}")
            except Exception as e:
                logger.error(f" [AUTO] Erreur indexation automatique {Path(file_path).name}: {e}")
        
        # Lancer en thread séparé pour ne pas bloquer le système
        thread = threading.Thread(target=delayed_process, daemon=True)
        thread.start()
    
    def on_modified(self, event):
        """Fichier modifié - Réindexation automatique si nécessaire"""
        if not event.is_directory:
            file_path = event.src_path
            
            # Ignorer les fichiers temporaires et éviter les doublons rapides
            if Path(file_path).name.startswith(('~$', '.')):
                return
                
            # Éviter le traitement en boucle (limitation par temps)
            current_time = time.time()
            if file_path in self.last_processed:
                if current_time - self.last_processed[file_path] < 5:  # 5 secondes minimum
                    return
            
            self.last_processed[file_path] = current_time
            logger.info(f"[AUTO] Modification détectée: {Path(file_path).name}")
            
            # Traitement asynchrone en arrière-plan
            import threading
            def delayed_reprocess():
                try:
                    time.sleep(1)  # Attendre la fin de l'écriture
                    if self.srmt_client.is_supported_file(file_path):
                        self.srmt_client.process_modified_file_background(file_path)
                        logger.info(f" [AUTO] Fichier réindexé automatiquement: {Path(file_path).name}")
                except Exception as e:
                    logger.error(f" [AUTO] Erreur réindexation automatique {Path(file_path).name}: {e}")
            
            # Lancer en thread séparé
            thread = threading.Thread(target=delayed_reprocess, daemon=True)
            thread.start()

class SrmtDocumindClient:
    """Client SRMT-DOCUMIND optimisé avec surveillance automatique pour la fiscalité et douanes sénégalaises"""
    
    def __init__(self):
        self.config = SrmtDocumindConfig()
        self.indexed_files = {}  # Cache des fichiers indexés {path: hash}
        self.observer = None  # Référence au watcher
        self.setup_chroma()
        self.setup_watch_directory()
        
        # Démarrer automatiquement la surveillance en arrière-plan
        surveillance_ok = False
        try:
            surveillance_ok = self.start_file_watcher()
        except Exception as e:
            logger.warning(f"  Surveillance automatique désactivée: {e}")
        
        if surveillance_ok:
            logger.info("   DOCUMIND initialisé - Surveillance automatique active")
        else:
            logger.info("   DOCUMIND initialisé - Mode manuel activé")
    
    def setup_chroma(self):
        """Initialise ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.chroma_client.get_collection("alex_documents")
            except:
                try:
                    self.collection = self.chroma_client.get_collection("alex_pro_docs")
                except:
                    self.collection = self.chroma_client.create_collection(
                        name="alex_pro_docs",
                        metadata={"description": "Documents ALEX"}
                    )
            
            # Charger la liste des fichiers déjà indexés
            self.load_indexed_files_cache()
            
        except Exception as e:
            logger.error(f"Erreur ChromaDB: {e}")
            self.collection = None
    
    def create_vector_store(self):
        """Crée une nouvelle collection ChromaDB"""
        try:
            # Supprimer l'ancienne collection si elle existe
            try:
                self.chroma_client.delete_collection("alex_documents")
            except:
                pass
            
            # Créer une nouvelle collection
            collection = self.chroma_client.create_collection(
                name="alex_documents",
                metadata={"hnsw:space": "cosine", "description": "Documents ALEX"}
            )
            self.collection = collection
            return collection
        except Exception as e:
            logger.error(f"Erreur création collection: {e}")
            return self.collection
    
    def setup_watch_directory(self):
        """Configure le répertoire à surveiller"""
        self.watch_dir = Path(self.config.WATCH_DIRECTORY)
        self.watch_dir.mkdir(exist_ok=True)
        logger.info(f"   Répertoire surveillé: {self.watch_dir.absolute()}")
    
    def start_file_watcher(self):
        """Démarre la surveillance automatique du répertoire avec redémarrage automatique"""
        try:
            if not self.watch_dir.exists():
                logger.warning(f"  Répertoire de surveillance introuvable: {self.watch_dir}")
                return False
                
            # Arrêter l'ancien observer s'il existe
            if hasattr(self, 'observer') and self.observer and self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            
            # Créer et démarrer le nouvel observer
            from watchdog.observers import Observer
            self.observer = Observer()
            handler = DocumentWatcherHandler(self)
            self.observer.schedule(handler, str(self.watch_dir), recursive=True)
            self.observer.daemon = True  # Thread daemon pour ne pas bloquer l'arrêt
            self.observer.start()
            
            logger.info(f"   [AUTO] Surveillance automatique active: {self.watch_dir}")
            logger.info("   [AUTO] Les nouveaux fichiers seront indexés automatiquement en arrière-plan")
            
            # Démarrer le monitoring de la surveillance
            self.start_watcher_monitor()
            
            # Scan initial en mode intelligent (respecte le cache)
            import threading
            def initial_scan():
                try:
                    time.sleep(1)  # Petite pause pour laisser le système s'initialiser
                    self.scan_existing_files()
                except Exception as e:
                    logger.warning(f"  [AUTO] Scan initial différé: {e}")
            
            # Scan initial en arrière-plan
            scan_thread = threading.Thread(target=initial_scan, daemon=True)
            scan_thread.start()
            
            return True
            
        except Exception as e:
            logger.warning(f"  Impossible de démarrer la surveillance automatique: {e}")
            logger.info("📚 Fonctionnement en mode manuel - utilisez les boutons pour indexer")
            self.observer = None
            return False

    def start_watcher_monitor(self):
        """Démarre un thread de surveillance pour redémarrer automatiquement l'observer si nécessaire"""
        import threading
        
        def monitor_watcher():
            """Surveille l'état de l'observer et le redémarre si nécessaire"""
            while True:
                try:
                    time.sleep(30)  # Vérifier toutes les 30 secondes
                    
                    # Vérifier si l'observer existe et fonctionne
                    if not hasattr(self, 'observer') or not self.observer:
                        logger.warning("   [AUTO] Observer non initialisé - Redémarrage...")
                        self.start_file_watcher()
                        continue
                    
                    if not self.observer.is_alive():
                        logger.warning("   [AUTO] Observer arrêté - Redémarrage automatique...")
                        self.start_file_watcher()
                        continue
                        
                except Exception as e:
                    logger.error(f"   [AUTO] Erreur monitoring surveillance: {e}")
                    try:
                        self.start_file_watcher()
                    except:
                        pass
                    
        # Lancer le monitoring en thread daemon
        monitor_thread = threading.Thread(target=monitor_watcher, daemon=True)
        monitor_thread.start()
    
    def load_indexed_files_cache(self):
        """Charge le cache des fichiers indexés depuis ChromaDB"""
        try:
            if self.collection:
                # Récupérer tous les documents avec leurs métadonnées
                results = self.collection.get(include=['metadatas'])
                if results and results['metadatas']:
                    for metadata in results['metadatas']:
                        if metadata and 'file_path' in metadata and 'file_hash' in metadata:
                            self.indexed_files[metadata['file_path']] = metadata['file_hash']
                logger.info(f"📚 Cache chargé: {len(self.indexed_files)} fichiers indexés")
        except Exception as e:
            logger.error(f"Erreur chargement cache: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Calcule le hash MD5 d'un fichier"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception as e:
            logger.error(f"Erreur calcul hash pour {file_path}: {e}")
            return ""
    
    def is_supported_file(self, file_path: str) -> bool:
        """Vérifie si le fichier est supporté et pas temporaire"""
        file_name = Path(file_path).name
        
        # Ignorer les fichiers temporaires
        if file_name.startswith('~$') or file_name.startswith('.'):
            return False
            
        return Path(file_path).suffix.lower() in self.config.SUPPORTED_EXTENSIONS
    
    def is_file_already_indexed(self, file_path: str) -> bool:
        """Vérifie si le fichier est déjà indexé (même contenu)"""
        if file_path not in self.indexed_files:
            return False
        
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.indexed_files.get(file_path, "")
        
        return current_hash == stored_hash and current_hash != ""
    
    def read_file_content(self, file_path: str) -> str:
        """Lit le contenu d'un fichier"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt' or file_ext == '.md':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2, ensure_ascii=False)
            
            elif file_ext == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_string()
            
            elif file_ext == '.odt':
                try:
                    # Essayer d'extraire le texte des fichiers ODT
                    import zipfile
                    import xml.etree.ElementTree as ET
                    
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        if 'content.xml' in zip_file.namelist():
                            content_xml = zip_file.read('content.xml')
                            root = ET.fromstring(content_xml)
                            
                            # Extraire tout le texte
                            text_content = []
                            for elem in root.iter():
                                if elem.text:
                                    text_content.append(elem.text.strip())
                                if elem.tail:
                                    text_content.append(elem.tail.strip())
                            
                            return ' '.join(filter(None, text_content))
                    return ""
                except Exception as odt_error:
                    logger.warning(f"Erreur lecture ODT {file_path}: {odt_error}")
                    return ""
            
            elif file_ext == '.pdf':
                try:
                    # Extraction PDF avec PyPDF2 ou pdfplumber
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            text_content = []
                            for page in pdf_reader.pages:
                                text_content.append(page.extract_text())
                            return '\n'.join(text_content)
                    except ImportError:
                        try:
                            import pdfplumber
                            with pdfplumber.open(file_path) as pdf:
                                text_content = []
                                for page in pdf.pages:
                                    text_content.append(page.extract_text() or '')
                                return '\n'.join(text_content)
                        except ImportError:
                            logger.warning(f"📄 PyPDF2 et pdfplumber non installés pour: {file_path}")
                            return f"Fichier PDF détecté: {Path(file_path).name} - Contenu non extractible"
                except Exception as pdf_error:
                    logger.error(f"Erreur extraction PDF {file_path}: {pdf_error}")
                    # Retourner au moins les métadonnées du fichier
                    return f"Document PDF: {Path(file_path).name} - Fichier détecté mais extraction échouée. Document disponible pour traitement."
            
            elif file_ext in ['.xlsx', '.xls']:
                try:
                    import pandas as pd
                    # Lire le fichier Excel avec limitation pour éviter les fichiers trop volumineux
                    excel_file = pd.ExcelFile(file_path)
                    text_content = []
                    total_rows_processed = 0
                    max_rows_per_file = 1000  # Limiter à 1000 lignes max par fichier Excel
                    
                    # Traiter chaque feuille avec limitation
                    for sheet_name in excel_file.sheet_names:
                        if total_rows_processed >= max_rows_per_file:
                            text_content.append(f"... [FICHIER TRONQUÉ - Plus de {max_rows_per_file} lignes] ...")
                            break
                            
                        text_content.append(f"=== Feuille: {sheet_name} ===")
                        
                        # Lire seulement les premières lignes de chaque feuille
                        rows_to_read = min(200, max_rows_per_file - total_rows_processed)
                        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=rows_to_read)
                        
                        if len(df) > 0:
                            # Résumé de la feuille plutôt que tout le contenu
                            text_content.append(f"Nombre de lignes: {len(df)}")
                            text_content.append(f"Colonnes: {', '.join(df.columns.astype(str))}")
                            
                            # Ajouter les premières lignes seulement
                            first_rows = df.head(10).to_string(index=False, na_rep='', max_cols=10)
                            text_content.append("Premières lignes:")
                            text_content.append(first_rows)
                            
                            total_rows_processed += len(df)
                        else:
                            text_content.append("Feuille vide")
                            
                        text_content.append("")  # Ligne vide entre les feuilles
                    
                    # Ajouter un résumé du fichier
                    summary = f"RÉSUMÉ FICHIER EXCEL: {Path(file_path).name}\n"
                    summary += f"Nombre de feuilles: {len(excel_file.sheet_names)}\n"
                    summary += f"Feuilles: {', '.join(excel_file.sheet_names)}\n"
                    summary += f"Lignes traitées: {total_rows_processed}\n"
                    
                    return summary + "\n" + '\n'.join(text_content)
                except ImportError:
                    logger.warning(f"📊 pandas/openpyxl non installés pour Excel: {file_path}")
                    return f"Fichier Excel détecté: {Path(file_path).name} - Installer pandas et openpyxl pour l'extraction"
                except Exception as excel_error:
                    logger.error(f"Erreur lecture Excel {file_path}: {excel_error}")
                    return f"Fichier Excel: {Path(file_path).name} - Erreur d'extraction: {str(excel_error)}"
            
            elif file_ext == '.docx':
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text_content = []
                    for paragraph in doc.paragraphs:
                        text_content.append(paragraph.text)
                    return '\n'.join(text_content)
                except ImportError:
                    logger.warning(f"📄 python-docx non installé pour: {file_path}")
                    return f"Fichier DOCX détecté: {Path(file_path).name} - Contenu non extractible"
                except Exception as docx_error:
                    logger.error(f"Erreur extraction DOCX {file_path}: {docx_error}")
                    return f"Fichier DOCX: {Path(file_path).name}"
            
            # Pour d'autres types de fichiers, essayer la lecture basique
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Erreur lecture fichier {file_path}: {e}")
            return ""
    
    def process_new_file(self, file_path: str):
        """Traite un nouveau fichier"""
        if not self.is_supported_file(file_path):
            logger.info(f"⏭️ Type de fichier non supporté: {file_path}")
            return
        
        if self.is_file_already_indexed(file_path):
            logger.info(f"✅ Fichier déjà indexé: {Path(file_path).name}")
            return
        
        self.index_file(file_path)
    
    def process_new_file_background(self, file_path: str):
        """Traite un nouveau fichier en arrière-plan (ne bloque pas le chatbot)"""
        try:
            if not self.is_supported_file(file_path):
                return
            
            if self.is_file_already_indexed(file_path):
                logger.debug(f"⏭️ [AUTO] Fichier déjà indexé: {Path(file_path).name}")
                return
            
            logger.info(f"🔄 [AUTO] Indexation en arrière-plan: {Path(file_path).name}")
            self.index_file(file_path)
            
        except Exception as e:
            logger.error(f"  [AUTO] Erreur traitement nouveau fichier {file_path}: {e}")
    
    def process_modified_file_background(self, file_path: str):
        """Traite un fichier modifié en arrière-plan"""
        try:
            if not self.is_supported_file(file_path):
                return
            
            current_hash = self.get_file_hash(file_path)
            stored_hash = self.indexed_files.get(file_path, "")
            
            if current_hash != stored_hash:
                logger.info(f"🔄 [AUTO] Réindexation automatique: {Path(file_path).name}")
                # Supprimer l'ancienne version
                self.remove_file_from_index(file_path)
                # Réindexer
                self.index_file(file_path)
            else:
                logger.debug(f"⏭️ [AUTO] Fichier inchangé: {Path(file_path).name}")
                
        except Exception as e:
            logger.error(f"  [AUTO] Erreur traitement fichier modifié {file_path}: {e}")
    
    def process_modified_file(self, file_path: str):
        """Traite un fichier modifié (version manuelle)"""
        if not self.is_supported_file(file_path):
            return
        
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.indexed_files.get(file_path, "")
        
        if current_hash != stored_hash:
            logger.info(f"🔄 Réindexation du fichier modifié: {Path(file_path).name}")
            # Supprimer l'ancienne version
            self.remove_file_from_index(file_path)
            # Réindexer
            self.index_file(file_path)
    
    def index_file(self, file_path: str):
        """Indexe un fichier dans ChromaDB avec positions précises"""
        try:
            content = self.read_file_content(file_path)
            if not content.strip():
                logger.warning(f"  Fichier vide: {file_path}")
                return
            
            # Découper le contenu en chunks avec positions
            chunks_with_positions = self.chunk_text_with_positions(content, file_path)
            if not chunks_with_positions:
                return
            
            logger.info(f"🔄 Indexation de {Path(file_path).name} ({len(chunks_with_positions)} chunks)")
            
            # Générer les embeddings en batch pour optimiser
            embeddings = []
            valid_chunks = []
            valid_positions = []
            
            # Traitement par petits groupes pour éviter les timeouts
            batch_size = 3
            for i in range(0, len(chunks_with_positions), batch_size):
                batch_chunks = chunks_with_positions[i:i + batch_size]
                
                for chunk_info in batch_chunks:
                    embedding = self.generate_embeddings(chunk_info['text'])
                    if embedding:
                        embeddings.append(embedding)
                        valid_chunks.append(chunk_info['text'])
                        valid_positions.append(chunk_info)
                
                # Petit délai entre les batches pour éviter de surcharger Ollama
                if i + batch_size < len(chunks_with_positions):
                    time.sleep(0.1)
            
            if not embeddings:
                logger.error(f"  Impossible de générer les embeddings pour: {file_path}")
                return
            
            # Ajouter à ChromaDB avec métadonnées de position
            file_hash = self.get_file_hash(file_path)
            file_name = Path(file_path).name
            
            ids = [f"{file_name}_{i}_{file_hash[:8]}" for i in range(len(valid_chunks))]
            metadatas = [
                {
                    "file_path": pos_info['file_path'],
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "start_pos": pos_info['start_pos'],
                    "end_pos": pos_info['end_pos'],
                    "line_start": pos_info['line_start'],
                    "line_end": pos_info['line_end'],
                    "page_start": pos_info['page_start'],
                    "page_end": pos_info['page_end'],
                    "article_ref": pos_info.get('article_ref', 'Section générale'),
                    "article_number": pos_info.get('article_number', ''),
                    "section": pos_info.get('section', ''),
                    "sous_section": pos_info.get('sous_section', ''),
                    "chapitre": pos_info.get('chapitre', ''),
                    "titre": pos_info.get('titre', '')
                }
                for i, pos_info in enumerate(valid_positions)
            ]
            
            self.collection.add(
                embeddings=embeddings,
                documents=valid_chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            # Mettre à jour le cache
            self.indexed_files[file_path] = file_hash
            
            logger.info(f"✅ Fichier indexé: {file_name} ({len(valid_chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"  Erreur indexation {file_path}: {e}")
    
    def remove_file_from_index(self, file_path: str):
        """Supprime un fichier de l'index"""
        try:
            # Trouver tous les documents de ce fichier
            results = self.collection.get(
                where={"file_path": file_path},
                include=['metadatas']
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Ancien index supprimé pour: {Path(file_path).name}")
                
        except Exception as e:
            logger.error(f"Erreur suppression index: {e}")
    
    def chunk_text_with_positions(self, text: str, file_path: str, chunk_size: int = 1500, overlap: int = 100) -> List[Dict]:
        """Découpe le texte en chunks avec positions précises pour références - Optimisé pour documents juridiques"""
        
        # Détecter si c'est un document juridique (Code des douanes, des impôts, etc.)
        is_legal_document = any(keyword in file_path.lower() for keyword in ['code', 'douane', 'impot', 'impôt', 'fiscal', 'loi', 'cgi', 'dgi'])
        
        if is_legal_document:
            return self.chunk_legal_document(text, file_path, chunk_size, overlap)
        
        # Découpage standard pour autres documents
        if len(text) <= chunk_size:
            return [{
                'text': text,
                'file_path': file_path,
                'start_pos': 0,
                'end_pos': len(text),
                'line_start': 1,
                'line_end': len(text.split('\n')),
                'page_start': 1,
                'page_end': 1,
                'article_ref': 'Document complet'
            }]
        
        chunks = []
        lines = text.split('\n')
        start = 0
        current_line = 1
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Essayer de couper à un point naturel (phrase)
            if end < len(text):
                last_sentence = max(
                    text.rfind('.', start, end),
                    text.rfind('\n', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                if last_sentence > start + 200:
                    end = last_sentence + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                # Calculer les numéros de lignes
                text_before = text[:start]
                text_chunk = text[start:end]
                
                line_start = text_before.count('\n') + 1
                line_end = line_start + text_chunk.count('\n')
                
                # Estimation de page (environ 50 lignes par page)
                page_start = max(1, (line_start - 1) // 50 + 1)
                page_end = max(1, (line_end - 1) // 50 + 1)
                
                chunks.append({
                    'text': chunk_text,
                    'file_path': file_path,
                    'start_pos': start,
                    'end_pos': end,
                    'line_start': line_start,
                    'line_end': line_end,
                    'page_start': page_start,
                    'page_end': page_end,
                    'article_ref': f'Section lignes {line_start}-{line_end}'
                })
            
            start = end - overlap if end < len(text) else end
        
        return chunks

    def chunk_legal_document(self, text: str, file_path: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        """Découpage spécialisé pour documents juridiques (codes) avec identification des articles"""
        import re
        
        chunks = []
        lines = text.split('\n')
        
        # Patterns pour identifier la structure hiérarchique du Code des Impôts et Douanes
        article_pattern = re.compile(r'^(Article\s*\d+(?:-\d+)?|Art\.\s*\d+(?:-\d+)?|ARTICLE\s*\d+(?:-\d+)?)', re.IGNORECASE)
        section_pattern = re.compile(r'^(SECTION\s*[IVX0-9]+|Section\s*[IVX0-9]+)', re.IGNORECASE)
        sous_section_pattern = re.compile(r'^(Sous-section\s*\d+|SOUS-SECTION\s*\d+)', re.IGNORECASE)
        chapitre_pattern = re.compile(r'^(CHAPITRE\s*[IVX0-9]+|Chapitre\s*[IVX0-9]+)', re.IGNORECASE)
        titre_pattern = re.compile(r'^(TITRE\s*[IVX0-9]+|Titre\s*[IVX0-9]+)', re.IGNORECASE)
        
        current_article = "Document"
        current_section = ""
        current_sous_section = ""
        current_chapitre = ""
        current_titre = ""
        
        i = 0
        while i < len(lines):
            chunk_lines = []
            chunk_start_line = i + 1
            article_ref = current_article
            
            # Détecter le début d'un nouvel élément hiérarchique
            line = lines[i].strip()
            
            if titre_pattern.match(line):
                current_titre = line
                current_chapitre = ""
                current_section = ""
                current_sous_section = ""
                current_article = line
                
            elif chapitre_pattern.match(line):
                current_chapitre = line
                current_section = ""
                current_sous_section = ""
                current_article = line
                
            elif section_pattern.match(line):
                current_section = line
                current_sous_section = ""
                current_article = line
                
            elif sous_section_pattern.match(line):
                current_sous_section = line
                current_article = line
                
            elif article_pattern.match(line):
                current_article = line
                article_ref = current_article
            
            # Collecter les lignes pour ce chunk
            chunk_size_lines = chunk_size // 50  # Estimation : ~50 caractères par ligne
            start_i = i
            
            while i < len(lines) and len('\n'.join(chunk_lines)) < chunk_size:
                current_line = lines[i].strip()
                
                # Arrêter si on trouve un nouvel article (sauf si on vient de commencer)
                if i > start_i and article_pattern.match(current_line):
                    break
                    
                chunk_lines.append(lines[i])
                i += 1
                
                # Limiter la taille pour éviter des chunks trop gros
                if len(chunk_lines) > chunk_size_lines:
                    break
            
            # Créer le chunk si on a du contenu
            if chunk_lines:
                chunk_text = '\n'.join(chunk_lines).strip()
                
                if chunk_text:
                    # Calculer les positions
                    text_before = '\n'.join(lines[:start_i])
                    start_pos = len(text_before) + (1 if text_before else 0)
                    end_pos = start_pos + len(chunk_text)
                    
                    line_start = start_i + 1
                    line_end = i
                    
                    page_start = max(1, (line_start - 1) // 50 + 1)
                    page_end = max(1, (line_end - 1) // 50 + 1)
                    
                    # Créer la référence hiérarchique complète
                    ref_parts = []
                    if current_titre:
                        ref_parts.append(current_titre)
                    if current_chapitre:
                        ref_parts.append(current_chapitre)
                    if current_section:
                        ref_parts.append(current_section)
                    if current_sous_section:
                        ref_parts.append(current_sous_section)
                    if current_article and current_article not in ref_parts:
                        ref_parts.append(current_article)
                    
                    full_ref = " > ".join(ref_parts) if ref_parts else article_ref
                    
                    chunks.append({
                        'text': chunk_text,
                        'file_path': file_path,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'line_start': line_start,
                        'line_end': line_end,
                        'page_start': page_start,
                        'page_end': page_end,
                        'article_ref': full_ref,
                        'article_number': current_article,
                        'section': current_section,
                        'sous_section': current_sous_section,
                        'chapitre': current_chapitre,
                        'titre': current_titre
                    })
            
            # Éviter les boucles infinies
            if i <= start_i:
                i += 1
        
        logger.info(f"📖 Document juridique découpé: {len(chunks)} articles/sections identifiés")
        return chunks

    def chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
        """Découpe le texte en chunks (version simplifiée pour compatibilité)"""
        chunks_with_pos = self.chunk_text_with_positions(text, "", chunk_size, overlap)
        return [chunk['text'] for chunk in chunks_with_pos]
    
    def scan_existing_files(self):
        """Scanne les fichiers existants au démarrage avec optimisations"""
        logger.info("� Scan optimisé des fichiers existants...")
        
        try:
            import concurrent.futures
            
            # Collecter tous les fichiers et classifier
            files_to_index = []
            already_indexed = []
            
            for file_path in self.watch_dir.rglob('*'):
                if file_path.is_file() and self.is_supported_file(str(file_path)):
                    if self.is_file_already_indexed(str(file_path)):
                        already_indexed.append(str(file_path))
                    else:
                        files_to_index.append(str(file_path))
            
            # Afficher le statut
            if already_indexed:
                logger.info(f"⏭️ {len(already_indexed)} fichiers déjà indexés (ignorés):")
                for file_path in already_indexed:
                    logger.info(f"   ⏭️ {Path(file_path).name}")
            
            if not files_to_index:
                logger.info("✅ Tous les fichiers sont déjà indexés - Aucun nouveau fichier à traiter")
                return
            
            logger.info(f"📚 Indexation de {len(files_to_index)} fichiers en parallèle...")
            
            # Traitement parallèle avec ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Soumettre tous les fichiers pour traitement
                future_to_file = {
                    executor.submit(self.index_file, file_path): file_path 
                    for file_path in files_to_index
                }
                
                # Collecter les résultats
                count = 0
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        future.result()
                        count += 1
                        if count % 5 == 0:  # Progress indicator
                            logger.info(f"⏳ {count}/{len(files_to_index)} fichiers traités...")
                    except Exception as e:
                        logger.error(f"  Erreur avec {file_path}: {e}")
            
            logger.info(f"🎯 {count} fichiers indexés avec succès!")
            
        except Exception as e:
            logger.warning(f"  Erreur lors du scan initial: {e}")
            logger.info("📚 Fallback: indexation séquentielle")
            # Fallback vers méthode séquentielle
            count = 0
            for file_path in self.watch_dir.rglob('*'):
                if file_path.is_file() and self.is_supported_file(str(file_path)):
                    if not self.is_file_already_indexed(str(file_path)):
                        self.index_file(str(file_path))
                        count += 1

    def generate_embeddings(self, text: str, max_retries: int = 2) -> List[float]:
        """Génère des embeddings avec optimisations et retry"""
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.config.OLLAMA_EMBEDDING_MODEL,
                    "prompt": text
                }
                
                # Timeout réduit et session réutilisable
                if not hasattr(self, '_session'):
                    self._session = requests.Session()
                    self._session.headers.update({'Connection': 'keep-alive'})
                
                # Timeout progressif selon l'essai
                timeout = 30 + (attempt * 15)  # 30s, 45s, 60s
                logger.info(f"🔄 Tentative {attempt + 1}/{max_retries + 1} embedding (timeout: {timeout}s)")
                
                response = self._session.post(
                    f"{self.config.OLLAMA_BASE_URL}/api/embeddings",
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Embedding généré avec succès (tentative {attempt + 1})")
                    return response.json()['embedding']
                else:
                    logger.warning(f"⚠️ Réponse HTTP {response.status_code} (tentative {attempt + 1})")
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"⏱️ Timeout tentative {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(2)  # Pause avant retry
                    continue
            except Exception as e:
                logger.error(f"❌ Erreur embedding (tentative {attempt + 1}): {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
        
        logger.error(f"💥 Échec génération embedding après {max_retries + 1} tentatives")
        return []
    
    def deduplicate_references(self, references: List[Dict]) -> List[Dict]:
        """Déduplique les références intelligemment pour éviter les répétitions"""
        if not references:
            return []
        
        # Grouper par fichier d'abord
        file_groups = {}
        for ref in references:
            file_name = ref.get('file_name', 'unknown')
            if file_name not in file_groups:
                file_groups[file_name] = []
            file_groups[file_name].append(ref)
        
        deduplicated = []
        
        for file_name, file_refs in file_groups.items():
            # Trier par page puis par ligne pour chaque fichier
            file_refs.sort(key=lambda x: (x.get('page_start', 1), x.get('line_start', 1)))
            
            # Fusionner les références proches dans le même fichier
            merged_refs = []
            current_ref = None
            
            for ref in file_refs:
                page_start = ref.get('page_start', 1)
                line_start = ref.get('line_start', 1)
                
                if current_ref is None:
                    current_ref = ref.copy()
                    merged_refs.append(current_ref)
                else:
                    # Vérifier si cette référence est proche de la précédente
                    prev_page = current_ref.get('page_start', 1)
                    prev_line = current_ref.get('line_start', 1)
                    
                    # Si même page et lignes proches (< 50 lignes d'écart), fusionner
                    if (page_start == prev_page and abs(line_start - prev_line) < 50) or \
                       (abs(page_start - prev_page) <= 1 and abs(line_start - prev_line) < 100):
                        
                        # Fusionner: étendre la zone et combiner les snippets
                        current_ref['line_end'] = max(current_ref.get('line_end', line_start), ref.get('line_end', line_start))
                        current_ref['page_end'] = max(current_ref.get('page_end', page_start), ref.get('page_end', page_start))
                        
                        # Mettre à jour la localisation
                        if current_ref['line_start'] == current_ref['line_end']:
                            current_ref['location'] = f"ligne {current_ref['line_start']}"
                        else:
                            current_ref['location'] = f"lignes {current_ref['line_start']}-{current_ref['line_end']}"
                            
                        if current_ref['page_start'] == current_ref['page_end']:
                            current_ref['page_info'] = f"page {current_ref['page_start']}"
                        else:
                            current_ref['page_info'] = f"pages {current_ref['page_start']}-{current_ref['page_end']}"
                        
                        # Garder le snippet le plus représentatif
                        if len(ref.get('snippet', '')) > len(current_ref.get('snippet', '')):
                            current_ref['snippet'] = ref['snippet']
                        
                        logger.debug(f"🔗 Références fusionnées dans {file_name}: {current_ref['location']}")
                    else:
                        # Nouvelle zone distincte
                        current_ref = ref.copy()
                        merged_refs.append(current_ref)
                        logger.debug(f"➕ Nouvelle référence dans {file_name}: {ref.get('location')}")
            
            deduplicated.extend(merged_refs)
        
        # Limiter à un nombre raisonnable par fichier (max 2 références par fichier)
        final_refs = []
        file_count = {}
        
        for ref in deduplicated:
            file_name = ref.get('file_name', 'unknown')
            file_count[file_name] = file_count.get(file_name, 0)
            
            if file_count[file_name] < 2:  # Max 2 références par fichier
                final_refs.append(ref)
                file_count[file_name] += 1
        
        logger.info(f"🔧 Déduplication intelligente: {len(references)} → {len(final_refs)} références optimisées")
        return final_refs

    def detect_query_domain(self, query: str) -> str:
        """Détecte si la question porte sur les impôts, les douanes, ou les deux"""
        query_lower = query.lower()
        
        # Mots-clés spécifiques aux impôts
        impots_keywords = [
            'impot', 'impôt', 'impots', 'impôts', 'fiscal', 'fiscale', 'fiscalité',
            'contribuable', 'contribuables', 'tva', 'is', 'ir', 'ircm', 'cgi',
            'déclaration de revenus', 'assiette fiscale', 'base imposable',
            'personne imposable', 'personnes imposables', 'assujetti', 'redevable',
            'déduction fiscale', 'exonération fiscale', 'crédit d\'impôt',
            # Termes juridiques des sociétés (domaine fiscal)
            'société', 'sociétés', 'societe', 'societes',
            'société par actions', 'société à responsabilité limitée', 
            'sarl', 'sa', 'sas', 'société anonyme',
            'capital social', 'actionnaire', 'actionnaires', 'associé', 'associés',
            'bénéfice', 'bénéfices', 'résultat fiscal', 'impôt sur les sociétés',
            'entreprise', 'entreprises', 'personne morale', 'personnes morales'
        ]
        
        # Mots-clés spécifiques aux douanes
        douanes_keywords = [
            'douane', 'douanes', 'douanier', 'douanière', 'dédouanement',
            'importation', 'exportation', 'marchandise', 'marchandises',
            'bureau de douane', 'aéroport douanier', 'port douanier',
            'transit', 'droit de douane', 'tarif douanier', 'nomenclature',
            'espèce d\'une marchandise', 'classement douanier', 'origine des marchandises'
        ]
        
        # Compter les occurrences avec pondération
        impots_score = 0
        douanes_score = 0
        
        # Termes à fort poids pour les impôts (sociétés, fiscal)
        high_weight_impots = ['société', 'sociétés', 'societe', 'societes', 'sarl', 'sa', 'sas', 
                              'société par actions', 'société à responsabilité limitée', 'impôt sur les sociétés',
                              'capital social', 'actionnaire', 'bénéfice', 'résultat fiscal']
        
        # Termes à fort poids pour les douanes
        high_weight_douanes = ['marchandise', 'marchandises', 'dédouanement', 'importation', 'exportation',
                               'bureau de douane', 'tarif douanier', 'espèce d\'une marchandise']
        
        # Calculer les scores avec pondération
        for keyword in impots_keywords:
            if keyword in query_lower:
                weight = 3 if keyword in high_weight_impots else 1
                impots_score += weight
                
        for keyword in douanes_keywords:
            if keyword in query_lower:
                weight = 3 if keyword in high_weight_douanes else 1
                douanes_score += weight
        
        # Déterminer le domaine
        if impots_score > douanes_score:
            logger.info(f"🏛️ Question détectée comme FISCALE/IMPÔTS (score: {impots_score} vs {douanes_score})")
            return "impots"
        elif douanes_score > impots_score:
            logger.info(f"🚢 Question détectée comme DOUANIÈRE (score: {douanes_score} vs {impots_score})")
            return "douanes"
        else:
            logger.info(f"🔄 Question GÉNÉRALE ou ambiguë (impôts: {impots_score}, douanes: {douanes_score})")
            return "general"

    def execute_jupyter_code(self, code: str, query: str) -> Dict:
        """Exécute du code Python/Jupyter et retourne les résultats"""
        try:
            import pandas as pd
            import numpy as np
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            # Détecter quelle source de données utiliser
            data_source = self.detect_data_source(query)
            
            # Chargement des données selon la source
            if data_source == "contribuable":
                contribuable_path = os.path.join(self.watch_dir, "contribuable.csv")
                if not os.path.exists(contribuable_path):
                    return {"error": "Fichier contribuable.csv non trouvé"}
                try:
                    df = pd.read_csv(contribuable_path, sep=';')
                except Exception as e:
                    return {"error": f"Erreur lecture contribuable.csv: {str(e)}"}
            else:
                declaration_path = os.path.join(self.watch_dir, "declaration.csv")
                if not os.path.exists(declaration_path):
                    return {"error": "Fichier declaration.csv non trouvé"}
                try:
                    df = pd.read_csv(declaration_path)
                except Exception as e:
                    return {"error": f"Erreur lecture declaration.csv: {str(e)}"}
            
            # Créer un environnement d'exécution sécurisé
            namespace = {
                'pd': pd,
                'np': np,
                'df': df,
                'query': query,
                'format_montant': self.format_montant,
                'data_source': data_source
            }
            
            # Capturer les sorties
            output = io.StringIO()
            error_output = io.StringIO()
            
            try:
                with redirect_stdout(output), redirect_stderr(error_output):
                    # Exécuter le code
                    exec(code, namespace)
                
                # Récupérer les résultats
                stdout_result = output.getvalue()
                stderr_result = error_output.getvalue()
                
                if stderr_result:
                    logger.warning(f"⚠️ Avertissements lors de l'exécution: {stderr_result}")
                
                # Chercher des variables de résultat dans le namespace
                result_vars = {}
                for key, value in namespace.items():
                    if key not in ['pd', 'np', 'df', 'query', 'format_montant', '__builtins__']:
                        if not key.startswith('_'):
                            result_vars[key] = value
                
                logger.info(f"✅ Code Jupyter exécuté avec succès")
                return {
                    "stdout": stdout_result,
                    "variables": result_vars,
                    "success": True
                }
                
            except Exception as exec_error:
                logger.error(f"❌ Erreur d'exécution: {exec_error}")
                return {
                    "error": f"Erreur d'exécution: {str(exec_error)}",
                    "stderr": error_output.getvalue(),
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur setup Jupyter: {e}")
            return {"error": f"Erreur setup: {str(e)}", "success": False}

    def format_montant(self, montant):
        """Formate un montant en FCFA avec les bonnes unités"""
        import pandas as pd
        if pd.isna(montant) or montant == 0:
            return "0 FCFA"
        
        if montant >= 1_000_000_000:
            return f"{montant/1_000_000_000:.1f} milliards FCFA"
        elif montant >= 1_000_000:
            return f"{montant/1_000_000:.1f} millions FCFA"
        elif montant >= 1_000:
            return f"{montant/1_000:.1f}K FCFA"
        else:
            return f"{montant:,.0f} FCFA"

    def detect_data_source(self, query: str) -> str:
        """Détecte quel fichier de données utiliser selon la requête"""
        query_lower = query.lower()
        
        # Mots-clés pour le fichier contribuable.csv
        contribuable_keywords = [
            'personne physique', 'personne morale', 'contribuable', 'recouvrement', 
            'montant recouvre', 'montant declare', 'type_contribuable', 'bureau',
            'csf', 'dgid', 'direction', 'impot sur', 'taxe sur'
        ]
        
        # Mots-clés pour le fichier declaration.csv  
        declaration_keywords = [
            'declaration', 'chiffre affaires', 'tva', 'secteur', 'ville',
            'statut declaration', 'valide', 'en cours', 'ca total'
        ]
        
        # Compter les correspondances
        contribuable_score = sum(1 for keyword in contribuable_keywords if keyword in query_lower)
        declaration_score = sum(1 for keyword in declaration_keywords if keyword in query_lower)
        
        # Décision basée sur le score
        if contribuable_score > declaration_score:
            return "contribuable"
        elif declaration_score > contribuable_score:
            return "declaration"
        else:
            # Par défaut, utiliser declaration.csv sauf si mention explicite
            if any(word in query_lower for word in ['personne physique', 'personne morale', 'recouvrement']):
                return "contribuable"
            return "declaration"

    def generate_statistics_code(self, query: str) -> str:
        """Génère du code Python adapté à la requête utilisateur"""
        query_lower = query.lower()
        data_source = self.detect_data_source(query)
        
        # Code de base selon la source de données
        if data_source == "contribuable":
            base_code = '''
# Chargement et préparation des données contribuables
print("📊 ANALYSE DES CONTRIBUABLES - RECOUVREMENT & DÉCLARATIONS")
print("=" * 60)
try:
    df = pd.read_csv('documents/contribuable.csv', sep=';')
    print(f"📋 Données chargées: {len(df)} contribuables")
except:
    print("❌ Erreur chargement contribuable.csv")
    df = pd.DataFrame()
print()

'''
        else:
            base_code = '''
# Chargement et préparation des données déclarations
print("📊 ANALYSE DES DÉCLARATIONS FISCALES 2023")
print("=" * 50)
print(f"📋 Données chargées: {len(df)} entreprises")
print()

'''
        
        # Code spécifique selon la requête et la source
        if data_source == "contribuable":
            # Codes pour analyser contribuable.csv
            if any(word in query_lower for word in ['personne physique', 'physique']):
                code = base_code + '''
# Analyse spécifique des Personnes Physiques
if len(df) > 0:
    personnes_physiques = df[df['type_contribuable'] == 'Personne Physique'].copy()
    personnes_morales = df[df['type_contribuable'] == 'Personne Morale'].copy()
    
    print("👤 ANALYSE DES PERSONNES PHYSIQUES:")
    print(f"📊 Nombre: {len(personnes_physiques)} sur {len(df)} ({len(personnes_physiques)/len(df)*100:.1f}%)")
    
    if len(personnes_physiques) > 0:
        montant_declare_pp = personnes_physiques['montant_declare'].sum()
        montant_recouvre_pp = personnes_physiques['montant_recouvre'].sum()
        taux_recouvrement_pp = (montant_recouvre_pp / montant_declare_pp * 100) if montant_declare_pp > 0 else 0
        
        print("\\n💰 RECOUVREMENT PERSONNES PHYSIQUES:")
        print(f"• Montant déclaré: {format_montant(montant_declare_pp)}")
        print(f"• Montant recouvré: {format_montant(montant_recouvre_pp)}")
        print(f"• Taux de recouvrement: {taux_recouvrement_pp:.1f}%")
        print(f"• Montant moyen déclaré: {format_montant(personnes_physiques['montant_declare'].mean())}")
        
        print("\\n🏛️ TYPES D'IMPÔTS (TOP 5):")
        impots_pp = personnes_physiques['nom_taxe_impot'].value_counts().head(5)
        for impot, count in impots_pp.items():
            print(f"• {impot[:40]}: {count}")
            
        print("\\n🌍 BUREAUX (TOP 5):")
        bureaux_pp = personnes_physiques['nom_bureau_csf'].value_counts().head(5)
        for bureau, count in bureaux_pp.items():
            print(f"• {bureau}: {count}")
    else:
        print("⚠️ Aucune Personne Physique trouvée")
else:
    print("❌ Aucune donnée disponible")
'''
            
            elif any(word in query_lower for word in ['recouvrement', 'recouvre']):
                code = base_code + '''
# Analyse globale du recouvrement
if len(df) > 0:
    montant_total_declare = df['montant_declare'].sum()
    montant_total_recouvre = df['montant_recouvre'].sum() 
    taux_global = (montant_total_recouvre / montant_total_declare * 100) if montant_total_declare > 0 else 0
    
    print("💸 ANALYSE GLOBALE DU RECOUVREMENT:")
    print(f"• Total déclaré: {format_montant(montant_total_declare)}")
    print(f"• Total recouvré: {format_montant(montant_total_recouvre)}")
    print(f"• Taux de recouvrement global: {taux_global:.1f}%")
    print(f"• Écart de recouvrement: {format_montant(montant_total_declare - montant_total_recouvre)}")
    
    # Par type de contribuable
    pp_data = df[df['type_contribuable'] == 'Personne Physique']
    pm_data = df[df['type_contribuable'] == 'Personne Morale']
    
    if len(pp_data) > 0:
        pp_declare = pp_data['montant_declare'].sum()
        pp_recouvre = pp_data['montant_recouvre'].sum()
        pp_taux = (pp_recouvre / pp_declare * 100) if pp_declare > 0 else 0
        print(f"\\n👤 PERSONNES PHYSIQUES:")
        print(f"• Déclaré: {format_montant(pp_declare)} | Recouvré: {format_montant(pp_recouvre)}")
        print(f"• Taux: {pp_taux:.1f}% | {len(pp_data)} déclarations")
    
    if len(pm_data) > 0:
        pm_declare = pm_data['montant_declare'].sum()
        pm_recouvre = pm_data['montant_recouvre'].sum()
        pm_taux = (pm_recouvre / pm_declare * 100) if pm_declare > 0 else 0
        print(f"\\n🏢 PERSONNES MORALES:")
        print(f"• Déclaré: {format_montant(pm_declare)} | Recouvré: {format_montant(pm_recouvre)}")
        print(f"• Taux: {pm_taux:.1f}% | {len(pm_data)} déclarations")
        
    # Top 10 plus gros recouvrements
    print(f"\\n🏆 TOP 10 PLUS GROS RECOUVREMENTS:")
    top_recouvre = df.nlargest(10, 'montant_recouvre')
    for i, row in top_recouvre.iterrows():
        print(f"{list(top_recouvre.index).index(i)+1:2d}. {row['id_contribuable'][:15]} | {format_montant(row['montant_recouvre'])} | {row['type_contribuable']}")
else:
    print("❌ Aucune donnée disponible")
'''
            
            else:
                # Statistiques générales contribuables
                code = base_code + '''
# Statistiques générales contribuables
if len(df) > 0:
    pp_count = len(df[df['type_contribuable'] == 'Personne Physique'])
    pm_count = len(df[df['type_contribuable'] == 'Personne Morale'])
    
    print("🏢 VUE D'ENSEMBLE CONTRIBUABLES:")
    print(f"• Total contribuables: {len(df)}")
    print(f"• Personnes Physiques: {pp_count} ({pp_count/len(df)*100:.1f}%)")
    print(f"• Personnes Morales: {pm_count} ({pm_count/len(df)*100:.1f}%)")
    
    print("\\n💰 MONTANTS GLOBAUX:")
    total_declare = df['montant_declare'].sum()
    total_recouvre = df['montant_recouvre'].sum()
    print(f"• Total déclaré: {format_montant(total_declare)}")
    print(f"• Total recouvré: {format_montant(total_recouvre)}")
    print(f"• Taux de recouvrement: {(total_recouvre/total_declare*100):.1f}%")
    
    print("\\n🏛️ TOP 5 TYPES D'IMPÔTS:")
    impots = df['nom_taxe_impot'].value_counts().head(5)
    for impot, count in impots.items():
        print(f"• {impot[:50]}: {count}")
        
    print("\\n🌍 TOP 5 BUREAUX:")
    bureaux = df['nom_bureau_csf'].value_counts().head(5)
    for bureau, count in bureaux.items():
        print(f"• {bureau}: {count}")
else:
    print("❌ Aucune donnée disponible")
'''
        
        else:
            # Codes pour declaration.csv (code existant)
            if any(word in query_lower for word in ['statistiques', 'résumé', 'général', 'overview']):
                code = base_code + '''
# Statistiques générales
total_entreprises = len(df)
validées = len(df[df['Statut_Declaration'] == 'Validée'])
en_cours = len(df[df['Statut_Declaration'] == 'En cours'])
ca_total = df['Chiffre_Affaires_2023'].sum()
tva_totale = df['TVA_Due'].sum()
ca_moyen = df['Chiffre_Affaires_2023'].mean()

print("🏢 VUE D'ENSEMBLE:")
print(f"• Total entreprises: {total_entreprises}")
print(f"• Déclarations validées: {validées} ({validées/total_entreprises*100:.1f}%)")
print(f"• Déclarations en cours: {en_cours}")
print()
print("💰 INDICATEURS FINANCIERS:")
print(f"• CA total: {format_montant(ca_total)}")
print(f"• TVA totale: {format_montant(tva_totale)}")
print(f"• CA moyen: {format_montant(ca_moyen)}")
print()
print("🌍 RÉPARTITION:")
print(f"• {df['Secteur'].nunique()} secteurs d'activité")
print(f"• {df['Ville'].nunique()} villes")
'''
            
            elif any(word in query_lower for word in ['secteur', 'activité', 'domaine']):
                code = base_code + '''
# Analyse par secteur
print("🏭 ANALYSE PAR SECTEUR:")
print()
secteurs = df['Secteur'].value_counts()
ca_secteur = df.groupby('Secteur')['Chiffre_Affaires_2023'].sum().sort_values(ascending=False)
tva_secteur = df.groupby('Secteur')['TVA_Due'].sum().sort_values(ascending=False)

print("📊 Nombre d'entreprises par secteur:")
for secteur, count in secteurs.items():
    print(f"• {secteur}: {count} entreprises")

print()
print("💰 Chiffre d'affaires par secteur:")
for secteur, ca in ca_secteur.items():
    print(f"• {secteur}: {format_montant(ca)}")

print()
print("💸 TVA par secteur:")
for secteur, tva in tva_secteur.items():
    print(f"• {secteur}: {format_montant(tva)}")
'''
        
        elif any(word in query_lower for word in ['ville', 'géographique', 'dakar', 'région']):
            code = base_code + '''
# Analyse géographique
print("🌍 ANALYSE GÉOGRAPHIQUE:")
print()
villes = df['Ville'].value_counts()
ca_ville = df.groupby('Ville')['Chiffre_Affaires_2023'].sum().sort_values(ascending=False)

print("🏙️ Entreprises par ville:")
for ville, count in villes.items():
    pourcentage = count/len(df)*100
    print(f"• {ville}: {count} entreprises ({pourcentage:.1f}%)")

print()
print("💰 CA par ville:")
for ville, ca in ca_ville.items():
    print(f"• {ville}: {format_montant(ca)}")
'''
        
        elif any(word in query_lower for word in ['top', 'meilleur', 'plus gros', 'classement']):
            code = base_code + '''
# Top contributeurs
print("🏆 TOP CONTRIBUTEURS:")
print()
top_ca = df.nlargest(5, 'Chiffre_Affaires_2023')
top_tva = df.nlargest(5, 'TVA_Due')

print("💰 TOP 5 CHIFFRES D'AFFAIRES:")
for i, row in top_ca.iterrows():
    print(f"{list(top_ca.index).index(i)+1}. {row['Contribuable']} - {format_montant(row['Chiffre_Affaires_2023'])}")

print()
print("💸 TOP 5 TVA:")
for i, row in top_tva.iterrows():
    print(f"{list(top_tva.index).index(i)+1}. {row['Contribuable']} - {format_montant(row['TVA_Due'])}")
'''
        
        elif any(word in query_lower for word in ['tva', 'taxe']):
            code = base_code + '''
# Analyse TVA
print("� ANALYSE TVA:")
print()
tva_totale = df['TVA_Due'].sum()
tva_moyenne = df['TVA_Due'].mean()
tva_secteur = df.groupby('Secteur')['TVA_Due'].sum().sort_values(ascending=False)

print(f"💰 TVA totale collectée: {format_montant(tva_totale)}")
print(f"📊 TVA moyenne: {format_montant(tva_moyenne)}")
print()
print("🏭 TVA par secteur:")
for secteur, tva in tva_secteur.items():
    print(f"• {secteur}: {format_montant(tva)}")
'''
        
        else:
            # Code par défaut - statistiques de base
            code = base_code + '''
# Analyse générale
total = len(df)
ca_total = df['Chiffre_Affaires_2023'].sum()
tva_totale = df['TVA_Due'].sum()
validées = len(df[df['Statut_Declaration'] == 'Validée'])

print("📈 RÉSULTATS:")
print(f"• {total} entreprises analysées")
print(f"• CA total: {format_montant(ca_total)}")
print(f"• TVA totale: {format_montant(tva_totale)}")
print(f"• {validées} déclarations validées ({validées/total*100:.1f}%)")

# Variables pour l'application
result_summary = f"Analyse de {total} entreprises - CA: {format_montant(ca_total)}, TVA: {format_montant(tva_totale)}"
'''
        
        return code



    def search_context_with_references(self, query: str, limit: int = 5, declaration_mode: bool = False) -> Dict:
        """Recherche le contexte avec références précises et déduplication intelligente"""
        if not self.collection:
            logger.warning("  Aucune collection ChromaDB disponible")
            return {"context": "", "references": []}
        
        try:
            # Initialisation par défaut
            query_domain = None
            
            # Mode Déclaration activé - Exécution de code Jupyter pour les statistiques
            if declaration_mode:
                logger.info("📋 MODE DÉCLARATION ACTIVÉ - Exécution de code Jupyter")
                
                # Générer le code Python adapté à la requête
                python_code = self.generate_statistics_code(query)
                
                # Exécuter le code Python/Jupyter
                execution_result = self.execute_jupyter_code(python_code, query)
                
                if execution_result.get("success", False):
                    # Récupérer les résultats de l'exécution
                    stdout_output = execution_result.get("stdout", "")
                    variables = execution_result.get("variables", {})
                    
                    # Créer un contexte basé sur les résultats d'exécution
                    context = f"""ANALYSE STATISTIQUE EXÉCUTÉE - MODE DÉCLARATION

🔬 CODE PYTHON EXÉCUTÉ:
{python_code.strip()}

📊 RÉSULTATS D'EXÉCUTION:
{stdout_output}

💾 VARIABLES GÉNÉRÉES:
"""
                    
                    # Ajouter les variables générées
                    for var_name, var_value in variables.items():
                        if isinstance(var_value, (int, float)):
                            if var_name.endswith(('_total', '_totale')):
                                context += f"- {var_name}: {self.format_montant(var_value)}\n"
                            else:
                                context += f"- {var_name}: {var_value:,}\n"
                        elif isinstance(var_value, str):
                            context += f"- {var_name}: {var_value}\n"
                        else:
                            context += f"- {var_name}: {str(var_value)[:100]}...\n"
                    
                    context += f"""

✅ Analyse terminée avec succès. Ces résultats sont calculés dynamiquement à partir du fichier declaration.csv.
🎯 Vous pouvez poser des questions plus spécifiques pour des analyses détaillées.

DONNÉES SOURCE: documents/declaration.csv
MÉTHODE: Exécution Python/Jupyter en temps réel
FIABILITÉ: ✅ Calculs directs sur données brutes"""
                    
                    references = ["🔬 Code Python exécuté avec succès", "📊 Résultats calculés dynamiquement"]
                    
                    logger.info("✅ Code Jupyter exécuté et contexte généré")
                    
                else:
                    # Erreur d'exécution - fallback
                    error_msg = execution_result.get("error", "Erreur inconnue")
                    logger.error(f"❌ Erreur exécution Jupyter: {error_msg}")
                    
                    context = f"""⚠️ ERREUR LORS DE L'EXÉCUTION DU CODE STATISTIQUE

❌ Erreur: {error_msg}

📋 Mode déclaration activé mais impossible d'exécuter les calculs automatiques.
🔄 Basculement vers recherche vectorielle de base...

Veuillez vérifier:
• Le fichier declaration.csv existe dans documents/
• Les données sont au bon format
• Les colonnes attendues sont présentes"""
                    
                    references = ["❌ Erreur d'exécution du code Python"]
                
                return {
                    "context": context,
                    "references": references
                }
            else:
                # Mode normal - Détecter le domaine de la question
                query_domain = self.detect_query_domain(query)
                
                # Recherche spécialisée pour les articles (seulement en mode normal)
                article_result = self.search_specific_article(query)
                if article_result["context"]:
                    logger.info(f"🎯 Article spécifique trouvé: {query}")
                    return article_result
            
            # Générer embedding de la requête
            query_embedding = self.generate_embeddings(query)
            if not query_embedding:
                logger.warning("  Impossible de générer embedding pour la requête")
                return {"context": "", "references": []}
            
            # Rechercher dans ChromaDB avec filtre par domaine si spécifique
            logger.info(f"🔍 Recherche avec références: {query[:50]}...")
            
            # Préparer les filtres selon le domaine
            where_filter = {}
            if query_domain == "impots":
                # Filtrer pour ne chercher que dans le Code des Impôts
                where_filter = {"file_name": {"$eq": "Senegal-Code-des-impot.pdf"}}
                logger.info("📊 Recherche limitée au Code des Impôts")
            elif query_domain == "douanes":
                # Filtrer pour ne chercher que dans le Code des Douanes
                where_filter = {"file_name": {"$eq": "Senegal-Code-2014-des-douanes.pdf"}}
                logger.info("� Recherche limitée au Code des Douanes")
            # Si general, pas de filtre - cherche dans les deux codes
            
            # Effectuer la recherche avec ou sans filtre
            if where_filter:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit * 2,
                    where=where_filter,
                    include=['documents', 'metadatas']
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit * 2,  # Récupérer plus pour pouvoir dédupliquer
                    include=['documents', 'metadatas']
                )
            
            if results and results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                
                context_parts = []
                references = []
                
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    
                    if metadata:
                        file_name = metadata.get('file_name', 'Document inconnu')
                        file_path = metadata.get('file_path', '')
                        line_start = metadata.get('line_start', 1)
                        line_end = metadata.get('line_end', 1)
                        page_start = metadata.get('page_start', 1)
                        page_end = metadata.get('page_end', 1)
                        
                        # Créer la référence précise
                        if line_start == line_end:
                            location = f"ligne {line_start}"
                        else:
                            location = f"lignes {line_start}-{line_end}"
                        
                        if page_start == page_end:
                            page_info = f"page {page_start}"
                        else:
                            page_info = f"pages {page_start}-{page_end}"
                        
                        # Extraire les informations d'article si disponibles
                        article_ref = metadata.get('article_ref', 'Section générale')
                        article_number = metadata.get('article_number', '')
                        section = metadata.get('section', '')
                        titre = metadata.get('titre', '')
                        
                        reference = {
                            'file_name': file_name,
                            'file_path': file_path,
                            'location': location,
                            'page_info': page_info,
                            'line_start': line_start,
                            'line_end': line_end,
                            'page_start': page_start,
                            'page_end': page_end,
                            'article_ref': article_ref,
                            'article_number': article_number,
                            'section': section,
                            'titre': titre,
                            'snippet': doc[:150] + "..." if len(doc) > 150 else doc
                        }
                        references.append(reference)
                        
                        # Créer une source info enrichie avec l'article
                        if article_ref and article_ref != 'Section générale':
                            source_info = f"[📄 {file_name} - {article_ref}, {page_info}, {location}]"
                        else:
                            source_info = f"[📄 {file_name}, {page_info}, {location}]"
                        context_parts.append(f"{source_info}\n{doc}")
                    else:
                        context_parts.append(doc)
                
                # Dédupliquer les références intelligemment
                deduplicated_references = self.deduplicate_references(references)
                
                # Limiter au nombre demandé après déduplication
                final_references = deduplicated_references[:limit]
                final_context_parts = context_parts[:len(final_references)]
                
                logger.info(f"✅ Contexte avec références déduplicates: {len(final_references)} documents uniques")
                return {
                    "context": "\n\n".join(final_context_parts),
                    "references": final_references
                }
            else:
                logger.warning("  Aucun contexte trouvé dans ChromaDB")
                return {"context": "", "references": []}
        except Exception as e:
            logger.error(f"  Erreur recherche ChromaDB: {e}")
            return {"context": "", "references": []}

    def search_specific_article(self, query: str) -> Dict:
        """Recherche spécifique d'articles par numéro avec recherche étendue et filtrage par domaine"""
        import re
        
        # Détecter le domaine de la question
        query_domain = self.detect_query_domain(query)
        
        # Corriger les fautes de frappe courantes
        query_corrected = query.lower()
        query_corrected = re.sub(r'atrticle', 'article', query_corrected)  # Correction "atrticle" → "article"
        query_corrected = re.sub(r'artcile', 'article', query_corrected)   # Correction "artcile" → "article"
        
        # Détecter si la requête demande un article spécifique ou une sous-définition
        article_patterns = [
            r'article\s*(\d+)(?:\s+(?:point|section|alinéa|définition)\s*(\d+(?:-\d+)?))?',  # Article X point Y ou Article X section Y-Z
            r'article\s*(\d+(?:-\d+)?)',  # Article X-Y direct
            r'art\.\s*(\d+(?:-\d+)?)', 
            r'art\s*(\d+(?:-\d+)?)',
            r'l\'article\s*(\d+(?:-\d+)?)',
            r'article\s*premier',
            r'premier\s*article',
            r'point\s*(\d+(?:-\d+)?)\s*(?:de\s*l\')?article\s*(\d+)',  # Point X-Y de l'article Z
            r'définition\s*(\d+(?:-\d+)?)\s*(?:de\s*l\')?article\s*(\d+)',  # Définition X de l'article Y
        ]
        
        article_number = None
        sub_definition = None
        
        for pattern in article_patterns:
            match = re.search(pattern, query_corrected)  # Utiliser la query corrigée
            if match:
                if 'premier' in pattern:
                    article_number = "1"
                elif 'point' in pattern or 'définition' in pattern:
                    # Pattern inversé: "point X de l'article Y" ou "définition X de l'article Y"
                    if match.lastindex >= 2:
                        sub_definition = match.group(1)
                        article_number = match.group(2)
                    else:
                        article_number = match.group(1)
                else:
                    # Pattern normal: "article X point Y"
                    article_number = match.group(1)
                    if match.lastindex >= 2 and match.group(2):
                        sub_definition = match.group(2)
                break
        
        if not article_number:
            return {"context": "", "references": []}
        
        try:
            if sub_definition:
                logger.info(f"🎯 Recherche article spécifique: Article {article_number}, définition {sub_definition}")
            else:
                logger.info(f"🎯 Recherche article spécifique: Article {article_number}")
            
            # Recherche étendue par contenu avec patterns multiples
            search_patterns = [
                f"article {article_number}",
                f"art. {article_number}",
                f"art {article_number}",
                f"article{article_number}",
                f"Art. {article_number}",
                f"Article {article_number}",
                f"ARTICLE {article_number}"
            ]
            
            # Si on cherche une sous-définition spécifique (comme "point 2-1" dans Article 1)
            if sub_definition:
                search_patterns.extend([
                    f"{sub_definition}.",  # "2-1." ou "3."
                    f"{sub_definition} ",  # "2-1 " ou "3 "
                    f"{sub_definition}-",  # Pour les patterns comme "2-1-"
                    f"définition {sub_definition}",
                    f"point {sub_definition}",
                    f"alinéa {sub_definition}"
                ])
            
            # Patterns spécifiques selon les articles
            if article_number == "1":
                search_patterns.extend([
                    "aux fins du présent code",
                    "on entend par",
                    "définitions",
                    # Définitions Code des Douanes
                    "adhérent à la fraude",
                    "aéroport douanier",
                    "port douanier",
                    "bureau de douane",
                    # Définitions Code des Impôts
                    "espèce d'une marchandise",
                    "dénomination qui lui est attribuée",
                    "tarif des douanes",
                    "contribuable",
                    "impôt sur le revenu",
                    "taxe sur la valeur ajoutée"
                ])
            elif article_number == "4":
                search_patterns.extend([
                    "personnes imposables",
                    "sociétés par actions",
                    "sociétés à responsabilité limitée",
                    "sarl",
                    "société anonyme",
                    "impôt sur les sociétés",
                    "soumises à l'impôt",
                    "champ d'application",
                    "sous-section",
                    "personnes morales"
                ])
            
            # Ajouter les patterns pour les sous-sections si le numéro d'article contient un tiret
            if "-" in article_number:
                base_number = article_number.split("-")[0]
                search_patterns.extend([
                    f"article {base_number}-",
                    f"Art. {base_number}-",
                    f"Article {base_number}-",
                    f"ARTICLE {base_number}-"
                ])
            
            if article_number == "1":
                search_patterns.extend([
                    "article premier",
                    "Article premier", 
                    "ARTICLE PREMIER",
                    "Art. 1er",
                    "article 1er",
                    "Article 1er",
                    "ARTICLE 1ER",
                    "Article 1°",
                    "Art. 1°"
                ])
                # Patterns de début d'article strict pour Article 1
                strict_patterns = [
                    "article premier :",
                    "Article premier :",
                    "ARTICLE PREMIER :",
                    "article 1er :",
                    "Article 1er :",
                    "ART. 1ER :",
                    "Article 1 :",
                    "ARTICLE 1 :"
                ]
                search_patterns.extend(strict_patterns)
            
            all_results = []
            
            # Essayer chaque pattern de recherche
            for pattern in search_patterns:
                try:
                    # Recherche par embedding avec filtre par domaine
                    query_embedding = self.generate_embeddings(pattern)
                    if query_embedding:
                        # Préparer le filtre selon le domaine
                        where_filter = {}
                        if query_domain == "impots":
                            where_filter = {"file_name": {"$eq": "Senegal-Code-des-impot.pdf"}}
                        elif query_domain == "douanes":
                            where_filter = {"file_name": {"$eq": "Senegal-Code-2014-des-douanes.pdf"}}
                        
                        # Effectuer la recherche avec ou sans filtre
                        if where_filter:
                            results = self.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=10,
                                where=where_filter,
                                include=['documents', 'metadatas']
                            )
                        else:
                            results = self.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=10,
                                include=['documents', 'metadatas']
                            )
                        
                        if results and results['documents'] and results['documents'][0]:
                            documents = results['documents'][0]
                            metadatas = results['metadatas'][0] if results['metadatas'] else []
                            
                            # Filtrer pour ne garder que ceux qui contiennent vraiment l'article
                            for i, doc in enumerate(documents):
                                doc_lower = doc.lower()
                                
                                # Éviter les lignes de sommaire avec points de suite
                                if "......" in doc or "………" in doc or "......." in doc:
                                    continue
                                    
                                # Éviter les documents trop courts (probablement des références)
                                if len(doc.strip()) < 100:
                                    continue
                                
                                # Vérifier les patterns principaux
                                pattern_matches = [p for p in search_patterns if p.lower() in doc_lower]
                                if pattern_matches:
                                    # Débogage : log du document trouvé
                                    logger.debug(f"📍 Pattern trouvé: {pattern_matches[0]} dans: {doc[:150]}...")
                                    
                                    # Pour l'article 1, prioriser les patterns de début strict
                                    if article_number == "1":
                                        strict_match = any(p.lower() + " " in doc_lower for p in ["article premier", "article 1er", "article 1"])
                                        if not strict_match:
                                            # Si c'est juste une référence à l'article 1, ignorer
                                            if "l'article 1" in doc_lower or "à l'article 1" in doc_lower or "article 1er alinéa" in doc_lower:
                                                logger.debug(f"⏭️ Référence ignorée: {doc[:100]}...")
                                                continue
                                    
                                    # Vérifier qu'il y a du contenu substantiel après la mention de l'article
                                    article_found = False
                                    for pattern in pattern_matches:
                                        if pattern.lower() in doc_lower:
                                            article_pos = doc_lower.find(pattern.lower())
                                            content_after = doc[article_pos + len(pattern):].strip()
                                            if len(content_after) > 50:  # Il doit y avoir du contenu substantiel
                                                article_found = True
                                                break
                                    
                                    if article_found:
                                        metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                                        all_results.append((doc, metadata))
                                        logger.debug(f"✅ Article accepté: {doc[:100]}...")
                except Exception as e:
                    logger.debug(f"Erreur pattern {pattern}: {e}")
                    continue
            
            # Supprimer les doublons et prioriser les vrais articles
            unique_results = []
            seen_docs = set()
            priority_articles = []  # Articles qui commencent vraiment par "Article X :"
            reference_articles = []  # Articles qui font juste référence
            
            for doc, metadata in all_results:
                doc_hash = hash(doc[:100])  # Hash des 100 premiers caractères
                if doc_hash not in seen_docs:
                    seen_docs.add(doc_hash)
                    
                    # Prioriser les articles qui commencent réellement par "Article X :"
                    doc_start = doc.strip().lower()
                    
                    if article_number == "1":
                        # Pour Article 1, chercher les vrais débuts d'article
                        is_real_article = (
                            doc_start.startswith("article 1 :") or 
                            doc_start.startswith("article premier :") or
                            doc_start.startswith("article 1er :") or
                            # Vérifier aussi les patterns avec espaces
                            doc_start.startswith("article  1") or
                            # Pattern pour définitions (ce qu'on a vu dans les logs)
                            ("aux fins du présent code" in doc_start and "article 1" in doc_start)
                        )
                        
                        # Si on cherche une sous-définition spécifique, vérifier qu'elle est présente
                        if sub_definition and is_real_article:
                            # Vérifier que la sous-définition demandée est dans ce document
                            has_sub_def = (
                                f"{sub_definition}." in doc or 
                                f"{sub_definition} " in doc or
                                f"{sub_definition}-" in doc
                            )
                            if has_sub_def:
                                priority_articles.append((doc, metadata))
                                logger.info(f"✅ Sous-définition {sub_definition} trouvée dans Article 1")
                            else:
                                logger.debug(f"⏭️ Sous-définition {sub_definition} non trouvée dans ce chunk")
                        elif is_real_article:
                            priority_articles.append((doc, metadata))
                        else:
                            # Ignorer les simples références
                            if not ("à l'article 1" in doc_start or "l'article 1er alinéa" in doc_start):
                                reference_articles.append((doc, metadata))
                    else:
                        # Pour les autres articles
                        if doc_start.startswith(f"article {article_number} :") or doc_start.startswith(f"article  {article_number}"):
                            priority_articles.append((doc, metadata))
                        else:
                            reference_articles.append((doc, metadata))
            
            # Combiner avec priorité aux vrais articles
            unique_results = priority_articles + reference_articles
            unique_results = unique_results[:5]  # Limiter à 5 résultats maximum
            
            # Si on a des articles prioritaires, limiter aux 2 premiers + 1 référence maximum
            if priority_articles:
                unique_results = priority_articles[:2] + reference_articles[:1]
            else:
                unique_results = reference_articles[:3]
            
            if unique_results:
                context_parts = []
                references = []
                
                for doc, metadata in unique_results:
                    file_name = metadata.get('file_name', 'Document')
                    article_ref = metadata.get('article_ref', f'Article {article_number}')
                    
                    # Créer un snippet plus long pour l'article
                    snippet = doc[:500] + "..." if len(doc) > 500 else doc
                    
                    source_info = f"[📄 {file_name} - {article_ref}]"
                    context_parts.append(f"{source_info}\n{doc}")
                    
                    reference = {
                        'file_name': file_name,
                        'file_path': metadata.get('file_path', ''),
                        'article_ref': article_ref,
                        'article_number': metadata.get('article_number', f'Article {article_number}'),
                        'snippet': snippet,
                        'line_start': metadata.get('line_start', 1),
                        'line_end': metadata.get('line_end', 1),
                        'page_start': metadata.get('page_start', 1),
                        'page_end': metadata.get('page_end', 1)
                    }
                    references.append(reference)
                
                if sub_definition:
                    logger.info(f"✅ Article {article_number}, définition {sub_definition} trouvée: {len(unique_results)} sections")
                else:
                    logger.info(f"✅ Article {article_number} trouvé: {len(unique_results)} sections uniques")
                
                # Contexte spécialisé avec identification du code source
                final_context = "\n\n".join(context_parts)
                if priority_articles:
                    # Identifier le code source depuis les métadonnées
                    first_ref = references[0] if references else {}
                    file_name = first_ref.get('file_name', '').lower()
                    
                    if 'impot' in file_name:
                        code_source = "CODE GÉNÉRAL DES IMPÔTS (CGI) SÉNÉGAL"
                    elif 'douane' in file_name:
                        code_source = "CODE DES DOUANES SÉNÉGAL"
                    else:
                        code_source = "CODE JURIDIQUE SÉNÉGAL"
                    
                    if article_number == "1" and priority_articles:
                        # Mettre l'accent sur le vrai Article 1
                        priority_context = priority_articles[0][0]  # Premier article prioritaire
                        if sub_definition:
                            final_context = f"ARTICLE 1 DU {code_source} - DÉFINITION {sub_definition}:\n\n{priority_context}\n\n" + final_context
                        else:
                            final_context = f"ARTICLE 1 AUTHENTIQUE DU {code_source}:\n\n{priority_context}\n\n" + final_context
                    else:
                        final_context = f"ARTICLE {article_number} DU {code_source}:\n\n" + final_context
                
                return {
                    "context": final_context,
                    "references": references
                }
            
        except Exception as e:
            logger.error(f"Erreur recherche article spécifique: {e}")
        
        return {"context": "", "references": []}

    def search_context(self, query: str, limit: int = 5) -> str:
        """Recherche le contexte dans les documents indexés (version simple)"""
        result = self.search_context_with_references(query, limit)
        return result.get("context", "")
    
    def should_use_rag(self, message: str) -> bool:
        """Détermine si la question nécessite le RAG (documents légaux) ou juste Mistral"""
        message_lower = message.lower().strip()
        
        logger.info(f"🤖 Analyse RAG vs Mistral: '{message_lower}'")
        
        # Mots-clés qui nécessitent OBLIGATOIREMENT le RAG (documents légaux sénégalais)
        rag_required_keywords = [
            # Fiscalité sénégalaise spécifique
            'impot', 'impôt', 'impots', 'impôts', 'tva', 'is', 'ir', 'ircm',
            'taxe', 'taxes', 'droit', 'droits', 'tarif', 'taux',
            'déclaration', 'declaration', 'déclarer', 'declarer',
            'contribuable', 'contribuables', 'fiscal', 'fiscale', 'fiscalité', 'fiscalite',
            'cgi', 'code general', 'code général', 'dgi',
            # Douanes sénégalaises spécifique
            'douane', 'douanes', 'douanier', 'douanière', 'douaniere',
            'dédouanement', 'dedouanement', 'dédouaner', 'dedouaner',
            'importation', 'exportation', 'import', 'export', 'transit',
            'marchandise', 'marchandises', 'cargo', 'fret',
            'tarif douanier', 'nomenclature', 'classement',
            # Articles et références légales
            'article', 'art.', 'loi', 'décret', 'arrêté',
            'circulaire', 'instruction', 'réglementation',
            # Codes spécifiques sénégalais
            'code des impôts', 'code des douanes', 'code général des impôts',
            'cgi sénégal', 'code douanier',
            # Entités sénégalaises officielles
            'senegal', 'sénégal', 'dakar', 'sénégalais', 'sénégalaise',
            'direction générale', 'direction generale', 'ministère', 'ministere',
            # Procédures administratives sénégalaises
            'formulaire', 'formulaires', 'attestation', 'certificat',
            'autorisation', 'licence', 'permis', 'agrément', 'agrement'
        ]
        
        # Questions générales qui doivent utiliser SEULEMENT Mistral
        mistral_only_keywords = [
            # Développement et technologie
            'deployer', 'déployer', 'deployment', 'déploiement',
            'application', 'app', 'site web', 'website', 'serveur', 'server',
            'base de données', 'database', 'api', 'framework',
            'programmation', 'programming', 'coding', 'développement',
            'javascript', 'python', 'java', 'php', 'html', 'css',
            'docker', 'kubernetes', 'git', 'github', 'gitlab',
            'cloud', 'aws', 'azure', 'google cloud',
            # Technologie générale
            'installer', 'installation', 'configurer', 'configuration',
            'réseau', 'network', 'sécurité', 'security', 'backup',
            'ordinateur', 'computer', 'laptop', 'smartphone',
            # Questions générales
            'comment faire', 'how to', 'tutorial', 'tutoriel',
            'guide', 'méthode', 'technique', 'stratégie',
            'meilleur', 'best', 'comparaison', 'différence',
            # Autres domaines non-légaux
            'santé', 'health', 'sport', 'cuisine', 'voyage', 'musique',
            'éducation', 'formation', 'cours', 'apprentissage'
        ]
        
        # Vérifier d'abord si c'est une question qui nécessite Mistral seulement
        for mistral_word in mistral_only_keywords:
            if mistral_word in message_lower:
                logger.info(f"🎯 Question générale détectée: '{mistral_word}' - Utilisation Mistral seul")
                return False
        
        # Vérifier si c'est une question légale/fiscale qui nécessite le RAG
        for rag_word in rag_required_keywords:
            if rag_word in message_lower:
                logger.info(f"📚 Question légale détectée: '{rag_word}' - Utilisation RAG + documents")
                return True
        
        # Par défaut, utiliser Mistral pour les questions ambiguës
        logger.info(f"❓ Question ambiguë - Utilisation Mistral par défaut")
        return False

    def is_greeting_or_general(self, message: str) -> bool:
        """Détecte si le message est une salutation ou question générale"""
        message_lower = message.lower().strip()
        
        # Debug: log du message pour voir ce qui se passe
        logger.info(f"🔍 Analyse message salutation: '{message_lower}'")
        
        # Mots-clés techniques qui indiquent une question technique (pas une salutation)
        technical_keywords = [
            # Documents
            'document', 'fichier', 'pdf', 'word', 'excel', 'csv', 'txt',
            'recherche', 'rechercher', 'chercher', 'trouve', 'trouver',
            'analyse', 'analyser', 'résumé', 'résumer', 'contenu',
            # Fiscalité sénégalaise
            'impot', 'impôt', 'impots', 'impôts', 'tva', 'is', 'ir', 'ircm',
            'taxe', 'taxes', 'droit', 'droits', 'tarif', 'taux', 'calcul',
            'déclaration', 'declaration', 'déclarer', 'declarer',
            'contribuable', 'contribuables', 'fiscal', 'fiscale', 'fiscalité', 'fiscalite',
            'cgi', 'code general', 'code général', 'dgi',
            # Douanes sénégalaises
            'douane', 'douanes', 'douanier', 'douanière', 'douaniere',
            'dédouanement', 'dedouanement', 'dédouaner', 'dedouaner',
            'importation', 'exportation', 'import', 'export', 'transit',
            'marchandise', 'marchandises', 'cargo', 'fret',
            'tarif douanier', 'nomenclature', 'classement',
            # Procédures administratives
            'procedure', 'procédure', 'procedures', 'procédures',
            'formulaire', 'formulaires', 'attestation', 'certificat',
            'autorisation', 'licence', 'permis', 'agrément', 'agrement',
            # Entités sénégalaises
            'senegal', 'sénégal', 'dakar', 'sénégalais', 'sénégalaise',
            'direction générale', 'direction generale', 'ministère', 'ministere',
            # Termes génériques techniques
            'financement', 'crédit', 'prêt', 'investissement', 'garantie',
            'entreprise', 'société', 'projet', 'stratégie', 'plan', 'budget', 'rapport',
            'indexer', 'indexation', 'traitement', 'processus'
        ]
        
        # Vérifier si le message contient des mots techniques
        for tech_word in technical_keywords:
            if tech_word in message_lower:
                logger.info(f"❌ Mot technique détecté: '{tech_word}' - Pas une salutation")
                return False
        
        # Mots-clés de salutation
        greeting_words = [
            'salut', 'bonjour', 'bonsoir', 'hello', 'hi', 'hey', 
            'coucou', 'yo', 'wesh'
        ]
        
        # Expressions de politesse (uniquement si pas de contenu technique)
        polite_expressions = [
            'ça va', 'comment ça va', 'comment allez-vous', 'comment allez vous',
            'comment vous allez', 'ça roule', 'quoi de neuf', 'comment tu vas'
        ]
        
        # Questions générales sur ALEX (courtes et sans technique)
        general_questions = [
            'qui es-tu', 'que fais-tu', 'aide', 'help', 
            'présente-toi', 'tu es qui', 'c\'est quoi'
        ]
        
        # Vérifier si le message contient UNIQUEMENT des mots de salutation
        for word in greeting_words:
            if word in message_lower:
                # Vérifier que c'est bien une salutation simple (pas "bonjour, comment déployer...")
                if len(message_lower.split()) <= 4:  # Maximum 4 mots pour une salutation
                    logger.info(f"✅ Salutation simple détectée: '{word}'")
                    return True
            
        # Vérifier les expressions de politesse COURTES
        for expr in polite_expressions:
            if expr in message_lower and len(message_lower) <= 20:  # Expressions courtes seulement
                logger.info(f"✅ Expression de politesse courte détectée: '{expr}'")
                return True
            
        # Vérifier si c'est une question générale sur ALEX
        for q in general_questions:
            if q in message_lower and len(message_lower) <= 15:
                logger.info(f"✅ Question générale courte détectée: '{q}'")
                return True
        
        logger.info(f"❌ Pas de salutation détectée pour: '{message_lower}'")
        return False
    
    def generate_greeting_response(self, message: str) -> str:
        """Génère une réponse appropriée pour les salutations et questions générales"""
        message_lower = message.lower().strip()
        
        # Réponses aux salutations avec "comment allez-vous" ou similaire
        if any(pattern in message_lower for pattern in ['comment allez', 'comment ça va', 'ça va']):
            return """Bonjour ! Je vais très bien, merci ! Je suis SRMT-DOCUMIND, votre assistant IA spécialisé pour les contribuables sénégalais.

🏛️ Je maîtrise maintenant les DEUX codes principaux du Sénégal :

📋 Mes domaines d'expertise :
• 📊 Code Général des Impôts (CGI) du Sénégal
• 🚢 Code des Douanes sénégalais  
• 💼 Procédures fiscales et douanières
• 📝 Obligations déclaratives des contribuables
• 💰 Droits, taxes et impôts applicables

💡 Exemples de questions :
• "Quels sont les taux de TVA au Sénégal ?" (CGI)
• "Comment déclarer mes revenus ?" (CGI)
• "Procédure de dédouanement pour les marchandises" (Douanes)
• "Définition de l'espèce d'une marchandise" (Codes)

🎯 Je peux analyser et rechercher dans les deux codes simultanément !

Sur quoi puis-je vous aider aujourd'hui ?"""
        
        # Réponses aux salutations simples
        elif any(greeting in message_lower for greeting in ['salut', 'bonjour', 'hello', 'hi', 'hey', 'coucou']):
            return """Salut ! Je suis SRMT-DOCUMIND, votre assistant IA spécialisé en fiscalité et douanes sénégalaises ! 🇸🇳

🎯 Je vous aide avec :
• Le Code des Impôts sénégalais
• Le Code des Douanes du Sénégal
• Vos obligations fiscales et douanières
• L'analyse de documents administratifs

📚 Mes capacités :
• Analyser vos documents fiscaux (PDF, Word, Excel, etc.)
• Expliquer les procédures douanières
• Calculer vos taxes et droits
• Vous guider dans vos démarches

💼 Exemples pratiques :
• "Analyse ce document de la DGI"
• "Comment calculer l'IS de ma société ?"
• "Procédure d'exonération douanière"

N'hésitez pas à me poser vos questions fiscales et douanières !"""

        # Questions sur SRMT-DOCUMIND
        elif any(q in message_lower for q in ['qui es-tu', 'présente-toi', 'tu es qui']):
            return """Je suis SRMT-DOCUMIND - Assistant IA Expert Fiscal et Douanier du Sénégal 🇸🇳

🎯 Ma mission : Accompagner les contribuables sénégalais dans leurs démarches fiscales et douanières

📋 Mes spécialités :
• Code Général des Impôts (CGI) sénégalais
• Code des Douanes du Sénégal  
• Procédures DGI (Direction Générale des Impôts)
• Procédures douanières et de transit
• Calculs de taxes, droits et impôts

🔧 Ma technologie :
• Modèle IA Mistral 7B optimisé pour le droit fiscal sénégalais
• Base de données vectorielle spécialisée
• Analyse automatique des documents administratifs
• Interface adaptée aux professionnels

💡 Je peux vous aider avec :
• L'interprétation des textes fiscaux
• Le calcul de vos impôts et taxes
• L'analyse de vos documents DGI/Douanes
• Les procédures administratives

Posez-moi vos questions sur la fiscalité et les douanes sénégalaises !"""

        # Questions d'aide
        elif any(q in message_lower for q in ['aide', 'help', 'comment']):
            return """🎯 Guide d'utilisation SRMT-DOCUMIND - Expert Fiscal Sénégal

📋 Comment poser vos questions :
• Soyez précis : "Taux de TVA pour les services"
• Mentionnez le contexte : "Entreprise, particulier, import/export"
• Citez les articles si connus : "Article 123 du CGI"

🏛️ Domaines couverts :
• Impôt sur le Revenu (IR) et Impôt sur les Sociétés (IS)
• Taxe sur la Valeur Ajoutée (TVA)
• Droits de douane et taxes douanières
• Procédures déclaratives et de paiement
• Contentieux fiscal et douanier

💼 Types de demandes :
• Calculs d'impôts : "Comment calculer ma TVA ?"
• Procédures : "Étapes de dédouanement"
• Interprétation : "Que signifie cet article ?"
• Analyse de documents administratifs

📊 Formats supportés : PDF, Word, Excel, documents DGI/Douanes

🚀 Astuce : Plus votre question est précise, plus ma réponse sera adaptée !

Quelle est votre question fiscale ou douanière ?"""

        # Réponse par défaut
        else:
            return """Salut ! Je suis SRMT-DOCUMIND, votre assistant IA spécialisé en fiscalité et douanes sénégalaises ! 🇸🇳

Posez-moi vos questions sur le Code des Impôts, le Code des Douanes, ou analysez vos documents administratifs.

Exemple: "Quels sont les taux d'imposition ?" ou "Analyse ce document de la DGI" """

    def open_file_at_location(self, file_path: str, line_number: int = 1) -> bool:
        """Ouvre un fichier à une ligne spécifique"""
        try:
            import subprocess
            import sys
            
            # Corriger le chemin du fichier
            corrected_path = file_path
            
            # Si le chemin n'existe pas, essayer de le corriger
            if not os.path.exists(file_path):
                logger.info(f"Correction du chemin: {file_path}")
                
                # Essayer avec le répertoire de travail actuel
                if not os.path.isabs(file_path):
                    corrected_path = os.path.join(os.getcwd(), file_path)
                    logger.info(f"Tentative chemin absolu: {corrected_path}")
                
                # Si toujours pas trouvé, essayer avec le répertoire documents
                if not os.path.exists(corrected_path):
                    # Extraire juste le nom du fichier
                    filename = os.path.basename(file_path)
                    # Si le nom de fichier semble concaténé avec "documents"
                    if filename.startswith('documents') and len(filename) > 9:
                        filename = filename[9:]  # Enlever "documents"
                    
                    # Construire le chemin correct
                    corrected_path = os.path.join(self.watch_dir, filename)
                    logger.info(f"Tentative avec répertoire surveillé: {corrected_path}")
                
                # Dernière tentative: chercher le fichier dans le répertoire documents
                if not os.path.exists(corrected_path):
                    filename = os.path.basename(file_path)
                    if filename.startswith('documents'):
                        filename = filename[9:]
                    
                    # Chercher tous les fichiers qui correspondent
                    for file_in_dir in self.watch_dir.rglob('*'):
                        if file_in_dir.name == filename:
                            corrected_path = str(file_in_dir)
                            logger.info(f"Fichier trouvé: {corrected_path}")
                            break
            
            # Vérifier que le fichier existe maintenant
            if not os.path.exists(corrected_path):
                logger.error(f"Fichier introuvable même après correction: {file_path} -> {corrected_path}")
                return False
            
            logger.info(f"Ouverture du fichier: {corrected_path} à la ligne {line_number}")
            
            # Déterminer l'extension du fichier pour choisir la bonne application
            # Utiliser le chemin original pour détecter l'extension si le corrigé ne marche pas
            original_extension = Path(file_path).suffix.lower()
            corrected_extension = Path(corrected_path).suffix.lower()
            
            # Prendre l'extension qui a l'air correcte
            if original_extension and original_extension in ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt', '.md']:
                file_extension = original_extension
            else:
                file_extension = corrected_extension
            
            logger.info(f"Extension détectée: {file_extension} (original: {original_extension}, corrigé: {corrected_extension})")
            logger.info(f"Chemin original: {file_path}")
            logger.info(f"Chemin corrigé: {corrected_path}")
            
            # Ouvrir selon l'OS et le type de fichier
            if sys.platform.startswith('win'):
                # Pour les fichiers PDF
                if file_extension == '.pdf':
                    try:
                        # Essayer Adobe Reader d'abord
                        subprocess.run(['start', '', corrected_path], shell=True, check=False)
                        logger.info(f"Fichier PDF ouvert avec l'application par défaut")
                        return True
                    except:
                        # Fallback avec l'explorateur Windows
                        os.startfile(corrected_path)
                        logger.info(f"Fichier PDF ouvert avec l'explorateur")
                        return True
                
                # Pour les fichiers Word (.docx, .doc)
                elif file_extension in ['.docx', '.doc', '.odt']:
                    try:
                        # Ouvrir avec Word ou l'application par défaut
                        os.startfile(corrected_path)
                        logger.info(f"Fichier Word ouvert avec l'application par défaut")
                        return True
                    except Exception as e:
                        logger.error(f"Erreur ouverture fichier Word: {e}")
                        return False
                
                # Pour les fichiers Excel
                elif file_extension in ['.xlsx', '.xls', '.csv']:
                    try:
                        os.startfile(corrected_path)
                        logger.info(f"Fichier Excel ouvert avec l'application par défaut")
                        return True
                    except Exception as e:
                        logger.error(f"Erreur ouverture fichier Excel: {e}")
                        return False
                
                # Pour les fichiers texte - utiliser un éditeur de texte
                elif file_extension in ['.txt', '.md', '.json', '.py', '.js', '.html', '.css']:
                    try:
                        # Essayer VS Code d'abord (meilleur pour aller à une ligne)
                        subprocess.run(['code', '-g', f'{corrected_path}:{line_number}'], check=False)
                        logger.info(f"Fichier texte ouvert avec VS Code à la ligne {line_number}")
                        return True
                    except:
                        try:
                            # Essayer Notepad++ avec numéro de ligne
                            subprocess.run(['notepad++', f'-n{line_number}', corrected_path], check=False)
                            logger.info(f"Fichier texte ouvert avec Notepad++ à la ligne {line_number}")
                            return True
                        except:
                            try:
                                # Fallback Notepad simple
                                subprocess.run(['notepad', corrected_path], check=False)
                                logger.info(f"Fichier texte ouvert avec Notepad")
                                return True
                            except:
                                # Dernier fallback
                                os.startfile(corrected_path)
                                logger.info(f"Fichier ouvert avec application par défaut")
                                return True
                
                # Pour tous les autres types de fichiers
                else:
                    try:
                        os.startfile(corrected_path)
                        logger.info(f"Fichier ouvert avec l'application par défaut")
                        return True
                    except Exception as e:
                        logger.error(f"Erreur ouverture fichier: {e}")
                        return False
            else:
                # Linux/Mac - ouvrir selon le type de fichier
                if file_extension == '.pdf':
                    # Lecteurs PDF courants sur Linux/Mac
                    pdf_viewers = ['evince', 'okular', 'xpdf', 'open']  # 'open' pour Mac
                    for viewer in pdf_viewers:
                        try:
                            subprocess.run([viewer, corrected_path], check=False)
                            logger.info(f"Fichier PDF ouvert avec {viewer}")
                            return True
                        except:
                            continue
                    
                    # Fallback
                    subprocess.run(['xdg-open', corrected_path], check=False)
                    logger.info(f"Fichier PDF ouvert avec xdg-open")
                    return True
                
                elif file_extension in ['.docx', '.doc', '.odt', '.xlsx', '.xls']:
                    # Documents Office
                    try:
                        subprocess.run(['xdg-open', corrected_path], check=False)
                        logger.info(f"Fichier Office ouvert avec xdg-open")
                        return True
                    except Exception as e:
                        logger.error(f"Erreur ouverture fichier Office: {e}")
                        return False
                
                else:
                    # Fichiers texte - essayer différents éditeurs avec support des lignes
                    editors = ['code', 'gedit', 'nano', 'vim']
                    for editor in editors:
                        try:
                            if editor == 'code':
                                subprocess.run([editor, '-g', f'{corrected_path}:{line_number}'], check=False)
                            else:
                                subprocess.run([editor, corrected_path], check=False)
                            logger.info(f"Fichier texte ouvert avec {editor}")
                            return True
                        except:
                            continue
                            
                    # Fallback
                    subprocess.run(['xdg-open', corrected_path], check=False)
                    logger.info(f"Fichier ouvert avec xdg-open")
                    return True
                
        except Exception as e:
            logger.error(f"Erreur ouverture fichier: {e}")
            return False

    def generate_natural_greeting_response(self, message: str) -> str:
        """Génère une réponse naturelle aux salutations en utilisant Mistral directement"""
        try:
            # Prompt pour que Mistral réponde naturellement aux salutations
            greeting_prompt = f"""Tu es SRMT-DOCUMIND, un assistant IA intelligent spécialisé pour les contribuables sénégalais en fiscalité et douanes.

L'utilisateur te dit: "{message}"

IMPORTANT: Tu es un expert en Code des Impôts et Code des Douanes du Sénégal. Tu aides les contribuables sénégalais avec leurs questions fiscales et douanières.

Réponds de façon naturelle et professionnelle:
- Présente-toi comme SRMT-DOCUMIND, l'assistant expert fiscal et douanier sénégalais
- Précise tes spécialités : Code des Impôts, Code des Douanes, DGI, procédures fiscales
- Mentionne que tu peux analyser documents administratifs (PDF, Word, Excel)
- Reste professionnel et utilisé des émojis appropriés (🇸🇳, 🏛️, 📋)
- Invite l'utilisateur à poser ses questions fiscales/douanières
- Maximum 3-4 lignes

Réponse:"""

            payload = {
                "model": self.config.OLLAMA_CHAT_MODEL,
                "prompt": greeting_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Plus de créativité pour les salutations
                    "top_p": 0.9,
                    "max_tokens": 200
                }
            }
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                natural_response = response.json()['response']
                logger.info(f"🤖 Réponse naturelle de salutation générée")
                return natural_response.strip()
            else:
                # Fallback vers réponse prédéfinie
                return self.generate_greeting_response(message)
                
        except Exception as e:
            logger.error(f"Erreur génération réponse naturelle: {e}")
            # Fallback vers réponse prédéfinie
            return self.generate_greeting_response(message)

    def chat(self, message: str, declaration_mode: bool = False) -> Dict:
        """Génère une réponse de chat avec références précises"""
        try:
            # Vérifier si c'est une salutation ou question générale
            if self.is_greeting_or_general(message):
                # Pour les salutations, retourner une réponse simple sans références
                response_text = self.generate_natural_greeting_response(message)
                return {
                    "response": response_text,
                    "references": []
                }
            
            # NOUVELLE LOGIQUE HYBRIDE : Décider entre RAG et Mistral
            should_use_documents = self.should_use_rag(message)
            
            if should_use_documents or declaration_mode:
                if declaration_mode:
                    logger.info("📋 Utilisation du MODE DÉCLARATION - Focus sur declaration.csv")
                else:
                    logger.info("📚 Utilisation du RAG + Documents légaux sénégalais")
                    
                # Rechercher le contexte avec références précises
                search_result = self.search_context_with_references(message, limit=3, declaration_mode=declaration_mode)
                context = search_result.get("context", "")
                references = search_result.get("references", [])
                
                # Recherche supplémentaire avec mots-clés si pas de résultat (seulement en mode normal)
                if not declaration_mode:
                    keywords = [word for word in message.split() if len(word) > 3]
                    if keywords and not context:
                        keyword_query = " ".join(keywords)
                        search_result2 = self.search_context_with_references(keyword_query, limit=3, declaration_mode=False)
                        context = search_result2.get("context", "")
                        references = search_result2.get("references", [])
            else:
                logger.info("🤖 Utilisation de Mistral seul - Question générale")
                # Pour les questions générales, utiliser directement Mistral sans RAG
                prompt = f"""Tu es SRMT-DOCUMIND, un assistant IA intelligent et polyvalent.

L'utilisateur te pose cette question: "{message}"

INSTRUCTIONS:
- Réponds de manière complète et utile
- Si c'est une question technique (développement, déploiement, etc.), donne des conseils pratiques
- Si c'est sur la création d'entreprise en général, donne des informations générales (pas spécifiquement sénégalaises)
- Utilise tes connaissances générales, ne mentionne PAS les codes sénégalais pour cette question
- Sois pratique et concis
- N'utilise PAS les documents légaux sénégalais pour cette réponse

RÉPONSE:"""

                payload = {
                    "model": self.config.OLLAMA_CHAT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                }
                
                response = requests.post(
                    f"{self.config.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    mistral_response = response.json()['response']
                    logger.info("✅ Réponse Mistral générée avec succès")
                    return {
                        "response": mistral_response,
                        "references": []
                    }
                else:
                    return {
                        "response": "Désolé, je rencontre un problème technique avec le serveur IA. Veuillez réessayer.",
                        "references": []
                    }
            
            # Continuer avec la logique RAG si should_use_documents == True ou declaration_mode == True
            context = locals().get('context', '')
            references = locals().get('references', [])
            
            # FORCER l'utilisation du contexte des documents
            if context and context.strip():
                # Détecter le domaine de la question pour validation (sauf en mode déclaration)
                if not declaration_mode:
                    query_domain = self.detect_query_domain(message)
                    
                    # Vérifier la cohérence du domaine dans les références
                    if references:
                        context_domain = "general"
                        for ref in references:
                            file_name = ref.get('file_name', '').lower()
                            if 'impot' in file_name:
                                context_domain = "impots"
                                break
                            elif 'douane' in file_name:
                                context_domain = "douanes"
                                break
                        
                        # Vérifier la cohérence entre question et contexte
                        if query_domain != "general" and context_domain != "general" and query_domain != context_domain:
                            logger.warning(f"⚠️ Incohérence détectée: Question {query_domain} vs Contexte {context_domain}")
                            # Relancer une recherche plus ciblée
                            search_result = self.search_context_with_references(message, limit=5, declaration_mode=False)
                            context = search_result.get("context", "")
                            references = search_result.get("references", [])
                
                # Vérifier la pertinence du contexte
                question_keywords = message.lower().split()
                context_lower = context.lower()
                keyword_found = any(kw in context_lower for kw in question_keywords if len(kw) > 3)
                
                if declaration_mode:
                    # Mode Déclaration - Prompt spécialisé pour declaration.csv
                    prompt = f"""Tu es SRMT-DOCUMIND, assistant IA expert en analyse de données déclaratives.

⚠️ MODE DÉCLARATION ACTIVÉ - FOCUS EXCLUSIF SUR DECLARATION.CSV

DONNÉES DÉCLARATIVES TROUVÉES:
{context}

QUESTION DE L'UTILISATEUR: {message}

INSTRUCTIONS SPÉCIALISÉES MODE DÉCLARATION:
- Tu travailles EXCLUSIVEMENT avec les données du fichier declaration.csv
- Analyse UNIQUEMENT les informations déclaratives fournies dans le contexte
- Fournis des statistiques, analyses ou réponses basées sur ces données déclaratives
- Ne fait PAS référence aux Codes juridiques (Impôts/Douanes) en mode déclaration
- Concentre-toi sur les patterns, tendances et insights dans les données
- Si la donnée demandée n'est pas dans declaration.csv, indique-le clairement
- Structure ta réponse avec des chiffres précis et des analyses factuelles
- Mentionne toujours que tu analyses les données de declaration.csv

ANALYSE EXCLUSIVE DES DONNÉES DÉCLARATIVES:"""
                elif keyword_found or any(keyword in context_lower for keyword in ["impot", "tva", "douane", "fiscal", "cgi", "dgi", "senegal", "sénégal", "article"]):
                    # Mode normal - Identifier le code source pour une réponse ciblée
                    code_source = "Code juridique sénégalais"
                    if references:
                        file_name = references[0].get('file_name', '').lower()
                        if 'impot' in file_name:
                            code_source = "Code Général des Impôts (CGI) sénégalais"
                        elif 'douane' in file_name:
                            code_source = "Code des Douanes sénégalais"
                    
                    prompt = f"""Tu es SRMT-DOCUMIND, assistant IA expert en fiscalité et douanes sénégalaises.

CONTEXTE PRÉCIS DU {code_source.upper()}:
{context}

QUESTION DU CONTRIBUABLE: {message}

INSTRUCTIONS CRITIQUES:
- Tu es un expert du Code des Impôts et Code des Douanes du Sénégal
- Réponds EXCLUSIVEMENT avec les informations du {code_source}
- Ne mélange PAS les informations du Code des Douanes avec le Code des Impôts
- CITE PRÉCISÉMENT les numéros d'articles mentionnés dans le contexte
- Si la question porte sur un article spécifique, trouve et cite le contenu exact
- Structure ta réponse: Article X du {code_source} - [Contenu exact] - Explication pratique
- Mentionne TOUJOURS les références légales précises du {code_source}
- Si plusieurs articles sont pertinents, liste-les tous avec leurs contenus
- Reste précis et factuel selon le texte officiel du {code_source}
- Commence ta réponse en précisant de quel code provient l'information

RÉPONSE EXPERTE EXCLUSIVEMENT DU {code_source.upper()}:"""
                else:
                    if declaration_mode:
                        return {
                            "response": f"""📋 **MODE DÉCLARATION ACTIVÉ** mais aucune donnée pertinente trouvée dans declaration.csv.

⚠️ **Problèmes possibles :**
- Le fichier declaration.csv n'existe pas dans le dossier documents/
- Le fichier n'a pas encore été indexé automatiquement
- Les données recherchées ne correspondent pas au contenu du fichier
- Le fichier declaration.csv est vide ou mal formaté

🔧 **Actions recommandées :**
1. Vérifiez que declaration.csv est bien dans le dossier documents/
2. Attendez quelques secondes pour l'indexation automatique
3. Ou utilisez le bouton "Réindexer" pour forcer l'indexation
4. Désactivez le Mode Déclaration pour rechercher dans les Codes juridiques

💡 **Astuce :** Une fois declaration.csv ajouté et indexé, le Mode Déclaration permettra des analyses spécialisées sur vos données déclaratives.""",
                            "references": []
                        }
                    else:
                        return {
                            "response": f"""Le contexte trouvé ne semble pas correspondre directement à votre question fiscale/douanière.

💡 **Suggestions pour améliorer votre recherche :**
- Utilisez des termes fiscaux précis : "TVA", "impôt sur les sociétés", "déclaration"
- Mentionnez les codes : "CGI", "Code des douanes", "article"  
- Précisez le contexte : "entreprise", "particulier", "import/export"
- Essayez : "taux d'imposition", "procédure de déclaration", "calcul des taxes"

📚 Vérifiez que vos documents du Code des Impôts/Douanes sont bien indexés.

🤖 Si le problème persiste, cela peut être dû à une connexion lente avec le serveur IA.""",
                            "references": references
                        }
            else:
                return {
                    "response": f"""Cette information fiscale/douanière n'est pas disponible dans les documents indexés actuellement.

📚 **Documents fiscaux recommandés pour SRMT-DOCUMIND :**
- Code Général des Impôts (CGI) du Sénégal
- Code des Douanes sénégalais  
- Circulaires et instructions de la DGI
- Procédures douanières officielles

💡 **Exemples de questions fiscales/douanières :**
- "Quels sont les taux de TVA au Sénégal ?"
- "Comment calculer l'impôt sur les sociétés ?"
- "Procédure de dédouanement des marchandises"
- "Délais de déclaration fiscale"
- "Exonérations douanières disponibles"

⚠️ **Pour ajouter des documents fiscaux :**
- Placez vos documents (PDF, Word, Excel) dans le dossier surveillé
- L'indexation automatique peut prendre quelques instants
- Formats supportés : CGI, textes de loi, circulaires, formulaires

🔧 **Problème de connexion ?** 
Le serveur IA peut être temporairement lent. Réessayez dans quelques instants.""",
                    "references": []
                }
            
            # Générer la réponse avec Ollama
            payload = {
                "model": self.config.OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                ollama_response = response.json()['response']
                return {
                    "response": ollama_response,
                    "references": references
                }
            else:
                return {
                    "response": "Désolé, je rencontre un problème technique. Veuillez réessayer.",
                    "references": []
                }
                
        except Exception as e:
            logger.error(f"Erreur chat: {e}")
            return {
                "response": "Une erreur s'est produite. Veuillez réessayer dans un moment.",
                "references": []
            }

# Template HTML ultra moderne et responsif avec effets
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SRMT-DOCUMIND - Assistant Fiscal et Douanier Sénégal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #2C5530 0%, #1B4332 100%);
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
        }



        /* Application plein écran */
        .chat-app {
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 0;
            margin: 0;
            width: 100vw;
            max-width: none;
            background: transparent;
        }



        .chat-header {
            background: linear-gradient(135deg, #2C5530 0%, #1B4332 100%);
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
        }

        .chat-header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .chat-header p {
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
            font-weight: 300;
        }

        .chat-container {
            flex: 1;
            padding: 40px;
            overflow-y: auto;
            background: transparent;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }



        /* Scrollbar personnalisée */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #2D4D5B, #3388ff);
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .chat-container::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #039, #3388ff);
        }

        .message {
            margin-bottom: 20px;
            padding: 18px 24px;
            border-radius: 18px;
            animation: messageSlideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .message:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateY(30px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        .user-message {
            background: linear-gradient(135deg, #2C5530 0%, #1B4332 100%);
            color: white;
            margin-left: 25%;
            text-align: right;
            box-shadow: 0 4px 15px rgba(44, 85, 48, 0.3);
            border: none;
        }

        .user-message::after {
            display: none;
        }

        .assistant-message {
            background: rgba(255, 255, 255, 0.9);
            margin-right: 25%;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            color: #333;
            border: 1px solid rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
        }

        .assistant-message::after {
            display: none;
        }

        .chat-input-section {
            padding: 30px 40px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 -2px 20px rgba(0, 0, 0, 0.1);
        }

        .input-section {
            display: flex;
            gap: 15px;
            margin: 0;
            max-width: 1200px;
            margin: 0 auto;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        #messageInput {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50px;
            font-size: 16px;
            color: #333;
            outline: none;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
        }

        #messageInput::placeholder {
            color: #666;
        }

        #messageInput:focus {
            border-color: #2C5530;
            box-shadow: 0 0 0 4px rgba(44, 85, 48, 0.1);
        }



        .send-btn {
            background: linear-gradient(135deg, #2C5530 0%, #1B4332 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            min-width: 120px;
        }

        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(44, 85, 48, 0.4);
        }

        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* ====== STYLES CASE À COCHER MODE DÉCLARATION ====== */
        .declaration-mode-wrapper {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            max-width: 1200px;
            margin: 0 auto 10px auto;
        }

        .declaration-checkbox {
            width: 20px;
            height: 20px;
            border: 2px solid #2C5530;
            border-radius: 4px;
            background: transparent;
            cursor: pointer;
            position: relative;
            transition: all 0.2s ease;
        }

        .declaration-checkbox:checked {
            background: linear-gradient(135deg, #2C5530 0%, #1B4332 100%);
            border-color: #2C5530;
        }

        .declaration-checkbox:checked::after {
            content: '✓';
            position: absolute;
            top: -2px;
            left: 2px;
            color: white;
            font-size: 14px;
            font-weight: bold;
        }

        .declaration-label {
            font-weight: 500;
            color: #2C5530;
            cursor: pointer;
            font-size: 14px;
            user-select: none;
        }

        .declaration-info {
            font-size: 12px;
            color: #666;
            margin-left: 5px;
            font-style: italic;
        }

        .declaration-mode-indicator {
            display: none;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            animation: pulse 2s ease-in-out infinite;
            margin-left: 10px;
        }

        .declaration-mode-indicator.active {
            display: inline-block;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.05); }
        }



        .loading {
            display: none;
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            font-style: italic;
            font-weight: 500;
            margin: 15px 0;
            animation: loadingPulse 2s ease-in-out infinite;
        }

        @keyframes loadingPulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        .typing {
            display: inline-block;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: rgba(255, 255, 255, 0.9);
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }

        .loading-dots {
            display: inline-block;
            margin-left: 10px;
        }

        .loading-dots::after {
            content: '';
            animation: dots 1.5s steps(4, end) infinite;
        }

        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }



        /* Responsive Design */
        @media (max-width: 768px) {
            .chat-container {
                padding: 20px;
            }
            
            .chat-input-section {
                padding: 20px;
            }
            
            .input-section {
                flex-direction: column;
                gap: 15px;
            }
            
            .send-btn {
                width: 100%;
            }
            
            .message {
                margin-left: 5% !important;
                margin-right: 5% !important;
            }

            .chat-header h1 {
                font-size: 2em;
            }
        }

        @media (max-width: 480px) {
            .chat-container {
                padding: 15px;
            }
            
            .chat-input-section {
                padding: 15px;
            }
            
            .message {
                margin-left: 0 !important;
                margin-right: 0 !important;
                padding: 12px 16px;
            }

            .chat-header {
                padding: 20px;
            }

            .chat-header h1 {
                font-size: 1.8em;
            }
        }

        /* ====== BOUTONS CHATGPT ====== */
        .message {
            position: relative;
        }

        .message-actions {
            display: none;
            flex-direction: row;
            gap: 8px;
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid rgba(0, 0, 0, 0.1);
            opacity: 0;
            transform: translateY(5px);
            transition: all 0.3s ease;
        }

        .message:hover .message-actions {
            display: flex;
            opacity: 1;
            transform: translateY(0);
        }

        .message-btn {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            font-size: 12px;
            color: #333;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
        }

        .message-btn:hover {
            background: rgba(59, 130, 246, 0.1);
            border-color: rgba(59, 130, 246, 0.3);
            color: #3b82f6;
            transform: translateY(-1px);
        }

        .message-btn.edit:hover {
            background: rgba(251, 146, 60, 0.1);
            border-color: rgba(251, 146, 60, 0.3);
            color: #f59e0b;
        }

        .message-btn.regenerate:hover {
            background: rgba(34, 197, 94, 0.1);
            border-color: rgba(34, 197, 94, 0.3);
            color: #22c55e;
        }

        .message-btn.copy:hover {
            background: rgba(168, 85, 247, 0.1);
            border-color: rgba(168, 85, 247, 0.3);
            color: #a855f7;
        }

        /* Modal d'édition */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .modal-overlay.show {
            display: flex;
        }

        .modal-content {
            background: white;
            padding: 30px;
            border-radius: 15px;
            width: 90%;
            max-width: 500px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        .modal-content h3 {
            margin: 0 0 20px 0;
            color: #333;
            font-size: 1.2em;
        }

        .modal-textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border: 2px solid rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            outline: none;
            margin-bottom: 20px;
            box-sizing: border-box;
        }

        .modal-textarea:focus {
            border-color: #2C5530;
            box-shadow: 0 0 0 3px rgba(44, 85, 48, 0.1);
        }

        .modal-buttons {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
        }

        .modal-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .modal-btn.primary {
            background: linear-gradient(135deg, #2C5530 0%, #1B4332 100%);
            color: white;
        }

        .modal-btn.primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(44, 85, 48, 0.3);
        }

        .modal-btn.secondary {
            background: #f3f4f6;
            color: #374151;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        .modal-btn.secondary:hover {
            background: #e5e7eb;
        }

        /* ====== STYLES PANNEAU CONVERSATIONS ====== */
        .conversations-toggle {
            position: fixed;
            top: 20px;
            left: 20px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2C5530 0%, #1B4332 100%);
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 1001;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .conversations-toggle:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(44, 85, 48, 0.3);
        }

        .new-conversation-btn {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border: none;
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
            margin-right: 8px;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
        }

        .new-conversation-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        }

        .new-conversation-btn:active {
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
        }

        .conversations-panel {
            position: fixed;
            top: 0;
            left: 0;
            width: 350px;
            height: 100vh;
            background: white;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            display: none;
            flex-direction: column;
            border-right: 1px solid #e5e7eb;
        }

        .conversations-header {
            padding: 20px;
            border-bottom: 1px solid #e5e7eb;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .conversations-header h3 {
            margin: 0;
            font-size: 18px;
            color: #1f2937;
            font-weight: 600;
        }

        .conversations-actions {
            display: flex;
            gap: 8px;
        }

        .conv-btn {
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
        }

        .conv-btn.primary {
            background: linear-gradient(135deg, #2C5530 0%, #1B4332 100%);
            color: white;
        }

        .conv-btn.primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(44, 85, 48, 0.3);
        }

        .conv-btn.secondary {
            background: #f3f4f6;
            color: #6b7280;
        }

        .conv-btn.secondary:hover {
            background: #e5e7eb;
            color: #374151;
        }

        .conversations-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }

        .conversation-item {
            padding: 12px 15px;
            margin-bottom: 8px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid transparent;
            background: #f9fafb;
        }

        .conversation-item:hover {
            background: #f3f4f6;
            border-color: #d1d5db;
            transform: translateX(2px);
        }

        .conversation-item.active {
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            border-color: #10b981;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
        }

        .conversation-title {
            font-weight: 500;
            color: #1f2937;
            margin-bottom: 4px;
            font-size: 14px;
            line-height: 1.4;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .conversation-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .conversation-date {
            font-size: 12px;
            color: #6b7280;
        }

        .conversation-delete {
            background: none;
            border: none;
            color: #ef4444;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            opacity: 0;
            transition: all 0.2s ease;
            font-size: 14px;
        }

        .conversation-item:hover .conversation-delete {
            opacity: 1;
        }

        .conversation-delete:hover {
            background: #fef2f2;
            transform: scale(1.1);
        }

        /* Responsive pour mobile */
        @media (max-width: 768px) {
            .conversations-panel {
                width: 100%;
            }
            
            .conversations-toggle {
                top: 15px;
                left: 15px;
                width: 45px;
                height: 45px;
                font-size: 18px;
            }
            
            .new-conversation-btn {
                padding: 10px 12px;
                margin-right: 6px;
                font-size: 12px;
            }
            
            .input-section {
                gap: 8px;
            }
        }


    </style>
</head>
<body>
    <!-- Application DOCUMIND plein écran -->
    <div class="chat-app">
        <div class="container">
            <div class="chat-header">
                <h1>🇸🇳 SRMT-DOCUMIND</h1>
                <p>Assistant IA Expert en Fiscalité et Douanes Sénégalaises • Code des Impôts • Code des Douanes</p>
            </div>

        <div class="chat-container" id="chatContainer">
            <div class="message assistant-message">
                🇸🇳 Bonjour ! Je suis SRMT-DOCUMIND, votre assistant IA expert en fiscalité et douanes sénégalaises ! 
                
                🎯 Je vous accompagne dans vos démarches administratives et fiscales au Sénégal :
                • 🏛️ Code Général des Impôts (CGI) sénégalais
                • 📋 Code des Douanes du Sénégal
                • 💼 Procédures DGI et douanières
                • 📊 Calculs d'impôts, taxes et droits
                • 📄 Analyse de documents administratifs (PDF, Word, Excel...)
                
                ✨ **NOUVEAU : Mode Déclaration** 
                Activez la case "� Mode Déclaration" pour analyser vos données déclaratives (declaration.csv) exclusivement !
                
                �💡 Exemples : "Taux de TVA au Sénégal ?", "Comment déclarer mes revenus ?", "Procédure de dédouanement ?"
                📋 Mode Déclaration : "Statistiques TVA par secteur", "Entreprises en retard de déclaration"
                
                Posez-moi vos questions fiscales et douanières !
            </div>
        </div>

            <div class="loading" id="loading">
                <div class="typing"></div>
                <span>SRMT-DOCUMIND analyse votre question fiscal/douanière<span class="loading-dots"></span></span>
            </div>

            <div class="chat-input-section">
                <div class="declaration-mode-wrapper">
                    <input type="checkbox" id="declarationMode" class="declaration-checkbox" onchange="toggleDeclarationMode()">
                    <label for="declarationMode" class="declaration-label">
                        📋 Mode Déclaration
                    </label>
                    <span class="declaration-info">(Focus exclusif sur declaration.csv)</span>
                    <span id="declarationIndicator" class="declaration-mode-indicator">MODE DÉCLARATION ACTIVÉ</span>
                </div>
                <div class="input-section">
                    <input type="text" id="messageInput" placeholder="Posez votre question fiscale ou douanière (ex: taux TVA, procédure dédouanement...)..." onkeypress="checkEnter(event)">
                    <button class="new-conversation-btn" onclick="startNewConversation()" title="Nouvelle conversation">
                        ➕ Nouveau
                    </button>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                        📤 Envoyer
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Bouton flottant pour ouvrir le panneau des conversations -->
    <button id="conversationsToggle" class="conversations-toggle" onclick="toggleConversationsPanel()" title="Gérer les conversations">
        💬
    </button>

    <!-- Modal d'édition -->
    <div class="modal-overlay" id="editModal" onclick="closeEditModal()">
        <div class="modal-content" onclick="event.stopPropagation()">
            <h3>✏️ Modifier le message</h3>
            <textarea class="modal-textarea" id="editTextarea" placeholder="Modifiez votre message..."></textarea>
            <div class="modal-buttons">
                <button class="modal-btn secondary" onclick="closeEditModal()">Annuler</button>
                <button class="modal-btn primary" onclick="saveEditMessage()">💫 Envoyer</button>
            </div>
        </div>
    </div>

    <!-- Panneau de gestion des conversations -->
    <div id="conversationsPanel" class="conversations-panel">
        <div class="conversations-header">
            <h3>💬 Conversations</h3>
            <div class="conversations-actions">
                <button class="conv-btn primary" onclick="startNewConversation()" title="Nouvelle conversation">
                    ➕ Nouveau
                </button>
                <button class="conv-btn secondary" onclick="toggleConversationsPanel()" title="Fermer">
                    ✕
                </button>
            </div>
        </div>
        <div class="conversations-list" id="conversationsList">
            <!-- Liste des conversations générée dynamiquement -->
        </div>
    </div>

    <script>
        // DOCUMIND - Application plein écran

        function checkEnter(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        // ====== FONCTION MODE DÉCLARATION ======
        function toggleDeclarationMode() {
            const checkbox = document.getElementById('declarationMode');
            const indicator = document.getElementById('declarationIndicator');
            const messageInput = document.getElementById('messageInput');
            
            if (checkbox.checked) {
                // Mode Déclaration activé
                indicator.classList.add('active');
                messageInput.placeholder = '📋 Mode Déclaration activé - Recherche dans declaration.csv uniquement...';
                console.log('✅ MODE DÉCLARATION ACTIVÉ - Focus sur declaration.csv');
                
                // Optionnel: Afficher une notification
                showDeclarationModeNotification(true);
            } else {
                // Mode Déclaration désactivé
                indicator.classList.remove('active');
                messageInput.placeholder = 'Posez votre question fiscale ou douanière (ex: taux TVA, procédure dédouanement...)...';
                console.log('❌ MODE DÉCLARATION DÉSACTIVÉ - Recherche normale');
                
                // Optionnel: Afficher une notification
                showDeclarationModeNotification(false);
            }
        }

        function showDeclarationModeNotification(activated) {
            // Créer une notification temporaire
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                background: ${activated ? 'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)' : 'linear-gradient(135deg, #2C5530 0%, #1B4332 100%)'};
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                z-index: 10000;
                font-weight: 500;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                animation: slideInNotification 0.3s ease-out;
            `;
            
            notification.innerHTML = activated ? 
                '📋 MODE DÉCLARATION ACTIVÉ<br><small>Focus exclusif sur declaration.csv</small>' : 
                '📚 MODE NORMAL ACTIVÉ<br><small>Recherche dans tous les documents</small>';
            
            // Ajouter la notification au body
            document.body.appendChild(notification);
            
            // Supprimer automatiquement après 3 secondes
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.style.animation = 'slideOutNotification 0.3s ease-in';
                    setTimeout(() => {
                        if (notification.parentNode) {
                            document.body.removeChild(notification);
                        }
                    }, 300);
                }
            }, 3000);
        }

        // Ajouter les animations CSS pour les notifications
        const notificationStyles = `
            @keyframes slideInNotification {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOutNotification {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        
        // Injecter les styles
        const styleSheet = document.createElement('style');
        styleSheet.textContent = notificationStyles;
        document.head.appendChild(styleSheet);

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;

            // Vérifier si le mode déclaration est activé
            const declarationMode = document.getElementById('declarationMode').checked;
            
            const chatContainer = document.getElementById('chatContainer');
            const sendBtn = document.getElementById('sendBtn');
            const loading = document.getElementById('loading');

            // Ajouter le message utilisateur
            const userMessage = document.createElement('div');
            userMessage.className = 'message user-message';
            userMessage.textContent = message;
            chatContainer.appendChild(userMessage);

            // Vider l'input et désactiver le bouton
            input.value = '';
            sendBtn.disabled = true;
            loading.style.display = 'block';
            
            // Faire défiler vers le bas
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        message: message,
                        declaration_mode: declarationMode 
                    })
                });

                const data = await response.json();

                // Ajouter la réponse de l'assistant avec effet de frappe
                const assistantMessage = document.createElement('div');
                assistantMessage.className = 'message assistant-message';
                assistantMessage.innerHTML = '';
                chatContainer.appendChild(assistantMessage);
                
                // Effet de frappe
                const textSpan = document.createElement('span');
                assistantMessage.appendChild(textSpan);
                typewriterEffect(textSpan, data.response, 20);
                
                // Ajouter les références si disponibles
                if (data.references && data.references.length > 0) {
                    setTimeout(() => {
                        addReferencesSection(assistantMessage, data.references);
                    }, 1500);
                }
                
                // Ajouter les effets de survol
                setTimeout(() => addMessageEffects(), 1000);

            } catch (error) {
                const errorMessage = document.createElement('div');
                errorMessage.className = 'message assistant-message';
                errorMessage.textContent = 'Désolé, une erreur s\\'est produite. Veuillez réessayer.';
                errorMessage.style.color = '#e74c3c';
                chatContainer.appendChild(errorMessage);
            }

            // Réactiver le bouton et cacher le loading
            sendBtn.disabled = false;
            loading.style.display = 'none';
            
            // Faire défiler vers le bas
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Ajouter les boutons aux nouveaux messages
            setTimeout(() => {
                addChatButtons();
                // Sauvegarder automatiquement la conversation
                saveCurrentConversation();
            }, 300);
        }

        function clearChat() {
            const chatContainer = document.getElementById('chatContainer');
            
            // Animation de fade out
            chatContainer.style.opacity = '0.5';
            chatContainer.style.transform = 'scale(0.95)';
            
            setTimeout(() => {
                chatContainer.innerHTML = `
                    <div class="message assistant-message">
                        🇸🇳 Bonjour ! Je suis SRMT-DOCUMIND, votre assistant IA expert en fiscalité et douanes sénégalaises ! 
                        
                        🎯 Je vous accompagne dans vos démarches administratives et fiscales au Sénégal :
                        • 🏛️ Code Général des Impôts (CGI) sénégalais
                        • 📋 Code des Douanes du Sénégal
                        • 💼 Procédures DGI et douanières
                        • 📊 Calculs d'impôts, taxes et droits
                        • 📄 Analyse de documents administratifs (PDF, Word, Excel...)
                        
                        💡 Exemples : "Taux de TVA au Sénégal ?", "Comment déclarer mes revenus ?", "Procédure de dédouanement ?"
                        
                        Posez-moi vos questions fiscales et douanières !
                    </div>
                `;
                
                // Animation de fade in
                chatContainer.style.opacity = '1';
                chatContainer.style.transform = 'scale(1)';
            }, 200);
        }

        // Effets au survol des messages
        function addMessageEffects() {
            const messages = document.querySelectorAll('.message');
            messages.forEach(message => {
                message.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-2px) scale(1.01)';
                });
                
                message.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0) scale(1)';
                });
            });
        }

        // Effet de frappe pour les réponses
        function typewriterEffect(element, text, speed = 30) {
            element.textContent = '';
            let i = 0;
            const timer = setInterval(() => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                } else {
                    clearInterval(timer);
                }
            }, speed);
        }

        // Ajouter une section de références avec liens
        function addReferencesSection(messageElement, references) {
            const referencesDiv = document.createElement('div');
            referencesDiv.style.cssText = `
                margin-top: 15px;
                padding: 12px;
                background: rgba(102, 126, 234, 0.1);
                border-left: 3px solid #667eea;
                border-radius: 8px;
                font-size: 13px;
            `;
            
            referencesDiv.innerHTML = '<strong>📚 Références précises :</strong>';
            
            references.forEach((ref, index) => {
                const refItem = document.createElement('div');
                refItem.style.cssText = `
                    margin: 8px 0;
                    padding: 8px;
                    background: rgba(255, 255, 255, 0.7);
                    border-radius: 5px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                `;
                
                refItem.innerHTML = `
                    <div style="font-weight: 600; color: #667eea;">📄 ${ref.file_name}</div>
                    <div style="color: #666; margin: 4px 0;">
                        📍 ${ref.page_info} • ${ref.location}
                    </div>
                    <div style="color: #888; font-size: 12px; font-style: italic;">
                        "${ref.snippet}"
                    </div>
                    <div style="margin-top: 5px;">
                        <button onclick="openFile('${ref.file_path}', ${ref.line_start})" 
                                style="background: #667eea; color: white; border: none; padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">
                            🔗 Ouvrir à cette position
                        </button>
                    </div>
                `;
                
                refItem.addEventListener('mouseenter', function() {
                    this.style.background = 'rgba(102, 126, 234, 0.2)';
                    this.style.transform = 'translateX(5px)';
                });
                
                refItem.addEventListener('mouseleave', function() {
                    this.style.background = 'rgba(255, 255, 255, 0.7)';
                    this.style.transform = 'translateX(0)';
                });
                
                referencesDiv.appendChild(refItem);
            });
            
            messageElement.appendChild(referencesDiv);
        }

        // Fonction pour ouvrir un fichier à une position spécifique
        async function openFile(filePath, lineNumber) {
            try {
                const response = await fetch('/open_file', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        file_path: filePath,
                        line_number: lineNumber
                    })
                });

                const data = await response.json();
                
                if (data.success) {
                    // Animation de succès
                    const btn = event.target;
                    const originalText = btn.textContent;
                    btn.textContent = '✅ Ouvert!';
                    btn.style.background = '#27ae60';
                    
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.style.background = '#667eea';
                    }, 2000);
                } else {
                    alert('Impossible d\\'ouvrir le fichier: ' + data.error);
                }
            } catch (error) {
                alert('Erreur lors de l\\'ouverture du fichier');
                console.error('Erreur ouverture fichier:', error);
            }
        }

        // Initialisation au chargement
        window.onload = function() {
            addMessageEffects();
            
            // Focus automatique sur l'input
            document.getElementById('messageInput').focus();
            
            // Animation d'entrée de l'application
            const container = document.querySelector('.container');
            container.style.opacity = '0';
            container.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                container.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                container.style.opacity = '1';
                container.style.transform = 'translateY(0)';
            }, 100);
        };

        // ====== FONCTIONS BOUTONS CHATGPT ======

        // Ajouter les boutons aux messages existants et nouveaux
        function addChatButtons() {
            const messages = document.querySelectorAll('.message');
            messages.forEach(message => {
                if (message.querySelector('.message-actions')) return;
                
                const actionsDiv = document.createElement('div');
                actionsDiv.className = 'message-actions';
                
                if (message.classList.contains('user-message')) {
                    actionsDiv.innerHTML = `
                        <button class="message-btn edit" onclick="editMessage(this)">✏️ Modifier</button>
                        <button class="message-btn copy" onclick="copyMessage(this)">📋 Copier</button>
                    `;
                } else if (message.classList.contains('assistant-message')) {
                    actionsDiv.innerHTML = `
                        <button class="message-btn regenerate" onclick="regenerateMessage(this)">🔄 Régénérer</button>
                        <button class="message-btn copy" onclick="copyMessage(this)">📋 Copier</button>
                    `;
                }
                
                message.appendChild(actionsDiv);
            });
        }

        // Modifier un message
        function editMessage(btn) {
            console.log('✏️ Bouton modifier cliqué');
            
            const message = btn.closest('.message');
            if (!message) {
                console.error('❌ Message parent non trouvé');
                alert('Erreur: Message non trouvé');
                return;
            }
            
            console.log('📝 Message trouvé:', message);
            
            // Extraire le texte (plusieurs méthodes)
            let text = '';
            
            // Méthode 1: Premier nœud texte
            if (message.childNodes[0] && message.childNodes[0].textContent) {
                text = message.childNodes[0].textContent.trim();
                console.log('📋 Texte méthode 1:', text);
            }
            
            // Méthode 2: Span avec effet de frappe
            if (!text && message.querySelector('span')) {
                text = message.querySelector('span').textContent.trim();
                console.log('📋 Texte méthode 2 (span):', text);
            }
            
            // Méthode 3: Tout le texte moins les boutons
            if (!text) {
                const clone = message.cloneNode(true);
                const actionsDiv = clone.querySelector('.message-actions');
                if (actionsDiv) actionsDiv.remove();
                text = clone.textContent.trim();
                console.log('📋 Texte méthode 3 (clone):', text);
            }
            
            if (!text) {
                console.error('❌ Aucun texte extrait');
                alert('Erreur: Impossible d\\'extraire le texte du message');
                return;
            }
            
            console.log('✅ Texte final à éditer:', text);
            
            // Ouvrir le modal
            const modal = document.getElementById('editModal');
            const textarea = document.getElementById('editTextarea');
            
            if (!modal || !textarea) {
                console.error('❌ Éléments du modal non trouvés', {modal, textarea});
                alert('Erreur: Modal ou textarea non trouvé');
                return;
            }
            
            textarea.value = text;
            modal.classList.add('show');
            modal.messageElement = message;
            
            // Focus sur la textarea
            setTimeout(() => {
                textarea.focus();
                textarea.setSelectionRange(textarea.value.length, textarea.value.length);
            }, 100);
            
            console.log('✅ Modal ouvert avec succès');
        }

        // Fermer le modal
        function closeEditModal() {
            console.log('❌ Fermeture du modal d\\'édition');
            const modal = document.getElementById('editModal');
            if (modal) {
                modal.classList.remove('show');
                console.log('✅ Modal fermé');
            } else {
                console.error('❌ Modal non trouvé pour fermeture');
            }
        }

        // Sauvegarder le message modifié
        function saveEditMessage() {
            console.log('💾 Sauvegarde du message modifié');
            
            const modal = document.getElementById('editModal');
            const textarea = document.getElementById('editTextarea');
            
            if (!modal || !textarea) {
                console.error('❌ Éléments manquants', {modal, textarea});
                alert('Erreur: Éléments du modal manquants');
                return;
            }
            
            const newText = textarea.value.trim();
            console.log('📝 Nouveau texte:', newText);
            
            if (!newText) {
                console.log('❌ Texte vide, abandon');
                alert('Veuillez saisir un message');
                return;
            }
            
            const messageElement = modal.messageElement;
            if (!messageElement) {
                console.error('❌ Message element manquant');
                alert('Erreur: Message à modifier non trouvé');
                return;
            }
            
            console.log('🗑️ Suppression des messages à partir de:', messageElement);
            
            closeEditModal();
            
            // Supprimer les messages à partir de celui modifié
            const chatContainer = document.getElementById('chatContainer');
            const messages = Array.from(chatContainer.children);
            const messageIndex = messages.indexOf(messageElement);
            
            console.log(`🗑️ Index du message: ${messageIndex}, Total messages: ${messages.length}`);
            
            for (let i = messages.length - 1; i >= messageIndex; i--) {
                if (messages[i] && messages[i].classList && messages[i].classList.contains('message')) {
                    console.log(`🗑️ Suppression message ${i}`);
                    chatContainer.removeChild(messages[i]);
                }
            }
            
            // Renvoyer le nouveau message
            console.log('📤 Envoi du nouveau message:', newText);
            const messageInput = document.getElementById('messageInput');
            messageInput.value = newText;
            sendMessage();
        }

        // Régénérer une réponse
        function regenerateMessage(btn) {
            const assistantMsg = btn.closest('.message');
            let userMsg = assistantMsg.previousElementSibling;
            
            while (userMsg && !userMsg.classList.contains('user-message')) {
                userMsg = userMsg.previousElementSibling;
            }
            
            if (userMsg) {
                const userText = userMsg.childNodes[0].textContent || userMsg.querySelector('span')?.textContent || '';
                assistantMsg.remove();
                document.getElementById('messageInput').value = userText.trim();
                sendMessage();
            }
        }

        // Copier un message
        function copyMessage(btn) {
            const message = btn.closest('.message');
            const text = message.childNodes[0].textContent || message.querySelector('span')?.textContent || '';
            
            navigator.clipboard.writeText(text.trim()).then(() => {
                const originalText = btn.innerHTML;
                btn.innerHTML = '✅ Copié!';
                btn.style.background = 'rgba(34, 197, 94, 0.1)';
                btn.style.color = '#22c55e';
                
                setTimeout(() => {
                    btn.innerHTML = originalText;
                    btn.style.background = '';
                    btn.style.color = '';
                }, 2000);
            }).catch(() => alert('Erreur de copie'));
        }

        // ====== SYSTÈME DE MÉMOIRE DES CONVERSATIONS ======
        
        // Gestionnaire de mémoire des conversations
        class ConversationMemory {
            constructor() {
                this.currentConversationId = null;
                this.conversations = this.loadConversations();
                this.autoSaveEnabled = true;
            }
            
            // Charger toutes les conversations depuis localStorage
            loadConversations() {
                try {
                    const saved = localStorage.getItem('srmt_conversations');
                    return saved ? JSON.parse(saved) : {};
                } catch (e) {
                    console.error('❌ Erreur chargement conversations:', e);
                    return {};
                }
            }
            
            // Sauvegarder toutes les conversations
            saveConversations() {
                try {
                    localStorage.setItem('srmt_conversations', JSON.stringify(this.conversations));
                    console.log('💾 Conversations sauvegardées');
                } catch (e) {
                    console.error('❌ Erreur sauvegarde conversations:', e);
                }
            }
            
            // Créer une nouvelle conversation
            createNewConversation() {
                const id = 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                const conversation = {
                    id: id,
                    title: 'Nouvelle conversation',
                    messages: [],
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString()
                };
                
                this.conversations[id] = conversation;
                this.currentConversationId = id;
                this.saveConversations();
                
                console.log('🆕 Nouvelle conversation créée:', id);
                return id;
            }
            
            // Sauvegarder la conversation actuelle
            saveCurrentConversation() {
                if (!this.currentConversationId || !this.autoSaveEnabled) return;
                
                const chatContainer = document.getElementById('chatContainer');
                const messages = [];
                
                chatContainer.querySelectorAll('.message').forEach(msg => {
                    const isUser = msg.classList.contains('user-message');
                    let text = '';
                    
                    // Extraire le texte proprement
                    const span = msg.querySelector('span');
                    if (span && span.textContent.trim()) {
                        text = span.textContent.trim();
                    } else if (msg.childNodes[0] && msg.childNodes[0].textContent) {
                        text = msg.childNodes[0].textContent.trim();
                    } else {
                        const clone = msg.cloneNode(true);
                        const actions = clone.querySelector('.message-actions');
                        if (actions) actions.remove();
                        text = clone.textContent.trim();
                    }
                    
                    if (text && !text.includes('🇸🇳 Bonjour ! Je suis SRMT-DOCUMIND')) {
                        messages.push({
                            type: isUser ? 'user' : 'assistant',
                            content: text,
                            timestamp: new Date().toISOString()
                        });
                    }
                });
                
                if (this.conversations[this.currentConversationId]) {
                    this.conversations[this.currentConversationId].messages = messages;
                    this.conversations[this.currentConversationId].updatedAt = new Date().toISOString();
                    
                    // Générer un titre automatique basé sur le premier message
                    if (messages.length > 0 && this.conversations[this.currentConversationId].title === 'Nouvelle conversation') {
                        const firstUserMsg = messages.find(m => m.type === 'user');
                        if (firstUserMsg) {
                            this.conversations[this.currentConversationId].title = 
                                firstUserMsg.content.substring(0, 50) + (firstUserMsg.content.length > 50 ? '...' : '');
                        }
                    }
                    
                    this.saveConversations();
                    console.log('💾 Conversation sauvegardée:', this.currentConversationId);
                }
            }
            
            // Charger une conversation spécifique
            loadConversation(conversationId) {
                if (!this.conversations[conversationId]) {
                    console.error('❌ Conversation non trouvée:', conversationId);
                    return;
                }
                
                this.currentConversationId = conversationId;
                const conversation = this.conversations[conversationId];
                const chatContainer = document.getElementById('chatContainer');
                
                // Vider le chat actuel
                chatContainer.innerHTML = '';
                
                // Restaurer les messages
                conversation.messages.forEach(msg => {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${msg.type === 'user' ? 'user-message' : 'assistant-message'}`;
                    
                    if (msg.type === 'assistant') {
                        const span = document.createElement('span');
                        span.textContent = msg.content;
                        messageDiv.appendChild(span);
                    } else {
                        messageDiv.textContent = msg.content;
                    }
                    
                    chatContainer.appendChild(messageDiv);
                });
                
                // Ajouter les boutons aux messages restaurés
                setTimeout(() => {
                    addChatButtons();
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }, 100);
                
                console.log('📂 Conversation chargée:', conversationId, conversation.title);
            }
            
            // Supprimer une conversation
            deleteConversation(conversationId) {
                if (this.conversations[conversationId]) {
                    delete this.conversations[conversationId];
                    this.saveConversations();
                    
                    if (this.currentConversationId === conversationId) {
                        this.currentConversationId = null;
                        clearChat();
                    }
                    
                    console.log('🗑️ Conversation supprimée:', conversationId);
                }
            }
            
            // Obtenir la liste des conversations triées
            getConversationsList() {
                return Object.values(this.conversations)
                    .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
            }
        }
        
        // Instance globale du gestionnaire de mémoire
        const conversationMemory = new ConversationMemory();
        
        // Fonctions globales pour la gestion des conversations
        function startNewConversation() {
            console.log('🆕 Démarrage nouvelle conversation');
            conversationMemory.createNewConversation();
            clearChat();
            updateConversationsUI();
        }
        
        function saveCurrentConversation() {
            conversationMemory.saveCurrentConversation();
            updateConversationsUI();
        }
        
        function loadConversation(conversationId) {
            console.log('📂 Chargement conversation:', conversationId);
            conversationMemory.loadConversation(conversationId);
            updateConversationsUI();
        }
        
        function deleteConversation(conversationId) {
            if (confirm('Êtes-vous sûr de vouloir supprimer cette conversation ?')) {
                conversationMemory.deleteConversation(conversationId);
                updateConversationsUI();
            }
        }
        
        function toggleConversationsPanel() {
            const panel = document.getElementById('conversationsPanel');
            if (panel) {
                panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
                updateConversationsUI();
            }
        }
        
        function updateConversationsUI() {
            const conversationsList = document.getElementById('conversationsList');
            if (!conversationsList) return;
            
            const conversations = conversationMemory.getConversationsList();
            
            conversationsList.innerHTML = conversations.map(conv => `
                <div class="conversation-item ${conv.id === conversationMemory.currentConversationId ? 'active' : ''}" 
                     onclick="loadConversation('${conv.id}')">
                    <div class="conversation-title">${conv.title}</div>
                    <div class="conversation-meta">
                        <span class="conversation-date">${new Date(conv.updatedAt).toLocaleDateString('fr-FR')}</span>
                        <button class="conversation-delete" onclick="event.stopPropagation(); deleteConversation('${conv.id}')">🗑️</button>
                    </div>
                </div>
            `).join('');
        }

        // Ajouter les boutons au chargement et après chaque message
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(addChatButtons, 500);
            
            // Initialiser la mémoire des conversations
            if (!conversationMemory.currentConversationId) {
                conversationMemory.createNewConversation();
            }
            updateConversationsUI();
            
            // Gérer les touches dans le modal
            document.addEventListener('keydown', (e) => {
                const modal = document.getElementById('editModal');
                if (modal && modal.classList.contains('show')) {
                    if (e.key === 'Escape') {
                        console.log('⌨️ Touche Échap - Fermeture modal');
                        closeEditModal();
                        e.preventDefault();
                    } else if (e.key === 'Enter' && e.ctrlKey) {
                        console.log('⌨️ Ctrl+Entrée - Sauvegarde modal');
                        saveEditMessage();
                        e.preventDefault();
                    }
                }
            });
        });




    </script>
    
    <!-- Footer SRMT-DOCUMIND -->
    <div style="position: fixed; bottom: 15px; right: 20px; 
                color: rgba(255, 255, 255, 0.8); font-size: 12px; 
                background: rgba(44, 85, 48, 0.2); padding: 8px 15px; 
                border-radius: 20px; backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                z-index: 1000;">
        🇸🇳 Powered by <strong>SRMT-DOCUMIND</strong> • Expert Fiscal & Douanier
    </div>
</body>
</html>
"""

# Application Flask
app = Flask(__name__)

# Configuration CORS pour permettre l'intégration dans d'autres sites web
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

srmt_client = SrmtDocumindClient()

@app.route('/')
def home():
    """Page d'accueil"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint pour le chat avec références précises"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        declaration_mode = data.get('declaration_mode', False)
        
        if not message:
            return jsonify({
                'response': 'Veuillez saisir un message.',
                'references': []
            }), 400
        
        result = srmt_client.chat(message, declaration_mode=declaration_mode)
        return jsonify({
            'response': result.get('response', ''),
            'references': result.get('references', [])
        })
        
    except Exception as e:
        logger.error(f"Erreur chat endpoint: {e}")
        return jsonify({
            'response': 'Une erreur s\'est produite.',
            'references': []
        }), 500

@app.route('/open_file', methods=['POST'])
def open_file():
    """Endpoint pour ouvrir un fichier à une position spécifique"""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        line_number = data.get('line_number', 1)
        
        if not file_path:
            return jsonify({'error': 'Chemin de fichier manquant'}), 400
        
        success = srmt_client.open_file_at_location(file_path, line_number)
        
        if success:
            return jsonify({
                'message': f'Fichier ouvert: {Path(file_path).name} à la ligne {line_number}',
                'success': True
            })
        else:
            return jsonify({
                'error': f'Impossible d\'ouvrir le fichier: {Path(file_path).name}',
                'success': False
            }), 400
            
    except Exception as e:
        logger.error(f"Erreur open_file endpoint: {e}")
        return jsonify({'error': f'Erreur ouverture fichier: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Vérifie la santé de la connexion Ollama"""
    try:
        # Test rapide de connexion Ollama
        test_response = requests.get(
            f"{srmt_client.config.OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )
        ollama_status = "🟢 Connecté" if test_response.status_code == 200 else "🟡 Réponse inattendue"
    except:
        ollama_status = "🔴 Déconnecté"
    
    return jsonify({
        'ollama_status': ollama_status,
        'server_url': srmt_client.config.OLLAMA_BASE_URL,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint pour obtenir le statut de l'indexation"""
    try:
        # Vérifier le statut de la surveillance
        surveillance_status = "Inactive"
        auto_indexing = False
        if srmt_client.observer:
            if srmt_client.observer.is_alive():
                surveillance_status = "🔄 Active (Auto-indexation ON)"
                auto_indexing = True
            else:
                surveillance_status = "⏸️ Arrêté"
        
        # Lister les fichiers récents non indexés
        recent_files = []
        for file_path in srmt_client.watch_dir.rglob('*'):
            if file_path.is_file() and srmt_client.is_supported_file(str(file_path)):
                if not srmt_client.is_file_already_indexed(str(file_path)):
                    recent_files.append(str(file_path))
        
        status = {
            'indexed_files_count': len(srmt_client.indexed_files),
            'watch_directory': str(srmt_client.watch_dir.absolute()),
            'supported_extensions': srmt_client.config.SUPPORTED_EXTENSIONS,
            'indexed_files': [Path(f).name for f in srmt_client.indexed_files.keys()],
            'non_indexed_files': [Path(f).name for f in recent_files],
            'surveillance_status': surveillance_status,
            'auto_indexing': auto_indexing
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Erreur status endpoint: {e}")
        return jsonify({'error': 'Erreur récupération statut'}), 500

@app.route('/restart_watcher', methods=['POST'])
def restart_watcher():
    """Redémarre la surveillance automatique des fichiers"""
    try:
        logger.info("🔄 Redémarrage manuel de la surveillance automatique...")
        
        # Redémarrer la surveillance
        success = srmt_client.start_file_watcher()
        
        if success:
            return jsonify({
                'message': 'Surveillance automatique redémarrée avec succès',
                'status': 'active',
                'watch_directory': str(srmt_client.watch_dir)
            })
        else:
            return jsonify({
                'message': 'Échec du redémarrage de la surveillance',
                'status': 'inactive',
                'error': 'Impossible de démarrer l\'observer'
            }), 500
            
    except Exception as e:
        logger.error(f"Erreur restart_watcher: {e}")
        return jsonify({'error': f'Erreur redémarrage surveillance: {str(e)}'}), 500

@app.route('/force_check_new', methods=['POST'])
def force_check_new():
    """Force la vérification et indexation des nouveaux fichiers"""
    try:
        logger.info("🔍 Vérification manuelle des nouveaux fichiers...")
        
        new_files_indexed = 0
        for file_path in srmt_client.watch_dir.rglob('*'):
            if file_path.is_file() and srmt_client.is_supported_file(str(file_path)):
                if not srmt_client.is_file_already_indexed(str(file_path)):
                    logger.info(f"🆕 Indexation nouveau fichier: {file_path.name}")
                    srmt_client.index_file(str(file_path))
                    new_files_indexed += 1
        
        return jsonify({
            'message': f'{new_files_indexed} nouveaux fichiers indexés',
            'total_indexed': len(srmt_client.indexed_files)
        })
        
    except Exception as e:
        logger.error(f"Erreur check nouveaux fichiers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/force_full_reindex', methods=['POST'])
def force_full_reindex():
    """Force la réindexation complète de TOUS les fichiers (ignore le cache)"""
    try:
        # Diagnostic avant indexation
        logger.info(f"🔍 RÉINDEXATION FORCÉE COMPLÈTE")
        
        # Lister tous les fichiers supportés
        supported_files = []
        for file_path in srmt_client.watch_dir.rglob('*'):
            if file_path.is_file() and srmt_client.is_supported_file(str(file_path)):
                supported_files.append(str(file_path))
        
        # VIDER COMPLÈTEMENT le cache et ChromaDB
        srmt_client.indexed_files.clear()
        try:
            if hasattr(srmt_client, 'collection') and srmt_client.collection:
                srmt_client.create_vector_store()
                logger.info("🗑️ Base vectorielle et cache complètement vidés")
        except Exception as e:
            logger.warning(f"  Erreur vidage: {e}")
        
        # Indexation complète
        srmt_client.scan_existing_files()
        
        return jsonify({
            'message': f'Réindexation COMPLÈTE terminée: {len(supported_files)} fichiers retraités',
            'indexed_count': len(srmt_client.indexed_files),
            'files_found': len(supported_files),
            'cache_cleared': True
        })
    except Exception as e:
        logger.error(f"Erreur force_full_reindex: {e}")
        return jsonify({'error': 'Erreur réindexation complète'}), 500

@app.route('/reindex', methods=['POST'])
def smart_reindex():
    """Réindexation intelligente (respecte le cache des fichiers déjà indexés)"""
    try:
        # Diagnostic avant indexation
        logger.info(f"🔍 Scan du dossier: {srmt_client.config.WATCH_DIRECTORY}")
        
        # Lister tous les fichiers supportés
        supported_files = []
        for file_path in srmt_client.watch_dir.rglob('*'):
            if file_path.is_file() and srmt_client.is_supported_file(str(file_path)):
                supported_files.append(str(file_path))
        
        logger.info(f"   {len(supported_files)} fichiers supportés trouvés:")
        for file_path in supported_files:
            logger.info(f"   - {Path(file_path).name}")
        
        # Vider le cache ChromaDB complètement
        try:
            if hasattr(srmt_client, 'collection') and srmt_client.collection:
                srmt_client.create_vector_store()
                logger.info("🗑️ Base vectorielle vidée complètement")
            else:
                logger.info("🔄 Création nouvelle base vectorielle")
                srmt_client.create_vector_store()
        except Exception as e:
            logger.warning(f"  Erreur vidage base: {e}")
            # Fallback : créer une nouvelle collection
            try:
                srmt_client.create_vector_store()
            except Exception as e2:
                logger.error(f"  Erreur création base: {e2}")
        
        # NE PAS vider le cache local - garder la mémoire des fichiers indexés
        # srmt_client.indexed_files.clear()  # COMMENTÉ pour éviter réindexation
        
        # Relancer le scan avec respect du cache
        try:
            srmt_client.scan_existing_files()
            already_indexed = len([f for f in supported_files if srmt_client.is_file_already_indexed(f)])
            newly_indexed = len(srmt_client.indexed_files) - already_indexed
            message = f'Scan terminé: {already_indexed} déjà indexés, {newly_indexed} nouveaux fichiers traités'
            logger.info(f"✅ Indexation terminée: {len(srmt_client.indexed_files)} fichiers au total")
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation: {e}")
            message = f'Réindexation échouée: Vérifiez la connexion Ollama'
        
        return jsonify({
            'message': message,
            'indexed_count': len(srmt_client.indexed_files),
            'files_found': len(supported_files),
            'files_list': [Path(f).name for f in supported_files[:5]]  # Top 5 files
        })
    except Exception as e:
        logger.error(f"Erreur reindex endpoint: {e}")
        return jsonify({'error': 'Erreur réindexation'}), 500

@app.route('/start_indexing', methods=['POST'])
def start_indexing():
    """Démarre l'indexation initiale"""
    try:
        srmt_client.scan_existing_files()
        return jsonify({
            'message': 'Indexation démarrée',
            'indexed_count': len(srmt_client.indexed_files)
        })
    except Exception as e:
        logger.error(f"Erreur start_indexing: {e}")
        return jsonify({'error': f'Erreur indexation: {str(e)}'}), 500

@app.route('/diagnostic', methods=['GET'])
def diagnostic_files():
    """Diagnostic des fichiers indexés"""
    try:
        # Lister tous les fichiers du dossier
        all_files = []
        supported_files = []
        indexed_files = list(srmt_client.indexed_files.keys())
        
        for file_path in srmt_client.watch_dir.rglob('*'):
            if file_path.is_file():
                all_files.append(str(file_path))
                if srmt_client.is_supported_file(str(file_path)):
                    supported_files.append(str(file_path))
        
        # Compter les éléments dans ChromaDB avec diagnostic
        try:
            collection_count = srmt_client.vector_store.count()
            logger.info(f"📊 ChromaDB count: {collection_count}")
        except Exception as e:
            logger.error(f"  Erreur ChromaDB count: {e}")
            collection_count = 0
            
        # Vérifier la collection elle-même
        try:
            # Essayer de récupérer quelques documents pour tester
            test_results = srmt_client.vector_store.peek(limit=5)
            actual_chunks = len(test_results.get('documents', []))
            logger.info(f"🔍 Documents réels dans ChromaDB: {actual_chunks}")
            if actual_chunks > collection_count:
                collection_count = actual_chunks
        except Exception as e:
            logger.warning(f"  Erreur peek ChromaDB: {e}")
        
        return jsonify({
            'dossier_surveille': srmt_client.config.WATCH_DIRECTORY,
            'fichiers_totaux': len(all_files),
            'fichiers_supportes': len(supported_files),
            'fichiers_indexes': len(indexed_files),
            'chunks_chromadb': collection_count,
            'liste_supportes': [Path(f).name for f in supported_files],
            'liste_indexes': [Path(f).name for f in indexed_files],
            'formats_supportes': ['.pdf', '.txt', '.docx', '.odt']
        })
    except Exception as e:
        logger.error(f"Erreur diagnostic: {e}")
        return jsonify({'error': f'Erreur diagnostic: {str(e)}'}), 500



@app.route('/debug_context', methods=['POST'])
def debug_context():
    """Debug endpoint pour voir le contexte réel"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query manquante'}), 400
        
        # Recherche avec debug
        context = srmt_client.search_context(query, limit=3)
        
        # Récupérer aussi quelques documents de ChromaDB
        try:
            sample_docs = srmt_client.collection.peek(limit=3)
            sample_content = sample_docs.get('documents', [])[:3] if sample_docs else []
        except:
            sample_content = []
        
        return jsonify({
            'query': query,
            'context_found': context,
            'context_length': len(context) if context else 0,
            'sample_documents': sample_content,
            'collection_count': srmt_client.collection.count() if srmt_client.collection else 0
        })
        
    except Exception as e:
        logger.error(f"Erreur debug_context: {e}")
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

def cleanup():
    """Nettoyage à la fermeture"""
    try:
        if hasattr(srmt_client, 'observer') and srmt_client.observer:
            srmt_client.observer.stop()
            srmt_client.observer.join()
            logger.info("🛑 Surveillance arrêtée proprement")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {e}")

def app_srmt_documind():
    """Lance l'application SRMT-DOCUMIND"""
    print("🇸🇳 Démarrage de SRMT-DOCUMIND - Assistant Fiscal & Douanier Sénégal...")
    print("=" * 70)
    print(f"🔗 URL Ollama: {SrmtDocumindConfig.OLLAMA_BASE_URL}")
    print(f"🤖 Modèle IA: {SrmtDocumindConfig.OLLAMA_CHAT_MODEL}")
    print(f"📁 Répertoire surveillé: {SrmtDocumindConfig.WATCH_DIRECTORY}")
    print("🏛️ Spécialisation: Code des Impôts & Code des Douanes Sénégal")
    print("🌐 Démarrage de l'interface web...")
    
    try:
        app.run(
            host="0.0.0.0",
            port=8505,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n👋 Arrêt de SRMT-DOCUMIND...")
        cleanup()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        cleanup()

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup)
    app_srmt_documind()
