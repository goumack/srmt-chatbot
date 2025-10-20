"""
SRMT-DOCUMIND - Assistant IA Spécialisé Fiscal et Douanier (MODE RAG STRICT)
Assistant IA intelligent pour les contribuables sénégalais
Focalisé exclusivement sur les documents fiscaux et douaniers indexés
Version optimisée - Mode RAG strict - Réponses basées uniquement sur les documents
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
        
        # Mots-clés spécifiques aux technologies non fiscales
        non_fiscal_keywords = [
            'openshift', 'kubernetes', 'docker', 'flutter', 'android', 'ios', 
            'programmation', 'développement', 'application mobile', 'app', 
            'python', 'javascript', 'développer', 'programmer', 'coder',
            'web', 'site web', 'déployer', 'cloud', 'aws', 'azure', 'git',
            'github', 'windows', 'linux', 'mac', 'apple', 'iphone', 'samsung',
            'facebook', 'instagram', 'twitter', 'réseau social'
        ]
        
        # Vérifier d'abord si c'est une question clairement non fiscale
        for keyword in non_fiscal_keywords:
            if keyword in query_lower:
                logger.info(f"🚫 Question NON FISCALE détectée: {keyword}")
                return "non_fiscal"
        
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

    def  search_context_with_references(self, query: str, limit: int = 5) -> Dict:
        """Recherche le contexte avec références précises et déduplication intelligente"""
        if not self.collection:
            logger.warning("  Aucune collection ChromaDB disponible")
            return {"context": "", "references": []}
        
        try:
            # Détecter le domaine de la question
            query_domain = self.detect_query_domain(query)
            
            # Recherche spécialisée pour les articles
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
                            'snippet': doc[:150] + "..." if len(doc) > 150 else doc,
                            'content': doc  # Ajouter le contenu complet pour le filtrage
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
        """Mode RAG strict - Force l'utilisation exclusive des documents indexés"""
        # Fonction simplifiée en mode RAG strict
        return True

    def is_greeting_or_general(self, message: str) -> bool:
        """Détecte si le message est une simple salutation - Version minimaliste pour mode RAG strict"""
        message_lower = message.lower().strip()
        
        # En mode RAG strict, seules les salutations très simples sont traitées différemment
        greeting_words = ['salut', 'bonjour', 'bonsoir', 'hello', 'hi', 'hey']
        
        # Salutations simples uniquement
        if any(greeting in message_lower for greeting in greeting_words):
            # Maximum 3 mots pour une salutation
            if len(message_lower.split()) <= 3:
                return True
        return False
    
    def generate_greeting_response(self, message: str) -> str:
        """Génère une réponse simplifiée aux salutations - Mode RAG strict"""
        # En mode RAG strict, réponse unique et courte qui rappelle la spécialisation fiscale
        return """Bonjour ! Je suis SRMT-DOCUMIND, votre assistant IA spécialisé UNIQUEMENT en fiscalité sénégalaise.

⚠️ MODE RAG STRICT : Je réponds exclusivement sur la base des documents fiscaux indexés.

🔍 Posez-moi vos questions sur :
• Code Général des Impôts (CGI) du Sénégal
• Code des Douanes sénégalais
• Articles et textes fiscaux sénégalais"""
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
                    "num_ctx": 1024,     # Contexte réduit pour salutation rapide
                    "num_predict": 150,   # Limite tokens pour réponse courte
                    "temperature": 0.7,  # Plus de créativité pour les salutations
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=20  # Timeout réduit pour salutations rapides
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

    
    def _format_direct_response(self, message: str, references: list) -> dict:
        """
        Formate une réponse en affichant directement les extraits des documents
        (utilisé en fallback si Ollama timeout)
        """
        response_parts = []
        
        response_parts.append(f"📋 RÉPONSES TROUVÉES DANS LES DOCUMENTS")
        response_parts.append(f"Question: {message}")
        response_parts.append("=" * 70)
        
        if not references:
            response_parts.append("\n⚠️ Aucune référence trouvée.")
            return {
                "response": "\n".join(response_parts),
                "references": []
            }
        
        # Afficher toutes les références trouvées
        for idx, ref in enumerate(references, 1):
            article_ref = ref.get('article_ref', 'Section')
            file_name = ref.get('file_name', 'Document')
            content = ref.get('content', ref.get('snippet', ''))
            page = ref.get('page', 'N/A')
            start_line = ref.get('start_line', 'N/A')
            end_line = ref.get('end_line', 'N/A')
            
            response_parts.append(f"\n📄 RÉFÉRENCE {idx}")
            response_parts.append(f"Source: {file_name}")
            response_parts.append(f"Article: {article_ref}")
            response_parts.append(f"Localisation: Page {page}, lignes {start_line}-{end_line}")
            response_parts.append(f"\n📖 TEXTE EXACT DU DOCUMENT:")
            response_parts.append(f'"{content}"')
            response_parts.append("\n" + "-" * 70)
        
        response_parts.append(f"\n⚠️ IMPORTANT: Les textes ci-dessus sont des extraits EXACTS des documents officiels.")
        response_parts.append(f"Aucune modification n'a été apportée au contenu.")
        response_parts.append(f"\n📊 Total: {len(references)} référence(s) trouvée(s) et triée(s) par pertinence.")
        
        return {
            "response": "\n".join(response_parts),
            "references": references
        }
    
    def is_fiscal_related_question(self, message: str) -> bool:
        """Utilise le modèle Mistral pour déterminer si la question est liée à la fiscalité"""
        try:
            prompt = f"""Détermine si la question suivante est liée à la fiscalité, aux impôts, aux douanes, ou à la réglementation fiscale sénégalaise.
            
Question: "{message}"

Réponds uniquement par 'OUI' si la question concerne la fiscalité, les impôts, les douanes, ou la réglementation fiscale.
Réponds uniquement par 'NON' si la question ne concerne PAS la fiscalité mais un autre sujet comme la technologie, le sport, la cuisine, etc.
IMPORTANT: Réponds UNIQUEMENT par 'OUI' ou 'NON'."""

            payload = {
                "model": self.config.OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Température faible pour des réponses cohérentes
                    "top_p": 0.9,
                    "max_tokens": 10  # On a besoin que d'un seul mot
                }
            }
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()['response'].strip().upper()
                is_fiscal = result == 'OUI'
                logger.info(f"🔍 Classification par Mistral: {message[:50]}... -> {'FISCALE' if is_fiscal else 'NON FISCALE'}")
                return is_fiscal
            else:
                # En cas d'erreur, utiliser la méthode par défaut
                logger.warning(f"⚠️ Erreur lors de la classification par Mistral: {response.status_code}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la classification: {e}")
            # En cas d'erreur, considérer que c'est une question fiscale par défaut
            return True

    def chat(self, message: str) -> Dict:
        """Génère une réponse basée uniquement sur les documents indexés (mode RAG strict)"""
        try:
            # Salutations: répondre directement sans recherche documentaire
            if self.is_greeting_or_general(message):
                response_text = self.generate_natural_greeting_response(message)
                return {"response": response_text, "references": []}
            
            
            # Détecter si c'est une question non fiscale avec Mistral
            if not self.is_fiscal_related_question(message):
                return {
                    "response": f"""⚠️ QUESTION NON FISCALE DÉTECTÉE

Je suis uniquement conçu pour répondre à des questions liées à la fiscalité sénégalaise.
Je ne peux pas répondre à votre question car elle n'est pas liée au domaine fiscal ou douanier.

� **Suggestions :**
- Posez une question sur le Code des Impôts/Douanes sénégalais
- Utilisez des termes fiscaux précis (TVA, IS, dédouanement)
- Mentionnez un article spécifique si possible

ℹ️ En mode RAG strict, je ne réponds qu'aux questions fiscales basées sur les documents.""",
                    "references": []
                }
                
# RECHERCHE EN MODE RAG STRICT AVEC REFORMULATIONS ET FILTRAGE TEXTUEL
            logger.info(f"🔍 Recherche documentaire pour: '{message[:50]}...'")
            
            message_lower = message.lower()
            
            # ÉTAPE 0: Recherche textuelle directe pour les questions spécifiques
            # Pour les questions sur le taux de TVA, chercher directement "18" et "TVA"
            # Pour les questions sur le taux de TAF, chercher directement "17" et "activités financières"
            direct_search_refs = []
            
            # Question sur le taux de TVA
            if ("taux" in message_lower and "tva" in message_lower) or ("tva" in message_lower and ("combien" in message_lower or "quel" in message_lower)):
                logger.info("🎯 Détection question sur taux TVA - recherche textuelle directe")
                
                # Récupérer TOUS les documents de la collection
                try:
                    all_docs = self.collection.get(
                        where={"file_name": "Senegal-Code-des-impot.pdf"},
                        include=['documents', 'metadatas']
                    )
                    
                    if all_docs and all_docs['documents']:
                        logger.info(f"📊 Analyse de {len(all_docs['documents'])} documents pour trouver le taux de TVA")
                        
                        # Chercher les documents qui contiennent "18" ET ("tva" OU "taxe sur la valeur ajoutée") ET ("taux" OU "fixé")
                        for idx, doc in enumerate(all_docs['documents']):
                            doc_lower = doc.lower()
                            metadata = all_docs['metadatas'][idx] if all_docs['metadatas'] else {}
                            
                            # Critères de recherche stricts pour le taux de TVA
                            has_18 = "18" in doc_lower and ("%" in doc_lower or "pour cent" in doc_lower)
                            has_tva = "tva" in doc_lower or "taxe sur la valeur ajoutée" in doc_lower
                            has_taux = "taux" in doc_lower or "fixé" in doc_lower
                            
                            if has_18 and has_tva and has_taux:
                                logger.info(f"✅ TVA TROUVÉE! Document avec '18%' + 'TVA' + 'taux' à la page {metadata.get('page_start', '?')}")
                                
                                # Créer une référence avec score maximum
                                ref = {
                                    'file_name': metadata.get('file_name', ''),
                                    'file_path': metadata.get('file_path', ''),
                                    'page': metadata.get('page_start', 1),
                                    'page_start': metadata.get('page_start', 1),
                                    'page_end': metadata.get('page_end', 1),
                                    'start_line': metadata.get('line_start', 1),
                                    'end_line': metadata.get('line_end', 1),
                                    'line_start': metadata.get('line_start', 1),
                                    'line_end': metadata.get('line_end', 1),
                                    'article_ref': metadata.get('article_ref', ''),
                                    'content': doc,
                                    'snippet': doc[:200] + "..." if len(doc) > 200 else doc,
                                    '_score': 1000  # Score maximum pour correspondance exacte
                                }
                                direct_search_refs.append(ref)
                        
                        if direct_search_refs:
                            logger.info(f"🎯 TVA: {len(direct_search_refs)} références exactes trouvées avec recherche textuelle")
                except Exception as e:
                    logger.error(f"Erreur recherche textuelle TVA: {e}")
            
            # Question sur le taux de la Taxe sur les Activités Financières (TAF)
            elif ("taux" in message_lower and ("activités financières" in message_lower or "activite financiere" in message_lower or "activité financière" in message_lower or "taf" in message_lower)):
                logger.info("🎯 Détection question sur taux TAF - recherche textuelle directe")
                
                try:
                    all_docs = self.collection.get(
                        where={"file_name": "Senegal-Code-des-impot.pdf"},
                        include=['documents', 'metadatas']
                    )
                    
                    if all_docs and all_docs['documents']:
                        logger.info(f"📊 Analyse de {len(all_docs['documents'])} documents pour trouver le taux TAF")
                        
                        # Chercher: "17" ET ("activités financières" OU "TAF") ET ("taux" OU "%")
                        for idx, doc in enumerate(all_docs['documents']):
                            doc_lower = doc.lower()
                            metadata = all_docs['metadatas'][idx] if all_docs['metadatas'] else {}
                            
                            # Critères pour TAF
                            has_17 = "17" in doc_lower and ("%" in doc_lower or "pour cent" in doc_lower)
                            has_taf = "activités financières" in doc_lower or "activite financiere" in doc_lower or "taf" in doc_lower
                            has_taux = "taux" in doc_lower or "taxe" in doc_lower
                            
                            # Log de debug pour Article 404
                            article_ref = metadata.get('article_ref', '')
                            if "404" in article_ref or (has_17 and has_taf):
                                logger.info(f"🔍 Article {article_ref}: has_17={has_17}, has_taf={has_taf}, has_taux={has_taux}")
                            
                            if has_17 and has_taf and has_taux:
                                logger.info(f"✅ TAF TROUVÉE! Document avec '17%' + 'activités financières' + 'taux' à l'article {metadata.get('article_ref', '?')}")
                                
                                ref = {
                                    'file_name': metadata.get('file_name', ''),
                                    'file_path': metadata.get('file_path', ''),
                                    'page': metadata.get('page_start', 1),
                                    'page_start': metadata.get('page_start', 1),
                                    'page_end': metadata.get('page_end', 1),
                                    'start_line': metadata.get('line_start', 1),
                                    'end_line': metadata.get('line_end', 1),
                                    'line_start': metadata.get('line_start', 1),
                                    'line_end': metadata.get('line_end', 1),
                                    'article_ref': metadata.get('article_ref', ''),
                                    'content': doc,
                                    'snippet': doc[:200] + "..." if len(doc) > 200 else doc,
                                    '_score': 1000
                                }
                                direct_search_refs.append(ref)
                        
                        if direct_search_refs:
                            logger.info(f"🎯 TAF: {len(direct_search_refs)} références exactes trouvées avec recherche textuelle")
                except Exception as e:
                    logger.error(f"Erreur recherche textuelle TAF: {e}")
            
            # Question sur les cigarettes (base imposable, taxes spécifiques)
            elif ("cigarette" in message_lower or "tabac" in message_lower) and ("base" in message_lower or "imposable" in message_lower or "taxe" in message_lower):
                logger.info("🎯 Détection question sur cigarettes - recherche textuelle directe (Article 408 + Arrêté)")
                
                try:
                    all_docs = self.collection.get(
                        where={"file_name": "Senegal-Code-des-impot.pdf"},
                        include=['documents', 'metadatas']
                    )
                    
                    if all_docs and all_docs['documents']:
                        logger.info(f"📊 Analyse de {len(all_docs['documents'])} documents pour Article 408 cigarettes")
                        
                        # Chercher: (Article 408 OU Arrêté 019479) ET (cigarette OU tabac) ET (base imposable OU 280 FCFA)
                        for idx, doc in enumerate(all_docs['documents']):
                            doc_lower = doc.lower()
                            metadata = all_docs['metadatas'][idx] if all_docs['metadatas'] else {}
                            article_ref = metadata.get('article_ref', '')
                            
                            # Critères pour cigarettes
                            is_article_408 = "408" in article_ref or "article 408" in doc_lower
                            is_arrete = "019479" in doc or "arrêté" in doc_lower
                            has_cigarette = "cigarette" in doc_lower or "tabac" in doc_lower
                            has_base = "base imposable" in doc_lower or "280" in doc or "prix" in doc_lower
                            
                            # Log de debug
                            if is_article_408 or is_arrete:
                                logger.info(f"🔍 Doc {article_ref}: art408={is_article_408}, arrêté={is_arrete}, cigarette={has_cigarette}, base={has_base}")
                            
                            if (is_article_408 or is_arrete) and has_cigarette and has_base:
                                logger.info(f"✅ CIGARETTES TROUVÉE! Article 408 ou Arrêté 019479 à l'article {article_ref}")
                                
                                ref = {
                                    'file_name': metadata.get('file_name', ''),
                                    'file_path': metadata.get('file_path', ''),
                                    'page': metadata.get('page_start', 1),
                                    'page_start': metadata.get('page_start', 1),
                                    'page_end': metadata.get('page_end', 1),
                                    'start_line': metadata.get('line_start', 1),
                                    'end_line': metadata.get('line_end', 1),
                                    'line_start': metadata.get('line_start', 1),
                                    'line_end': metadata.get('line_end', 1),
                                    'article_ref': article_ref,
                                    'content': doc,
                                    'snippet': doc[:200] + "..." if len(doc) > 200 else doc,
                                    '_score': 1000
                                }
                                direct_search_refs.append(ref)
                        
                        if direct_search_refs:
                            logger.info(f"🎯 CIGARETTES: {len(direct_search_refs)} références exactes trouvées (Article 408)")
                except Exception as e:
                    logger.error(f"Erreur recherche textuelle cigarettes: {e}")
            
            # Générer des reformulations de la question pour améliorer la recherche
            reformulations = [message]  # Toujours inclure la question originale
            
            # Pour les questions sur les taux
            if "quel" in message_lower and "taux" in message_lower and "tva" in message_lower:
                reformulations.extend([
                    "Le taux de la TVA est fixé à 18%",
                    "Article 369 taux TVA dix-huit pour cent",
                    "taux TVA 18 pourcentage"
                ])
            elif "taux" in message_lower and "tva" in message_lower:
                reformulations.extend([
                    "Le taux de la TVA est de 18%",
                    "TVA 18 pourcentage",
                    "Article taux TVA pourcentage"
                ])
            elif "taux" in message_lower and ("activités financières" in message_lower or "activite financiere" in message_lower):
                reformulations.extend([
                    "Article 404 taux taxe activités financières",
                    "taux taxe activités financières 17%",
                    "taxe activités financières 17 pour cent",
                    "TAF taux dix-sept pourcent"
                ])
            elif "taux" in message_lower and ("activité financière" in message_lower or "activites financieres" in message_lower):
                reformulations.extend([
                    "Article 404 taux taxe activités financières",
                    "taux taxe activités financières 17%",
                    "taxe activités financières 17 pour cent"
                ])
            
            # Pour les questions sur cigarettes/tabac
            if ("cigarette" in message_lower or "tabac" in message_lower) and ("base" in message_lower or "imposable" in message_lower or "taxe" in message_lower):
                reformulations.extend([
                    "Article 408 base imposable cigarette",
                    "Arrêté 019479 cigarette base imposable",
                    "cigarette 280 FCFA prix minimum",
                    "taxe spécifique cigarette paquet",
                    "base imposable tabac fabrication locale importation"
                ])
            
            # Pour les questions sur procédures
            if "procédure" in message_lower or "comment" in message_lower:
                reformulations.append(message_lower.replace("comment", "procédure"))
            
            # Étape 1: Recherche ÉTENDUE avec toutes les reformulations (augmenter à 20 résultats par requête)
            all_contexts = []
            all_references = []
            all_scores = {}  # Pour tracker le score de pertinence de chaque référence
            
            for idx, query in enumerate(reformulations):
                search_result = self.search_context_with_references(query, limit=20)  # Augmenté de 15 à 20
                if search_result.get("context"):
                    all_contexts.append(search_result.get("context", ""))
                    refs = search_result.get("references", [])
                    
                    # Attribuer un score de pertinence basé sur l'ordre de la requête et la position
                    for pos, ref in enumerate(refs):
                        ref_key = (ref.get('file_name'), ref.get('page'), ref.get('start_line'))
                        # Score: première requête = meilleur score, première position = meilleur score
                        score = (len(reformulations) - idx) * 100 + (15 - pos)
                        
                        if ref_key in all_scores:
                            all_scores[ref_key] = max(all_scores[ref_key], score)  # Garder le meilleur score
                        else:
                            all_scores[ref_key] = score
                        
                        # Ajouter le score à la référence
                        ref['_score'] = score
                    
                    all_references.extend(refs)
            
            # Ajouter les références de recherche textuelle directe (score maximum)
            if direct_search_refs:
                logger.info(f"➕ Ajout de {len(direct_search_refs)} références de recherche textuelle directe")
                all_references.extend(direct_search_refs)
                # Ajouter aussi leurs contextes
                for ref in direct_search_refs:
                    if ref.get('content'):
                        all_contexts.append(ref['content'])
            
            # Dédupliquer, filtrer et trier les résultats par pertinence
            if all_contexts or direct_search_refs:
                # VALIDATION DE PERTINENCE STRICTE sur TOUTES les références
                message_keywords = set(message_lower.replace("?", "").split())
                
                # Mots-clés spécifiques à détecter
                looking_for_tva = "tva" in message_lower or "taxe sur la valeur" in message_lower
                looking_for_taf = "activités financières" in message_lower or "activite financiere" in message_lower or "taf" in message_lower
                looking_for_douane = "douane" in message_lower or "dédouanement" in message_lower
                
                # Dédupliquer d'abord toutes les références
                seen_refs = {}
                for ref in all_references:
                    ref_key = (ref.get('file_name'), ref.get('page'), ref.get('start_line'))
                    if ref_key not in seen_refs:
                        seen_refs[ref_key] = ref
                    else:
                        # Garder le score le plus élevé si déjà vu
                        if ref.get('_score', 0) > seen_refs[ref_key].get('_score', 0):
                            seen_refs[ref_key] = ref
                
                unique_refs = list(seen_refs.values())
                logger.info(f"📊 {len(unique_refs)} références uniques trouvées")
                
                # Filtrer les références non pertinentes
                filtered_references = []
                for ref in unique_refs:
                    # Récupérer le contexte de cette référence
                    ref_content = ref.get('content', '').lower()
                    is_relevant = True
                    
                    # Si on cherche la TVA, rejeter les contextes qui parlent d'autres taxes
                    if looking_for_tva:
                        # Rejeter si c'est une autre taxe (pas TVA)
                        if "taxe sur les activités financières" in ref_content:
                            logger.warning(f"⚠️ Rejeté (page {ref.get('page')}): Taxe sur activités financières (pas TVA)")
                            is_relevant = False
                        elif "contribution forfaitaire" in ref_content and "tva" not in ref_content:
                            logger.warning(f"⚠️ Rejeté (page {ref.get('page')}): Contribution forfaitaire (pas TVA)")
                            is_relevant = False
                        elif "obligations" in ref_content and "émises" in ref_content and "tva" not in ref_content:
                            logger.warning(f"⚠️ Rejeté (page {ref.get('page')}): Taux sur obligations (pas TVA)")
                            is_relevant = False
                        # Accepter seulement si on trouve "tva" ou "taxe sur la valeur ajoutée"
                        elif "tva" not in ref_content and "taxe sur la valeur ajoutée" not in ref_content:
                            logger.warning(f"⚠️ Rejeté (page {ref.get('page')}): Ne mentionne pas explicitement TVA")
                            is_relevant = False
                    
                    # Si on cherche la TAF, rejeter les contextes qui ne parlent PAS de TAF
                    elif looking_for_taf:
                        ref_file = ref.get('file_name', '').lower()
                        # Rejeter si c'est le Code des DOUANES (on veut le Code des Impôts)
                        if "douane" in ref_file:
                            logger.warning(f"⚠️ Rejeté: Article {ref.get('article_ref', '?')} du Code des Douanes (on cherche TAF dans Code des Impôts)")
                            is_relevant = False
                        # Accepter si contient "financier" ou "financi" (pour gérer l'OCR avec espaces)
                        elif not any(word in ref_content for word in ["financier", "financi", "taf"]):
                            logger.warning(f"⚠️ Rejeté (page {ref.get('page')}): Ne mentionne pas activités financières")
                            is_relevant = False
                        # Rejeter si c'est la table des matières ou introduction générique
                        elif ("livre premier" in ref_content or "titre i" in ref_content or "chapitre premier" in ref_content):
                            # Mais accepter si contient le taux 17
                            if "17" not in ref_content and "%" not in ref_content:
                                logger.warning(f"⚠️ Rejeté (page {ref.get('page')}): Table des matières ou introduction (pas le taux)")
                                is_relevant = False
                    
                    if is_relevant:
                        filtered_references.append(ref)
                
                # Trier par score de pertinence (du plus haut au plus bas)
                filtered_references.sort(key=lambda x: x.get('_score', 0), reverse=True)
                
                logger.info(f"✅ {len(filtered_references)} références pertinentes après filtrage")
                
                # Prendre les TOP 20 références les plus pertinentes (augmenté de 10 à 20)
                references = filtered_references[:20]
                
                # Construire le contexte à partir de TOUTES les références (pas seulement les 5 meilleures)
                filtered_contexts = []
                for ref in references:  # TOUTES les références pour que le modèle les voie
                    if ref.get('content'):
                        filtered_contexts.append(ref['content'])
                
                # Utiliser TOUS les contextes filtrés (réduit de 10 à 5 pour accélérer)
                if filtered_contexts:
                    context = "\n\n".join(filtered_contexts[:5])  # Top 5 contextes au lieu de 10
                    logger.info(f"📄 Contexte construit à partir de {len(filtered_contexts)} extraits")
                else:
                    # Si le filtrage a tout rejeté, réessayer avec recherche plus spécifique
                    logger.warning("⚠️ Tous les contextes rejetés - Recherche plus spécifique...")
                    if looking_for_tva:
                        # Recherche ultra-spécifique pour TVA
                        specific_query = "Article taux taxe sur la valeur ajoutée TVA"
                        search_result = self.search_context_with_references(specific_query, limit=15)
                        context = search_result.get("context", "")
                        filtered_references = search_result.get("references", [])
                    elif looking_for_taf:
                        # Recherche ultra-spécifique pour TAF
                        specific_query = "Article 404 taux taxe activités financières 17%"
                        search_result = self.search_context_with_references(specific_query, limit=15)
                        context = search_result.get("context", "")
                        filtered_references = search_result.get("references", [])
                    else:
                        context = "\n\n".join(all_contexts[:3])  # Fallback sur les résultats non filtrés
                        filtered_references = all_references
                
                # Dédupliquer les références par file_name + page
                seen = set()
                unique_refs = []
                for ref in filtered_references:
                    key = (ref.get('file_name'), ref.get('page'))
                    if key not in seen:
                        seen.add(key)
                        unique_refs.append(ref)
                references = unique_refs[:20]  # Augmenté de 10 à 20 références max
                logger.info(f"✅ Trouvé avec reformulations: {len(reformulations)} variantes testées")
            else:
                context = ""
                references = []
            
            # Étape 2: Si toujours aucun résultat, essai avec mots-clés extraits
            if not context:
                keywords = [word for word in message.split() if len(word) > 3]
                if keywords:
                    keyword_query = " ".join(keywords)
                    logger.info(f"🔄 Recherche secondaire avec mots-clés: '{keyword_query}'")
                    search_result2 = self.search_context_with_references(keyword_query, limit=10)
                    context = search_result2.get("context", "")
                    references = search_result2.get("references", [])
            
            # FORCER l'utilisation du contexte des documents
            if context and context.strip():
                # Détecter le domaine de la question pour validation
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
                        search_result = self.search_context_with_references(message, limit=5)
                        context = search_result.get("context", "")
                        references = search_result.get("references", [])
                
                # Vérifier la pertinence du contexte
                question_keywords = message.lower().split()
                context_lower = context.lower()
                keyword_found = any(kw in context_lower for kw in question_keywords if len(kw) > 3)
                
                if keyword_found or any(keyword in context_lower for keyword in ["impot", "tva", "douane", "fiscal", "cgi", "dgi", "senegal", "sénégal", "article"]):
                    # Identifier le code source précisément pour une réponse ciblée
                    code_source = "Document juridique sénégalais"
                    if references:
                        file_name = references[0].get('file_name', '').lower()
                        if 'impot' in file_name:
                            code_source = "Code Général des Impôts (CGI)"
                        elif 'douane' in file_name:
                            code_source = "Code des Douanes"
                    
                    prompt = f"""Tu es SRMT-DOCUMIND, assistant IA expert en fiscalité et législation douanière sénégalaise.

QUESTION: {message}

DOCUMENTS FOURNIS ({code_source.upper()}):
{context}

=== INSTRUCTIONS CRITIQUES ===

🎯 TA MISSION:
Analyse les documents fournis et réponds à la question en citant EXACTEMENT les textes trouvés.

⚠️ RÈGLES ABSOLUES - ZÉRO HALLUCINATION:

1. TU NE DOIS RÉPONDRE QU'AVEC CE QUI EST DANS LES DOCUMENTS CI-DESSUS
   - Cite les articles tels qu'ils apparaissent (ex: Article 404, Article 408)
   - Copie les chiffres/taux EXACTEMENT comme écrits (17% reste 17%, pas 6% ou autre chose)
   - Ne change RIEN au texte original

2. SI PLUSIEURS DOCUMENTS RÉPONDENT À LA QUESTION:
   - Liste-les TOUS séparément
   - Ne mélange pas les informations de différents articles
   - Indique clairement la source de chaque information

3. SI AUCUN DOCUMENT NE RÉPOND:
   - Dis clairement: "Les documents fournis ne contiennent pas d'information sur [sujet demandé]"
   - N'invente JAMAIS d'articles ou de textes juridiques
   - Ne déduis pas, ne suppose pas, n'extrapole pas

4. INTERDICTIONS STRICTES:
   ❌ N'invente pas de numéros d'articles (ex: Article 238-0 n'existe pas si pas mentionné)
   ❌ Ne fabrique pas de texte juridique
   ❌ Ne modifie pas les chiffres
   ❌ Ne cite pas un article différent de celui qui répond vraiment à la question

� RAPPEL CRITIQUE:
Tu es un assistant juridique. Une erreur peut avoir des conséquences légales graves.
PRÉCISION = ZÉRO TOLÉRANCE pour les inventions.

Maintenant analyse les documents et réponds:"""
                else:
                    return {
                        "response": f"""⚠️ INFORMATION NON TROUVÉE

Je ne trouve pas d'information sur ce sujet dans les documents fiscaux indexés.

📌 **Suggestions :**
- Utilisez des termes précis du Code des Impôts/Douanes
- Référencez un article spécifique (ex: "Article 19 du CGI")
- Reformulez avec des termes fiscaux sénégalais

ℹ️ En mode RAG strict, je réponds uniquement sur la base des documents.""",
                        "references": references
                    }
            else:
                return {
                    "response": f"""⚠️ AUCUN DOCUMENT CORRESPONDANT

Je suis uniquement conçu pour répondre à des questions liées à la fiscalité sénégalaise.

📊 **Suggestions :**
- Posez une question sur le Code des Impôts/Douanes sénégalais
- Utilisez des termes fiscaux précis (TVA, IS, dédouanement)
- Mentionnez un article spécifique si possible

ℹ️ En mode RAG strict, je ne réponds qu'aux questions fiscales basées sur les documents.""",
                    "references": []
                }
            
            # Générer la réponse avec Ollama - MODE RAG STRICT AVEC CITATIONS
            payload = {
                "model": self.config.OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Température à 0 pour réponses exactes
                    "top_p": 0.9,
                    "repeat_penalty": 1.2,
                    "num_ctx": 4096  # Contexte réduit pour accélérer (de 8192 à 4096)
                }
            }
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=300  # Timeout augmenté à 5 minutes
            )
            
            if response.status_code == 200:
                ollama_response = response.json()['response']
                return {
                    "response": ollama_response,
                    "references": references
                }
            else:
                return {
                    "response": f"❌ Erreur technique (code {response.status_code}). Veuillez réessayer.",
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
        :root {
            --senegal-green: #00853F;
            --senegal-yellow: #FDEF42;
            --senegal-red: #E31B23;
            --primary-gradient: linear-gradient(135deg, var(--senegal-green) 0%, #006838 100%);
            --secondary-gradient: linear-gradient(135deg, #1a472a 0%, #0d2818 100%);
            --accent-color: var(--senegal-yellow);
            --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.1);
            --shadow-medium: 0 12px 48px rgba(0, 0, 0, 0.15);
            --shadow-hard: 0 20px 60px rgba(0, 0, 0, 0.25);
            --transition-smooth: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-bounce: all 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, var(--senegal-green) 0%, #006838 30%, #004d2a 100%);
            background-attachment: fixed;
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
            position: relative;
        }

        /* Effet de particules animées en arrière-plan */
        body::before {
            content: '';
            position: fixed;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(254, 239, 66, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(0, 133, 63, 0.08) 0%, transparent 50%);
            animation: backgroundPulse 15s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }

        @keyframes backgroundPulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.05); }
        }



        /* Application plein écran */
        .chat-app {
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(20px);
            position: relative;
            z-index: 1;
            box-shadow: 0 0 80px rgba(0, 0, 0, 0.15);
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
            background: var(--primary-gradient);
            color: white;
            padding: 40px 50px;
            text-align: center;
            position: relative;
            box-shadow: var(--shadow-hard);
            overflow: hidden;
            z-index: 10;
        }

        /* Drapeau sénégalais animé avec effet de vague */
        .chat-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                linear-gradient(90deg, 
                    var(--senegal-green) 0%, 
                    var(--senegal-green) 33%, 
                    var(--senegal-yellow) 33%, 
                    var(--senegal-yellow) 66%, 
                    var(--senegal-red) 66%, 
                    var(--senegal-red) 100%);
            opacity: 0.15;
            z-index: -1;
            animation: flagWave 8s ease-in-out infinite;
        }

        @keyframes flagWave {
            0%, 100% { transform: translateX(0) scaleX(1); }
            50% { transform: translateX(-2%) scaleX(1.02); }
        }

        /* Étoile centrale du drapeau (décoration subtile) */
        .chat-header::after {
            content: '★';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 180px;
            color: var(--senegal-yellow);
            opacity: 0.06;
            z-index: -1;
            animation: starPulse 4s ease-in-out infinite;
        }

        @keyframes starPulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1) rotate(0deg); }
            50% { transform: translate(-50%, -50%) scale(1.1) rotate(10deg); }
        }

        .chat-header h1 {
            margin: 0;
            font-size: 2.8em;
            font-weight: 800;
            text-shadow: 
                2px 2px 4px rgba(0, 0, 0, 0.3),
                0 0 40px rgba(254, 239, 66, 0.3);
            letter-spacing: 1px;
            animation: titleSlideIn 1s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            position: relative;
            z-index: 2;
        }

        @keyframes titleSlideIn {
            from {
                opacity: 0;
                transform: translateY(-30px) scale(0.9);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        .chat-header p {
            margin: 15px 0 0 0;
            font-size: 1.15em;
            opacity: 0.95;
            font-weight: 400;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
            animation: subtitleFadeIn 1.2s ease-out 0.3s both;
            position: relative;
            z-index: 2;
        }

        @keyframes subtitleFadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 0.95;
                transform: translateY(0);
            }
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
            margin-bottom: 24px;
            padding: 20px 28px;
            border-radius: 24px;
            animation: messageSlideIn 0.7s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            position: relative;
            backdrop-filter: blur(15px);
            transition: var(--transition-smooth);
            transform-origin: left center;
        }

        .message:hover {
            transform: translateX(8px) scale(1.02);
            box-shadow: var(--shadow-medium);
        }

        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateX(-50px) translateY(20px) scale(0.9) rotate(-2deg);
            }
            to {
                opacity: 1;
                transform: translateX(0) translateY(0) scale(1) rotate(0deg);
            }
        }

        .user-message {
            background: var(--primary-gradient);
            color: white;
            margin-left: 20%;
            text-align: right;
            box-shadow: 
                0 8px 24px rgba(0, 133, 63, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            border: none;
            transform-origin: right center;
            position: relative;
            overflow: hidden;
        }

        .user-message::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, transparent 0%, rgba(254, 239, 66, 0.1) 100%);
            pointer-events: none;
        }

        .user-message:hover {
            transform: translateX(-8px) scale(1.02);
        }

        .assistant-message {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
            margin-right: 20%;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
            color: #1e293b;
            border: 1px solid rgba(0, 133, 63, 0.1);
            backdrop-filter: blur(15px);
            position: relative;
            overflow: hidden;
        }

        .assistant-message::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 4px 0 0 4px;
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
            padding: 18px 28px;
            border: 2px solid rgba(0, 133, 63, 0.15);
            border-radius: 50px;
            font-size: 16px;
            color: #1e293b;
            outline: none;
            transition: var(--transition-smooth);
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
            box-shadow: 
                0 4px 16px rgba(0, 0, 0, 0.04),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }

        #messageInput::placeholder {
            color: #64748b;
            font-weight: 400;
        }

        #messageInput:focus {
            border-color: var(--senegal-green);
            box-shadow: 
                0 0 0 4px rgba(0, 133, 63, 0.12),
                0 8px 24px rgba(0, 133, 63, 0.15);
            transform: translateY(-2px);
            background: rgba(255, 255, 255, 1);
        }

        #messageInput:hover:not(:focus) {
            border-color: rgba(0, 133, 63, 0.25);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
        }

        .send-btn {
            background: var(--primary-gradient);
            color: white;
            border: none;
            padding: 18px 36px;
            border-radius: 50px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 700;
            transition: var(--transition-smooth);
            min-width: 140px;
            box-shadow: 
                0 8px 24px rgba(0, 133, 63, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            position: relative;
            overflow: hidden;
        }

        .send-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(254, 239, 66, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s ease, height 0.6s ease;
        }

        .send-btn:hover::before {
            width: 300px;
            height: 300px;
        }

        .send-btn:hover {
            transform: translateY(-4px) scale(1.05);
            box-shadow: 
                0 12px 36px rgba(0, 133, 63, 0.4),
                0 0 0 4px rgba(254, 239, 66, 0.2);
        }

        .send-btn:active {
            transform: translateY(-2px) scale(1.02);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .send-btn:disabled:hover {
            box-shadow: 0 8px 24px rgba(0, 133, 63, 0.3);
        }



        .loading {
            display: none;
            text-align: center;
            color: var(--senegal-green);
            font-style: italic;
            font-weight: 600;
            margin: 24px 0;
            animation: loadingPulse 2s ease-in-out infinite;
            padding: 20px;
            background: rgba(0, 133, 63, 0.05);
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }

        @keyframes loadingPulse {
            0%, 100% { 
                opacity: 0.7; 
                transform: scale(0.98);
            }
            50% { 
                opacity: 1; 
                transform: scale(1.02);
            }
        }

        .typing {
            display: inline-block;
            width: 28px;
            height: 28px;
            border: 4px solid rgba(0, 133, 63, 0.2);
            border-radius: 50%;
            border-top-color: var(--senegal-green);
            border-right-color: var(--senegal-yellow);
            border-bottom-color: var(--senegal-red);
            animation: spin 1.2s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
            margin-right: 12px;
            vertical-align: middle;
            box-shadow: 0 0 20px rgba(0, 133, 63, 0.2);
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

        /* ====== ANIMATIONS FLUIDES ET MICROINTERACTIONS ====== */
        
        /* Effet de brillance sur les boutons */
        .send-btn::after,
        .new-conversation-btn::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                45deg,
                transparent 30%,
                rgba(255, 255, 255, 0.3) 50%,
                transparent 70%
            );
            transform: translateX(-100%);
            transition: transform 0.6s ease;
        }

        .send-btn:hover::after,
        .new-conversation-btn:hover::after {
            transform: translateX(100%);
        }

        /* Particules flottantes dans le header */
        @keyframes float {
            0%, 100% {
                transform: translateY(0) translateX(0);
                opacity: 0.3;
            }
            50% {
                transform: translateY(-20px) translateX(10px);
                opacity: 0.6;
            }
        }

        .chat-header .particle {
            position: absolute;
            width: 6px;
            height: 6px;
            background: var(--senegal-yellow);
            border-radius: 50%;
            animation: float 4s ease-in-out infinite;
            opacity: 0.3;
        }

        /* Effet de vague sur scroll */
        @keyframes wave {
            0%, 100% { transform: translateX(0) translateY(0); }
            25% { transform: translateX(5px) translateY(-5px); }
            75% { transform: translateX(-5px) translateY(5px); }
        }

        .chat-container::-webkit-scrollbar-thumb:active {
            animation: wave 0.5s ease;
        }

        /* Effet de tape sur les inputs */
        #messageInput:focus {
            animation: inputFocus 0.3s ease;
        }

        @keyframes inputFocus {
            0% { transform: scale(1); }
            50% { transform: scale(1.01); }
            100% { transform: scale(1); }
        }

        /* Transition douce entre les pages de conversation */
        .chat-container.transitioning {
            animation: fadeSlide 0.4s ease;
        }

        @keyframes fadeSlide {
            0% {
                opacity: 0.5;
                transform: translateY(10px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Effet de survol premium sur les messages */
        .message::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border-radius: inherit;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }

        .user-message::after {
            background: radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 50%), 
                rgba(254, 239, 66, 0.2) 0%, 
                transparent 60%);
        }

        .assistant-message::after {
            background: radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 50%), 
                rgba(0, 133, 63, 0.1) 0%, 
                transparent 60%);
        }

        .message:hover::after {
            opacity: 1;
        }

        /* Badge "Nouveau" animé */
        @keyframes badgePulse {
            0%, 100% {
                transform: scale(1);
                box-shadow: 0 0 0 0 rgba(227, 27, 35, 0.4);
            }
            50% {
                transform: scale(1.05);
                box-shadow: 0 0 0 6px rgba(227, 27, 35, 0);
            }
        }

        /* Responsive Design Premium */
        @media (max-width: 1024px) {
            .chat-header h1 {
                font-size: 2.4em;
            }
            
            .message {
                margin-left: 10% !important;
                margin-right: 10% !important;
            }
        }

        @media (max-width: 768px) {
            .chat-container {
                padding: 16px;
            }
            
            .chat-header {
                padding: 30px 24px;
            }
            
            .chat-header h1 {
                font-size: 2em;
                letter-spacing: 0.5px;
            }
            
            .chat-header p {
                font-size: 1em;
            }
            
            .chat-input-section {
                padding: 16px;
            }
            
            .input-section {
                flex-direction: row;
                flex-wrap: wrap;
                gap: 10px;
            }
            
            #messageInput {
                width: 100%;
                padding: 16px 24px;
            }
            
            .new-conversation-btn,
            .send-btn {
                flex: 1;
                min-width: 120px;
                padding: 16px 24px;
            }
            
            .message {
                margin-left: 5% !important;
                margin-right: 5% !important;
                padding: 16px 20px;
            }
            
            .theme-toggle,
            .scroll-bottom {
                width: 48px;
                height: 48px;
                font-size: 18px;
            }
            
            .conversations-toggle {
                width: 54px;
                height: 54px;
                font-size: 20px;
                top: 24px;
                left: 24px;
            }
            
            .conversations-panel {
                width: 85%;
            }
        }

        @media (max-width: 480px) {
            .chat-container {
                padding: 12px;
            }
            
            .chat-header {
                padding: 24px 16px;
            }
            
            .chat-header h1 {
                font-size: 1.6em;
            }
            
            .chat-header p {
                font-size: 0.9em;
                margin-top: 10px;
            }
            
            .chat-input-section {
                padding: 12px;
            }
            
            .input-section {
                gap: 8px;
            }
            
            #messageInput {
                padding: 14px 20px;
                font-size: 15px;
            }
            
            .new-conversation-btn,
            .send-btn {
                padding: 14px 20px;
                font-size: 14px;
                min-width: 100px;
            }
            
            .message {
                margin-left: 0 !important;
                margin-right: 0 !important;
                padding: 14px 18px;
                border-radius: 18px;
            }
            
            .message:hover {
                transform: translateX(4px) scale(1.01);
            }
            
            .user-message:hover {
                transform: translateX(-4px) scale(1.01);
            }
            
            .theme-toggle {
                top: 20px;
                right: 20px;
                width: 44px;
                height: 44px;
                font-size: 16px;
            }
            
            .scroll-bottom {
                width: 50px;
                height: 50px;
                font-size: 18px;
                bottom: 120px;
                right: 20px;
            }
            
            .conversations-toggle {
                width: 50px;
                height: 50px;
                font-size: 18px;
                top: 20px;
                left: 20px;
            }
            
            .conversations-panel {
                width: 100%;
            }
            
            .message-actions {
                flex-wrap: wrap;
            }
            
            .message-btn {
                font-size: 12px;
                padding: 6px 12px;
            }
        }

        @media (max-width: 360px) {
            .chat-header h1 {
                font-size: 1.4em;
            }
            
            .input-section {
                flex-direction: column;
            }
            
            .new-conversation-btn,
            .send-btn {
                width: 100%;
            }
        }

        /* ====== THEME TOGGLE + SCROLL TO BOTTOM ====== */
        .theme-toggle {
            position: absolute;
            top: 30px;
            right: 30px;
            width: 52px;
            height: 52px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.15);
            color: var(--senegal-yellow);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            box-shadow: 
                0 6px 20px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
            transition: var(--transition-smooth);
            backdrop-filter: blur(10px);
            font-size: 22px;
            z-index: 100;
        }

        .theme-toggle:hover {
            transform: translateY(-4px) rotate(20deg) scale(1.1);
            background: rgba(255, 255, 255, 0.25);
            box-shadow: 
                0 10px 30px rgba(0, 0, 0, 0.25),
                0 0 0 4px rgba(254, 239, 66, 0.3);
        }

        .scroll-bottom {
            position: fixed;
            right: 30px;
            bottom: 140px;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            border: 2px solid rgba(0, 133, 63, 0.3);
            background: var(--primary-gradient);
            color: #fff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 
                0 8px 28px rgba(0, 133, 63, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            opacity: 0;
            transform: translateY(20px) scale(0.8);
            visibility: hidden;
            transition: var(--transition-bounce);
            z-index: 1001;
            font-size: 20px;
        }

        .scroll-bottom.visible {
            opacity: 1;
            transform: translateY(0) scale(1);
            visibility: visible;
            animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }

        @keyframes bounceIn {
            0% {
                transform: translateY(20px) scale(0.8);
                opacity: 0;
            }
            50% {
                transform: translateY(-5px) scale(1.05);
            }
            100% {
                transform: translateY(0) scale(1);
                opacity: 1;
            }
        }

        .scroll-bottom:hover {
            transform: translateY(-6px) scale(1.15);
            box-shadow: 
                0 12px 40px rgba(0, 133, 63, 0.5),
                0 0 0 4px rgba(254, 239, 66, 0.3);
        }

        /* Mode sombre avec couleurs du drapeau sénégalais */
        body.dark-mode {
            background: linear-gradient(135deg, #0a1f0f 0%, #051509 100%);
        }
        
        body.dark-mode::before {
            background: 
                radial-gradient(circle at 20% 50%, rgba(254, 239, 66, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(227, 27, 35, 0.05) 0%, transparent 50%);
        }
        
        body.dark-mode .chat-app {
            background: rgba(10, 20, 15, 0.95);
            box-shadow: 0 0 100px rgba(0, 133, 63, 0.2);
        }
        
        body.dark-mode .chat-header {
            background: linear-gradient(135deg, #0a2e17 0%, #051509 100%);
            box-shadow: 0 8px 40px rgba(0, 133, 63, 0.4);
        }
        
        body.dark-mode .assistant-message {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
            color: #E5E7EB;
            border-color: rgba(0, 133, 63, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        body.dark-mode .assistant-message::before {
            background: var(--primary-gradient);
        }
        
        body.dark-mode #messageInput {
            background: rgba(17, 24, 39, 0.95);
            color: #E5E7EB;
            border-color: rgba(0, 133, 63, 0.3);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        body.dark-mode #messageInput::placeholder {
            color: #94A3B8;
        }
        
        body.dark-mode #messageInput:focus {
            border-color: var(--senegal-yellow);
            box-shadow: 0 0 0 4px rgba(254, 239, 66, 0.15);
        }
        
        body.dark-mode .chat-input-section {
            background: rgba(10, 20, 15, 0.95);
            border-top-color: rgba(0, 133, 63, 0.3);
        }
        
        body.dark-mode .message-btn {
            background: rgba(30, 41, 59, 0.9);
            border-color: rgba(0, 133, 63, 0.3);
            color: #E5E7EB;
        }
        
        body.dark-mode .conversation-item { 
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
            color: #E5E7EB; 
            border-color: rgba(0, 133, 63, 0.2);
        }
        
        body.dark-mode .conversation-item:hover { 
            background: linear-gradient(135deg, rgba(0, 133, 63, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%);
            border-color: rgba(254, 239, 66, 0.4);
        }
        
        body.dark-mode .conversation-item.active {
            background: linear-gradient(135deg, rgba(0, 133, 63, 0.25) 0%, rgba(254, 239, 66, 0.15) 100%);
            border-color: var(--senegal-yellow);
        }
        
        body.dark-mode .conversations-panel { 
            background: linear-gradient(135deg, rgba(10, 20, 15, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
            border-right-color: var(--senegal-green);
        }
        
        body.dark-mode .conversations-header { 
            background: var(--secondary-gradient);
            border-bottom-color: rgba(0, 133, 63, 0.4);
        }
        
        body.dark-mode .loading {
            background: rgba(0, 133, 63, 0.1);
            color: var(--senegal-yellow);
        }
        
        body.dark-mode .modal-content {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #E5E7EB;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
        }
        
        body.dark-mode .modal-textarea {
            background: rgba(15, 23, 42, 0.95);
            color: #E5E7EB;
            border-color: rgba(0, 133, 63, 0.3);
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
            gap: 6px;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid rgba(0, 133, 63, 0.15);
            border-radius: 12px;
            font-size: 13px;
            color: #334155;
            cursor: pointer;
            transition: var(--transition-smooth);
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
        }

        .message-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.4s ease, height 0.4s ease;
        }

        .message-btn:hover::before {
            width: 200px;
            height: 200px;
        }

        .message-btn:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 6px 20px rgba(0, 133, 63, 0.15);
        }

        .message-btn.edit::before {
            background: rgba(251, 146, 60, 0.1);
        }

        .message-btn.edit:hover {
            border-color: rgba(251, 146, 60, 0.4);
            color: #f59e0b;
            box-shadow: 0 6px 20px rgba(251, 146, 60, 0.2);
        }

        .message-btn.regenerate::before {
            background: rgba(0, 133, 63, 0.1);
        }

        .message-btn.regenerate:hover {
            border-color: var(--senegal-green);
            color: var(--senegal-green);
            box-shadow: 0 6px 20px rgba(0, 133, 63, 0.2);
        }

        .message-btn.copy::before {
            background: rgba(254, 239, 66, 0.2);
        }

        .message-btn.copy:hover {
            border-color: var(--senegal-yellow);
            color: #a16207;
            box-shadow: 0 6px 20px rgba(254, 239, 66, 0.3);
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
            position: fixed !important;
            top: 30px;
            left: 30px;
            width: 65px;
            height: 65px;
            border-radius: 50%;
            background: var(--primary-gradient);
            border: 3px solid rgba(254, 239, 66, 0.6);
            color: white;
            font-size: 26px;
            cursor: pointer;
            box-shadow: 
                0 10px 35px rgba(0, 133, 63, 0.5),
                0 0 0 0 rgba(254, 239, 66, 0.7),
                inset 0 2px 0 rgba(255, 255, 255, 0.3);
            z-index: 10000 !important;
            transition: var(--transition-bounce);
            display: flex !important;
            align-items: center;
            justify-content: center;
            overflow: visible;
            animation: conversationPulse 3s ease-in-out infinite;
        }

        @keyframes conversationPulse {
            0%, 100% {
                box-shadow: 
                    0 10px 35px rgba(0, 133, 63, 0.5),
                    0 0 0 0 rgba(254, 239, 66, 0.7),
                    inset 0 2px 0 rgba(255, 255, 255, 0.3);
            }
            50% {
                box-shadow: 
                    0 10px 35px rgba(0, 133, 63, 0.5),
                    0 0 0 8px rgba(254, 239, 66, 0.3),
                    inset 0 2px 0 rgba(255, 255, 255, 0.3);
            }
        }

        .conversations-toggle::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle, rgba(254, 239, 66, 0.3) 0%, transparent 70%);
            animation: pulseRing 2s ease-in-out infinite;
        }

        @keyframes pulseRing {
            0%, 100% {
                transform: scale(0.8);
                opacity: 0;
            }
            50% {
                transform: scale(1.2);
                opacity: 1;
            }
        }

        .conversations-toggle:hover {
            transform: scale(1.2) rotate(15deg);
            box-shadow: 
                0 15px 50px rgba(0, 133, 63, 0.6),
                0 0 0 10px rgba(254, 239, 66, 0.4),
                inset 0 2px 0 rgba(255, 255, 255, 0.4);
            animation: none;
        }

        .conversations-toggle i {
            position: relative;
            z-index: 10001 !important;
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
            display: inline-block;
            color: white;
            font-size: 26px;
        }

        /* S'assurer que le bouton est toujours visible */
        #conversationsToggle {
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }

        .new-conversation-btn {
            background: linear-gradient(135deg, var(--senegal-yellow) 0%, #e6d000 100%);
            border: none;
            color: #1e293b;
            padding: 18px 28px;
            border-radius: 50px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 700;
            transition: var(--transition-smooth);
            margin-right: 12px;
            box-shadow: 
                0 6px 20px rgba(254, 239, 66, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            position: relative;
            overflow: hidden;
        }

        .new-conversation-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(227, 27, 35, 0.2);
            transform: translate(-50%, -50%);
            transition: width 0.5s ease, height 0.5s ease;
        }

        .new-conversation-btn:hover::before {
            width: 300px;
            height: 300px;
        }

        .new-conversation-btn:hover {
            transform: translateY(-4px) scale(1.05);
            box-shadow: 
                0 10px 30px rgba(254, 239, 66, 0.4),
                0 0 0 4px rgba(0, 133, 63, 0.15);
        }

        .new-conversation-btn:active {
            transform: translateY(-2px) scale(1.02);
        }

        .conversations-panel {
            position: fixed;
            top: 0;
            left: 0;
            width: 400px;
            height: 100vh;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%);
            box-shadow: 
                4px 0 40px rgba(0, 133, 63, 0.2),
                inset -1px 0 0 rgba(0, 133, 63, 0.15);
            z-index: 1001;
            display: none;
            flex-direction: column;
            border-right: 4px solid var(--senegal-green);
            backdrop-filter: blur(20px);
            transform: translateX(-100%);
            transition: transform 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }

        .conversations-panel.show {
            display: flex;
            transform: translateX(0);
        }

        @keyframes slideInLeft {
            from {
                transform: translateX(-100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        .conversations-header {
            padding: 28px 24px;
            border-bottom: 2px solid rgba(0, 133, 63, 0.15);
            background: var(--primary-gradient);
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0, 133, 63, 0.2);
        }

        .conversations-header h3 {
            color: white;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
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
            padding: 10px 18px;
            border: 2px solid transparent;
            border-radius: 12px;
            font-size: 13px;
            cursor: pointer;
            transition: var(--transition-smooth);
            font-weight: 700;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .conv-btn.primary {
            background: linear-gradient(135deg, var(--senegal-yellow) 0%, #e6d000 100%);
            color: #1e293b;
        }

        .conv-btn.primary:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 
                0 6px 20px rgba(254, 239, 66, 0.4),
                0 0 0 3px rgba(0, 133, 63, 0.2);
        }

        .conv-btn.secondary {
            background: rgba(255, 255, 255, 0.9);
            color: #1e293b;
            border-color: rgba(0, 133, 63, 0.2);
        }

        .conv-btn.secondary:hover {
            background: rgba(255, 255, 255, 1);
            border-color: var(--senegal-green);
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0, 133, 63, 0.2);
        }

        .conversations-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }

        .conversation-item {
            padding: 16px 20px;
            margin-bottom: 12px;
            border-radius: 16px;
            cursor: pointer;
            transition: var(--transition-smooth);
            border: 2px solid transparent;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(249, 250, 251, 0.9) 100%);
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
        }

        .conversation-item::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            width: 4px;
            height: 100%;
            background: var(--primary-gradient);
            transform: scaleY(0);
            transition: transform 0.3s ease;
        }

        .conversation-item:hover::before {
            transform: scaleY(1);
        }

        .conversation-item:hover {
            background: linear-gradient(135deg, rgba(254, 239, 66, 0.1) 0%, rgba(255, 255, 255, 0.95) 100%);
            border-color: rgba(0, 133, 63, 0.3);
            transform: translateX(8px) scale(1.02);
            box-shadow: 0 6px 24px rgba(0, 133, 63, 0.15);
        }

        .conversation-item.active {
            background: linear-gradient(135deg, rgba(0, 133, 63, 0.15) 0%, rgba(254, 239, 66, 0.15) 100%);
            border-color: var(--senegal-green);
            box-shadow: 
                0 6px 24px rgba(0, 133, 63, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.5);
        }

        .conversation-item.active::before {
            transform: scaleY(1);
            width: 6px;
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
                <h1>🇸🇳 SRMT-DOCUMIND - MODE RAG STRICT</h1>
                <p>Assistant IA Spécialisé sur Documents Fiscaux • Réponses Exclusives sur Base Documentaire Fiscale</p>
                <button id="themeToggle" class="theme-toggle" title="Changer de thème">
                    <i class="fa-solid fa-moon"></i>
                </button>
            </div>

        <div class="chat-container" id="chatContainer">
            <div class="message assistant-message">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <span style="font-size: 48px; filter: drop-shadow(0 2px 8px rgba(0, 133, 63, 0.3));">🇸🇳</span>
                    <div>
                        <div style="font-size: 1.3em; font-weight: 700; color: var(--senegal-green); margin-bottom: 4px;">
                            Bienvenue sur SRMT-DOCUMIND
                        </div>
                        <div style="font-size: 0.95em; color: #64748b; font-weight: 500;">
                            Assistant IA Expert en Fiscalité & Douanes du Sénégal
                        </div>
                    </div>
                </div>
                
                <div style="background: linear-gradient(135deg, rgba(0, 133, 63, 0.08) 0%, rgba(254, 239, 66, 0.08) 100%); 
                            padding: 16px; border-radius: 12px; margin-bottom: 16px; 
                            border-left: 4px solid var(--senegal-green);">
                    <div style="font-weight: 700; color: var(--senegal-green); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.1em;">⚡</span> Mode RAG Strict Activé
                    </div>
                    <div style="font-size: 0.9em; color: #475569; line-height: 1.6;">
                        Réponses exclusivement basées sur les documents fiscaux officiels indexés
                    </div>
                </div>
                
                <div style="margin-bottom: 16px;">
                    <div style="font-weight: 700; color: #1e293b; margin-bottom: 12px; font-size: 1.05em;">
                        📚 Domaines d'Expertise
                    </div>
                    <div style="display: grid; gap: 8px; margin-left: 8px;">
                        <div style="display: flex; align-items: start; gap: 8px;">
                            <span style="color: var(--senegal-green); font-weight: 700;">✓</span>
                            <span style="line-height: 1.5;">Code Général des Impôts (CGI) du Sénégal</span>
                        </div>
                        <div style="display: flex; align-items: start; gap: 8px;">
                            <span style="color: var(--senegal-green); font-weight: 700;">✓</span>
                            <span style="line-height: 1.5;">Code des Douanes de la République du Sénégal</span>
                        </div>
                        <div style="display: flex; align-items: start; gap: 8px;">
                            <span style="color: var(--senegal-green); font-weight: 700;">✓</span>
                            <span style="line-height: 1.5;">Textes fiscaux et réglementations douanières officielles</span>
                        </div>
                    </div>
                </div>
                
                <div style="background: linear-gradient(135deg, rgba(254, 239, 66, 0.1) 0%, rgba(255, 255, 255, 0.5) 100%); 
                            padding: 14px; border-radius: 12px; border: 2px solid rgba(254, 239, 66, 0.3);">
                    <div style="font-weight: 700; color: #854d0e; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.1em;">💡</span> Exemples de Questions
                    </div>
                    <div style="display: grid; gap: 6px; font-size: 0.9em; color: #475569; margin-left: 8px;">
                        <div style="line-height: 1.5;">• "Quels sont les taux de TVA selon le Code Général des Impôts ?"</div>
                        <div style="line-height: 1.5;">• "Quelle est la base imposable de l'impôt minimum forfaitaire ?"</div>
                        <div style="line-height: 1.5;">• "Comment fonctionne le régime de l'entrepôt de stockage selon le Code des Douanes 2014 ?"</div>
                        <div style="line-height: 1.5;">• "Quelles sont les conditions d'exonération de droits de douane ?"</div>
                        <div style="line-height: 1.5;">• "Que dit le Code des Impôts sur les plus-values de cession ?"</div>
                        <div style="line-height: 1.5;">• "Expliquez la procédure de dédouanement des marchandises"</div>
                    </div>
                </div>
                
                <div style="margin-top: 16px; padding-top: 16px; border-top: 2px solid rgba(0, 133, 63, 0.1); 
                            text-align: center; color: var(--senegal-green); font-weight: 600; font-size: 1.05em;">
                    🚀 Prêt à répondre à vos questions fiscales et douanières !
                </div>
            </div>
        </div>

            <div class="loading" id="loading">
                <div class="typing"></div>
                <span>SRMT-DOCUMIND analyse votre question fiscal/douanière<span class="loading-dots"></span></span>
            </div>

            <div class="chat-input-section">
                <div class="input-section">
                    <input type="text" id="messageInput" placeholder="Posez votre question sur le Code des Impôts ou Code des Douanes uniquement..." onkeypress="checkEnter(event)">
                    <button class="new-conversation-btn" onclick="startNewConversation()" title="Nouvelle conversation">
                         Nouveau
                    </button>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                         Envoyer
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Bouton flottant pour ouvrir le panneau des conversations (POSITION FIXE) -->
    <button id="conversationsToggle" class="conversations-toggle" onclick="toggleConversationsPanel()" title="💬 Historique des Conversations" style="position: fixed !important; top: 30px !important; left: 30px !important; z-index: 10000 !important; display: flex !important;">
        <i class="fa-solid fa-comments" style="color: white; font-size: 26px;"></i>
    </button>

    <!-- Bouton scroll vers le bas -->
    <button id="scrollBottom" class="scroll-bottom" title="Aller en bas">
        <i class="fa-solid fa-arrow-down"></i>
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
                     Nouveau
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

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;

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
            // Mettre à jour le bouton de scroll
            try { chatContainer.dispatchEvent(new Event('scroll')); } catch (e) {}

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
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
                // Mettre à jour le bouton de scroll après ajout de la réponse
                setTimeout(() => { try { chatContainer.dispatchEvent(new Event('scroll')); } catch (e) {} }, 50);

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
            // Mettre à jour le bouton de scroll
            try { chatContainer.dispatchEvent(new Event('scroll')); } catch (e) {}
            
            // Ajouter les boutons aux nouveaux messages
            setTimeout(() => {
                addChatButtons();
                // Sauvegarder automatiquement la conversation
                saveCurrentConversation();
            }, 300);
        }

        function clearChat() {
            const chatContainer = document.getElementById('chatContainer');
            
            // Animation élaborée de sortie
            chatContainer.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 1, 1)';
            chatContainer.style.opacity = '0';
            chatContainer.style.transform = 'scale(0.95) translateY(20px)';
            
            setTimeout(() => {
                chatContainer.innerHTML = `
                    <div class="message assistant-message" style="opacity: 0; transform: translateY(20px);">
                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                            <span style="font-size: 48px; filter: drop-shadow(0 2px 8px rgba(0, 133, 63, 0.3));">🇸🇳</span>
                            <div>
                                <div style="font-size: 1.3em; font-weight: 700; color: var(--senegal-green); margin-bottom: 4px;">
                                    Bienvenue sur SRMT-DOCUMIND
                                </div>
                                <div style="font-size: 0.95em; color: #64748b; font-weight: 500;">
                                    Assistant IA Expert en Fiscalité & Douanes du Sénégal
                                </div>
                            </div>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, rgba(0, 133, 63, 0.08) 0%, rgba(254, 239, 66, 0.08) 100%); 
                                    padding: 16px; border-radius: 12px; margin-bottom: 16px; 
                                    border-left: 4px solid var(--senegal-green);">
                            <div style="font-weight: 700; color: var(--senegal-green); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.1em;">⚡</span> Mode RAG Strict Activé
                            </div>
                            <div style="font-size: 0.9em; color: #475569; line-height: 1.6;">
                                Réponses exclusivement basées sur les documents fiscaux officiels indexés
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 16px;">
                            <div style="font-weight: 700; color: #1e293b; margin-bottom: 12px; font-size: 1.05em;">
                                📚 Domaines d'Expertise
                            </div>
                            <div style="display: grid; gap: 8px; margin-left: 8px;">
                                <div style="display: flex; align-items: start; gap: 8px;">
                                    <span style="color: var(--senegal-green); font-weight: 700;">✓</span>
                                    <span style="line-height: 1.5;">Code Général des Impôts (CGI) du Sénégal</span>
                                </div>
                                <div style="display: flex; align-items: start; gap: 8px;">
                                    <span style="color: var(--senegal-green); font-weight: 700;">✓</span>
                                    <span style="line-height: 1.5;">Code des Douanes de la République du Sénégal</span>
                                </div>
                                <div style="display: flex; align-items: start; gap: 8px;">
                                    <span style="color: var(--senegal-green); font-weight: 700;">✓</span>
                                    <span style="line-height: 1.5;">Textes fiscaux et réglementations douanières officielles</span>
                                </div>
                            </div>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, rgba(254, 239, 66, 0.1) 0%, rgba(255, 255, 255, 0.5) 100%); 
                                    padding: 14px; border-radius: 12px; border: 2px solid rgba(254, 239, 66, 0.3);">
                            <div style="font-weight: 700; color: #854d0e; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.1em;">💡</span> Exemples de Questions
                            </div>
                            <div style="display: grid; gap: 6px; font-size: 0.9em; color: #475569; margin-left: 8px;">
                                <div style="line-height: 1.5;">• "Quels sont les taux de TVA selon le Code Général des Impôts ?"</div>
                                <div style="line-height: 1.5;">• "Quelle est la base imposable de l'impôt minimum forfaitaire ?"</div>
                                <div style="line-height: 1.5;">• "Comment fonctionne le régime de l'entrepôt de stockage selon le Code des Douanes 2014 ?"</div>
                                <div style="line-height: 1.5;">• "Quelles sont les conditions d'exonération de droits de douane ?"</div>
                                <div style="line-height: 1.5;">• "Que dit le Code des Impôts sur les plus-values de cession ?"</div>
                                <div style="line-height: 1.5;">• "Expliquez la procédure de dédouanement des marchandises"</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 16px; padding-top: 16px; border-top: 2px solid rgba(0, 133, 63, 0.1); 
                                    text-align: center; color: var(--senegal-green); font-weight: 600; font-size: 1.05em;">
                            🚀 Prêt à répondre à vos questions fiscales et douanières !
                        </div>
                    </div>
                `;
                
                // Animation élaborée d'entrée avec rebond
                setTimeout(() => {
                    chatContainer.style.transition = 'all 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
                    chatContainer.style.opacity = '1';
                    chatContainer.style.transform = 'scale(1) translateY(0)';
                    
                    const welcomeMsg = chatContainer.querySelector('.message');
                    if (welcomeMsg) {
                        welcomeMsg.style.transition = 'all 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
                        welcomeMsg.style.opacity = '1';
                        welcomeMsg.style.transform = 'translateY(0)';
                    }
                    
                    addMessageEffects();
                    try { chatContainer.dispatchEvent(new Event('scroll')); } catch (e) {}
                }, 50);
            }, 400);
        }

        // Effets au survol des messages avec tracking de souris
        function addMessageEffects() {
            const messages = document.querySelectorAll('.message');
            messages.forEach(message => {
                // Effet de survol avec transformation fluide
                message.addEventListener('mouseenter', function(e) {
                    const isUser = this.classList.contains('user-message');
                    if (isUser) {
                        this.style.transform = 'translateX(-8px) scale(1.02)';
                    } else {
                        this.style.transform = 'translateX(8px) scale(1.02)';
                    }
                });
                
                message.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateX(0) scale(1)';
                });

                // Tracking de position de souris pour effet radial
                message.addEventListener('mousemove', function(e) {
                    const rect = this.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * 100;
                    const y = ((e.clientY - rect.top) / rect.height) * 100;
                    this.style.setProperty('--mouse-x', x + '%');
                    this.style.setProperty('--mouse-y', y + '%');
                });
            });
        }

        // Ajouter des particules décoratives au header
        function addHeaderParticles() {
            const header = document.querySelector('.chat-header');
            if (!header || header.querySelector('.particle')) return;

            for (let i = 0; i < 8; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 4 + 's';
                particle.style.animationDuration = (3 + Math.random() * 3) + 's';
                header.appendChild(particle);
            }
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
            
            referencesDiv.innerHTML = '<strong> Références précises :</strong>';
            
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
                    <div style="font-weight: 600; color: #667eea;"> ${ref.file_name}</div>
                    <div style="color: #666; margin: 4px 0;">
                         ${ref.page_info} • ${ref.location}
                    </div>
                    <div style="color: #888; font-size: 12px; font-style: italic;">
                        "${ref.snippet}"
                    </div>
                    <div style="margin-top: 5px;">
                        <button onclick="openFile('${ref.file_path}', ${ref.line_start})" 
                                style="background: #667eea; color: white; border: none; padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">
                             Ouvrir à cette position
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
            // Mise à jour scroll après ajout des références
            const container = document.getElementById('chatContainer');
            try { container.dispatchEvent(new Event('scroll')); } catch (e) {}
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
                    btn.textContent = ' Ouvert!';
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

        // Initialisation au chargement avec animations élégantes
        window.onload = function() {
            addMessageEffects();
            addHeaderParticles();
            
            // Focus automatique sur l'input avec animation
            const messageInput = document.getElementById('messageInput');
            setTimeout(() => {
                messageInput.focus();
                messageInput.style.animation = 'inputFocus 0.5s ease';
            }, 800);
            
            // Animation d'entrée en cascade
            const container = document.querySelector('.container');
            const chatHeader = document.querySelector('.chat-header');
            const chatContainer = document.getElementById('chatContainer');
            const inputSection = document.querySelector('.chat-input-section');
            
            // États initiaux
            container.style.opacity = '0';
            chatHeader.style.opacity = '0';
            chatHeader.style.transform = 'translateY(-30px)';
            chatContainer.style.opacity = '0';
            chatContainer.style.transform = 'translateY(20px)';
            inputSection.style.opacity = '0';
            inputSection.style.transform = 'translateY(30px)';
            
            // Animations en cascade
            setTimeout(() => {
                container.style.transition = 'opacity 0.6s ease';
                container.style.opacity = '1';
                
                setTimeout(() => {
                    chatHeader.style.transition = 'all 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
                    chatHeader.style.opacity = '1';
                    chatHeader.style.transform = 'translateY(0)';
                }, 100);
                
                setTimeout(() => {
                    chatContainer.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                    chatContainer.style.opacity = '1';
                    chatContainer.style.transform = 'translateY(0)';
                }, 300);
                
                setTimeout(() => {
                    inputSection.style.transition = 'all 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
                    inputSection.style.opacity = '1';
                    inputSection.style.transform = 'translateY(0)';
                }, 500);
            }, 100);
        };

        // ====== THEME (clair/sombre) ======
        function updateThemeIcon() {
            const toggle = document.getElementById('themeToggle');
            if (!toggle) return;
            const isDark = document.body.classList.contains('dark-mode');
            toggle.innerHTML = isDark
                ? '<i class="fa-solid fa-sun"></i>'
                : '<i class="fa-solid fa-moon"></i>';
        }

        function toggleTheme() {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            localStorage.setItem('srmt_theme', isDark ? 'dark' : 'light');
            updateThemeIcon();
        }

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
                        <button class="message-btn regenerate" onclick="regenerateMessage(this)"> Régénérer</button>
                        <button class="message-btn copy" onclick="copyMessage(this)"> Copier</button>
                    `;
                }
                
                message.appendChild(actionsDiv);
            });
        }

        // Modifier un message
        function editMessage(btn) {
            console.log(' Bouton modifier cliqué');
            
            const message = btn.closest('.message');
            if (!message) {
                console.error(' Message parent non trouvé');
                alert('Erreur: Message non trouvé');
                return;
            }
            
            console.log(' Message trouvé:', message);
            
            // Extraire le texte (plusieurs méthodes)
            let text = '';
            
            // Méthode 1: Premier nœud texte
            if (message.childNodes[0] && message.childNodes[0].textContent) {
                text = message.childNodes[0].textContent.trim();
                console.log(' Texte méthode 1:', text);
            }
            
            // Méthode 2: Span avec effet de frappe
            if (!text && message.querySelector('span')) {
                text = message.querySelector('span').textContent.trim();
                console.log(' Texte méthode 2 (span):', text);
            }
            
            // Méthode 3: Tout le texte moins les boutons
            if (!text) {
                const clone = message.cloneNode(true);
                const actionsDiv = clone.querySelector('.message-actions');
                if (actionsDiv) actionsDiv.remove();
                text = clone.textContent.trim();
                console.log(' Texte méthode 3 (clone):', text);
            }
            
            if (!text) {
                console.error(' Aucun texte extrait');
                alert('Erreur: Impossible d\\'extraire le texte du message');
                return;
            }
            
            console.log(' Texte final à éditer:', text);
            
            // Ouvrir le modal
            const modal = document.getElementById('editModal');
            const textarea = document.getElementById('editTextarea');
            
            if (!modal || !textarea) {
                console.error(' Éléments du modal non trouvés', {modal, textarea});
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
            
            console.log(' Modal ouvert avec succès');
        }

        // Fermer le modal
        function closeEditModal() {
            console.log(' Fermeture du modal d\\'édition');
            const modal = document.getElementById('editModal');
            if (modal) {
                modal.classList.remove('show');
                console.log(' Modal fermé');
            } else {
                console.error(' Modal non trouvé pour fermeture');
            }
        }

        // Sauvegarder le message modifié
        function saveEditMessage() {
            console.log(' Sauvegarde du message modifié');
            
            const modal = document.getElementById('editModal');
            const textarea = document.getElementById('editTextarea');
            
            if (!modal || !textarea) {
                console.error(' Éléments manquants', {modal, textarea});
                alert('Erreur: Éléments du modal manquants');
                return;
            }
            
            const newText = textarea.value.trim();
            console.log(' Nouveau texte:', newText);
            
            if (!newText) {
                console.log(' Texte vide, abandon');
                alert('Veuillez saisir un message');
                return;
            }
            
            const messageElement = modal.messageElement;
            if (!messageElement) {
                console.error(' Message element manquant');
                alert('Erreur: Message à modifier non trouvé');
                return;
            }
            
            console.log(' Suppression des messages à partir de:', messageElement);
            
            closeEditModal();
            
            // Supprimer les messages à partir de celui modifié
            const chatContainer = document.getElementById('chatContainer');
            const messages = Array.from(chatContainer.children);
            const messageIndex = messages.indexOf(messageElement);
            
            console.log(` Index du message: ${messageIndex}, Total messages: ${messages.length}`);
            
            for (let i = messages.length - 1; i >= messageIndex; i--) {
                if (messages[i] && messages[i].classList && messages[i].classList.contains('message')) {
                    console.log(` Suppression message ${i}`);
                    chatContainer.removeChild(messages[i]);
                }
            }
            
            // Renvoyer le nouveau message
            console.log(' Envoi du nouveau message:', newText);
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
                const isOpen = panel.classList.contains('show');
                if (isOpen) {
                    panel.classList.remove('show');
                    setTimeout(() => {
                        panel.style.display = 'none';
                    }, 400);
                } else {
                    panel.style.display = 'flex';
                    setTimeout(() => {
                        panel.classList.add('show');
                        updateConversationsUI();
                    }, 10);
                }
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

            // Appliquer le thème sauvegardé
            const savedTheme = localStorage.getItem('srmt_theme') || 'light';
            if (savedTheme === 'dark') {
                document.body.classList.add('dark-mode');
            }
            const themeToggle = document.getElementById('themeToggle');
            if (themeToggle) {
                updateThemeIcon();
                themeToggle.addEventListener('click', toggleTheme);
            }

            // Gestion du bouton "aller en bas"
            const chatContainer = document.getElementById('chatContainer');
            const scrollBtn = document.getElementById('scrollBottom');
            if (chatContainer && scrollBtn) {
                const updateScrollBtn = () => {
                    const delta = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight;
                    const nearBottom = delta < 80;
                    if (nearBottom) {
                        scrollBtn.classList.remove('visible');
                    } else {
                        scrollBtn.classList.add('visible');
                    }
                };
                chatContainer.addEventListener('scroll', updateScrollBtn);
                scrollBtn.addEventListener('click', () => {
                    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
                });
                // Initial state
                setTimeout(updateScrollBtn, 300);
            }
        });




    </script>
    
    <!-- Footer SRMT-DOCUMIND avec drapeau animé -->
    <div class="srmt-footer" style="position: fixed; bottom: 20px; right: 25px; 
                color: white; font-size: 13px; font-weight: 600;
                background: linear-gradient(135deg, var(--senegal-green) 0%, #006838 100%);
                padding: 12px 24px; 
                border-radius: 30px; 
                backdrop-filter: blur(15px);
                border: 2px solid rgba(254, 239, 66, 0.3);
                box-shadow: 0 8px 28px rgba(0, 133, 63, 0.4);
                z-index: 999;
                transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                display: flex;
                align-items: center;
                gap: 10px;">
        <span style="font-size: 20px; animation: wave 2s ease-in-out infinite;">🇸🇳</span>
        <span style="text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);">
            Powered by <strong style="color: var(--senegal-yellow); text-shadow: 0 0 10px rgba(254, 239, 66, 0.5);">ACCEL-TECH</strong>
        </span>
    </div>
    
    <style>
        .srmt-footer:hover {
            transform: translateY(-4px) scale(1.05);
            box-shadow: 0 12px 40px rgba(0, 133, 63, 0.6);
            border-color: var(--senegal-yellow);
        }
        
        @keyframes wave {
            0%, 100% { transform: rotate(-5deg); }
            50% { transform: rotate(5deg); }
        }
        
        @media (max-width: 768px) {
            .srmt-footer {
                bottom: 10px;
                right: 10px;
                font-size: 11px;
                padding: 8px 16px;
            }
        }
    </style>
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
        
        if not message:
            return jsonify({
                'response': 'Veuillez saisir un message.',
                'references': []
            }), 400
        
        result = srmt_client.chat(message)
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
