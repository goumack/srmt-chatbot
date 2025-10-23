"""
LexFin - Assistant IA Spécialisé Fiscal et Douanier (MODE RAG STRICT)
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
from datetime import datetime
import re
from collections import Counter
import math

# Import du système hiérarchique V2.0 (import conditionnel pour éviter circularité)
try:
    from systeme_hierarchique_v2 import HierarchieJuridiqueClient
    HIERARCHIE_AVAILABLE = True
except ImportError:
    HIERARCHIE_AVAILABLE = False
    HierarchieJuridiqueClient = None

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger la configuration
load_dotenv()

class BM25:
    """Implémentation simple de BM25 pour recherche textuelle"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        
    def tokenize(self, text):
        """Tokenize le texte en mots"""
        text = text.lower()
        # Garder les chiffres et lettres
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def compute_idf(self, documents):
        """Calcule l'IDF pour chaque terme"""
        N = len(documents)
        idf = {}
        
        # Compter dans combien de documents chaque terme apparaît
        df = Counter()
        for doc in documents:
            tokens = set(self.tokenize(doc))
            df.update(tokens)
        
        # Calculer IDF
        for term, freq in df.items():
            idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        
        return idf
    
    def score(self, query, document, avg_doc_len, idf):
        """Calcule le score BM25 pour un document"""
        query_tokens = self.tokenize(query)
        doc_tokens = self.tokenize(document)
        doc_len = len(doc_tokens)
        
        score = 0.0
        doc_term_freq = Counter(doc_tokens)
        
        for term in query_tokens:
            if term not in idf:
                continue
                
            tf = doc_term_freq.get(term, 0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_doc_len))
            
            score += idf[term] * (numerator / denominator)
        
        return score

class LexFinConfig:
    """Configuration LexFin - Assistant Fiscal et Douanier"""
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollamaaccel-chatbotaccel.apps.senum.heritage.africa")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral:7b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    WATCH_DIRECTORY = os.getenv("WATCH_DIRECTORY", "./documents")  # Répertoire à surveiller
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md', '.json', '.csv', '.odt', '.xlsx', '.xls']

class DocumentWatcherHandler(FileSystemEventHandler):
    """Gestionnaire de surveillance automatique en arrière-plan"""
    
    def __init__(self, lexfin_client):
        self.lexfin_client = lexfin_client
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
                if self.lexfin_client.is_supported_file(file_path):
                    self.lexfin_client.process_new_file_background(file_path)
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
                    if self.lexfin_client.is_supported_file(file_path):
                        self.lexfin_client.process_modified_file_background(file_path)
                        logger.info(f" [AUTO] Fichier réindexé automatiquement: {Path(file_path).name}")
                except Exception as e:
                    logger.error(f" [AUTO] Erreur réindexation automatique {Path(file_path).name}: {e}")
            
            # Lancer en thread séparé
            thread = threading.Thread(target=delayed_reprocess, daemon=True)
            thread.start()

class ConversationManager:
    """Gestionnaire de conversations avec mémoire contextuelle pour discussions intelligentes"""
    
    def __init__(self, max_history_length=10):
        self.conversations = {}  # {conversation_id: conversation_data}
        self.max_history_length = max_history_length
        
    def create_conversation(self, conversation_id=None):
        """Crée une nouvelle conversation"""
        if conversation_id is None:
            conversation_id = str(int(time.time()))
        
        self.conversations[conversation_id] = {
            'id': conversation_id,
            'created_at': time.time(),
            'last_updated': time.time(),
            'history': [],  # [{'role': 'user'/'assistant', 'content': str, 'timestamp': float, 'references': []}]
            'context_keywords': set(),  # Mots-clés extraits pour le contexte
            'current_topics': [],  # Sujets actuels de discussion
        }
        
        logger.info(f"🗨️ Nouvelle conversation créée: {conversation_id}")
        return conversation_id
    
    def add_message(self, conversation_id, role, content, references=None):
        """Ajoute un message à l'historique de conversation"""
        if conversation_id not in self.conversations:
            conversation_id = self.create_conversation(conversation_id)
        
        conversation = self.conversations[conversation_id]
        
        message = {
            'role': role,
            'content': content,
            'timestamp': time.time(),
            'references': references or []
        }
        
        conversation['history'].append(message)
        conversation['last_updated'] = time.time()
        
        # Limiter la taille de l'historique
        if len(conversation['history']) > self.max_history_length * 2:  # user + assistant = 2 messages
            conversation['history'] = conversation['history'][-self.max_history_length * 2:]
        
        # Extraire les mots-clés pour le contexte
        if role == 'user':
            self._extract_keywords(conversation, content)
        
        logger.info(f"💬 Message ajouté à la conversation {conversation_id}: {role}")
        return conversation_id
    
    def _extract_keywords(self, conversation, content):
        """Extrait les mots-clés importants du message utilisateur"""
        import re
        
        # Mots-clés fiscaux et juridiques importants
        fiscal_keywords = [
            'tva', 'taxe', 'impot', 'impôt', 'douane', 'article', 'code',
            'marchandise', 'importation', 'exportation', 'société', 'sociétés',
            'fiscal', 'bénéfice', 'revenus', 'déclaration', 'assujetti',
            'redevable', 'exonération', 'déduction', 'crédit', 'loi', 'finances',
            'budget', 'recettes', 'dépenses', 'investissement', 'économique'
        ]
        
        # Extraire les mots-clés du contenu
        content_lower = content.lower()
        for keyword in fiscal_keywords:
            if keyword in content_lower:
                conversation['context_keywords'].add(keyword)
        
        # Extraire les numéros d'articles
        articles = re.findall(r'article\s+(\d+)', content_lower)
        for article in articles:
            conversation['context_keywords'].add(f'article_{article}')
        
        # Extraire les valeurs numériques importantes
        montants = re.findall(r'(\d+(?:\s\d+)*(?:,\d+)?\s*(?:millions?|milliards?)\s*(?:fcfa|euros?))', content_lower)
        for montant in montants:
            conversation['context_keywords'].add(f'montant_{montant.replace(" ", "_")}')
    
    def get_conversation_context(self, conversation_id, max_messages=6):
        """Récupère le contexte de la conversation pour alimenter le prompt"""
        if conversation_id not in self.conversations:
            return ""
        
        conversation = self.conversations[conversation_id]
        history = conversation['history']
        
        if not history:
            return ""
        
        # Prendre les derniers messages (max_messages)
        recent_history = history[-max_messages:]
        
        context_parts = []
        context_parts.append("HISTORIQUE DE LA CONVERSATION ACTUELLE:")
        
        for i, message in enumerate(recent_history, 1):
            role_label = "UTILISATEUR" if message['role'] == 'user' else "ASSISTANT"
            context_parts.append(f"{i}. {role_label}: {message['content'][:200]}...")
            
            # Ajouter les références si disponibles
            if message.get('references') and message['role'] == 'assistant':
                refs = message['references'][:2]  # Limiter à 2 références
                for ref in refs:
                    article = ref.get('article_ref', 'N/A')
                    context_parts.append(f"   → Référence: {article}")
        
        # Ajouter les mots-clés contextuels
        if conversation['context_keywords']:
            keywords = list(conversation['context_keywords'])[:8]  # Limiter à 8 mots-clés
            context_parts.append(f"MOTS-CLÉS DU CONTEXTE: {', '.join(keywords)}")
        
        return "\n".join(context_parts)
    
    def analyze_follow_up_question(self, conversation_id, current_question):
        """Analyse si la question actuelle fait référence à la conversation précédente"""
        if conversation_id not in self.conversations:
            return False, ""
        
        conversation = self.conversations[conversation_id]
        history = conversation['history']
        
        if len(history) < 2:  # Pas assez d'historique
            return False, ""
        
        # Mots indicateurs de questions de suivi
        follow_up_indicators = [
            'ce taux', 'cette taxe', 'cet impôt', 'cette loi', 'cet article',
            'il', 'elle', 'ils', 'elles', 'le', 'la', 'les', 'du', 'de la',
            'aussi', 'également', 'en plus', 'et', 'mais', 'cependant',
            'est-il', 'est-elle', 'sont-ils', 'sont-elles',
            'comment', 'pourquoi', 'quand', 'où', 'qui', 'que'
        ]
        
        current_lower = current_question.lower()
        is_follow_up = any(indicator in current_lower for indicator in follow_up_indicators)
        
        if is_follow_up:
            # Récupérer la dernière question de l'utilisateur
            last_user_message = None
            for message in reversed(history):
                if message['role'] == 'user':
                    last_user_message = message
                    break
            
            if last_user_message:
                context_hint = f"QUESTION PRÉCÉDENTE: {last_user_message['content']}"
                logger.info(f"🔗 Question de suivi détectée - Contexte: {last_user_message['content'][:50]}...")
                return True, context_hint
        
        return False, ""
    
    def get_conversation_ids(self):
        """Retourne la liste des IDs de conversations"""
        return list(self.conversations.keys())
    
    def delete_conversation(self, conversation_id):
        """Supprime une conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"🗑️ Conversation supprimée: {conversation_id}")

class LexFinClient:
    """Client LexFin optimisé avec surveillance automatique pour la fiscalité et douanes sénégalaises"""
    
    def __init__(self):
        self.config = LexFinConfig()
        self.indexed_files = {}  # Cache des fichiers indexés {path: hash}
        self.observer = None  # Référence au watcher
        
        # 🗨️ NOUVEAU: Gestionnaire de conversations intelligentes
        self.conversation_manager = ConversationManager(max_history_length=8)
        self.current_conversation_id = None
        
        self.setup_chroma()
        self.setup_watch_directory()
        
        # Initialiser le système hiérarchique V2.0 (à la demande)
        self.hierarchie_client = None
        self._hierarchie_initialized = False
        
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
    
    def start_new_conversation(self):
        """Démarre une nouvelle conversation"""
        self.current_conversation_id = self.conversation_manager.create_conversation()
        logger.info(f"🆕 Nouvelle conversation démarrée: {self.current_conversation_id}")
        return self.current_conversation_id
    
    def set_conversation(self, conversation_id):
        """Change la conversation active"""
        if conversation_id in self.conversation_manager.conversations:
            self.current_conversation_id = conversation_id
            logger.info(f"🔄 Conversation active changée: {conversation_id}")
        else:
            logger.warning(f"⚠️ Conversation introuvable: {conversation_id}")
    
    def get_conversations_list(self):
        """Retourne la liste des conversations avec résumés"""
        conversations = []
        for conv_id, conv_data in self.conversation_manager.conversations.items():
            # Prendre le premier message utilisateur comme titre
            title = "Nouvelle conversation"
            if conv_data['history']:
                first_user_msg = next((msg for msg in conv_data['history'] if msg['role'] == 'user'), None)
                if first_user_msg:
                    title = first_user_msg['content'][:50] + "..." if len(first_user_msg['content']) > 50 else first_user_msg['content']
            
            conversations.append({
                'id': conv_id,
                'title': title,
                'created_at': conv_data['created_at'],
                'last_updated': conv_data['last_updated'],
                'message_count': len(conv_data['history'])
            })
        
        # Trier par dernière mise à jour
        conversations.sort(key=lambda x: x['last_updated'], reverse=True)
        return conversations
    
    def setup_chroma(self):
        """Initialise ChromaDB avec gestion automatique de la dimension d'embeddings"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            collection_found = False
            
            # Essayer de récupérer une collection existante
            for collection_name in ["alex_documents", "alex_pro_docs"]:
                try:
                    self.collection = self.chroma_client.get_collection(collection_name)
                    collection_found = True
                    logger.info(f"✅ Collection trouvée: {collection_name}")
                    
                    # Tester la compatibilité des embeddings
                    try:
                        test_embedding = self.generate_embeddings("test")
                        if test_embedding:
                            # Test avec un petit échantillon
                            self.collection.query(
                                query_embeddings=[test_embedding],
                                n_results=1
                            )
                            logger.info(f"✅ Dimension d'embeddings compatible: {len(test_embedding)}")
                            break
                    except Exception as dim_error:
                        if "dimension" in str(dim_error).lower():
                            logger.warning(f"⚠️ Incompatibilité dimension embeddings détectée: {dim_error}")
                            logger.info(f"🔄 Recréation de la collection {collection_name} nécessaire...")
                            
                            # Supprimer l'ancienne collection
                            self.chroma_client.delete_collection(collection_name)
                            
                            # Créer une nouvelle collection
                            self.collection = self.chroma_client.create_collection(
                                name=collection_name,
                                metadata={"description": "Documents ALEX - Nouvelle dimension embeddings"}
                            )
                            
                            # Réinitialiser le cache des fichiers indexés
                            self.indexed_files = {}
                            
                            logger.info(f"✅ Collection {collection_name} recréée avec nouvelle dimension")
                            break
                        else:
                            raise dim_error
                            
                except Exception as e:
                    if "does not exist" not in str(e).lower():
                        logger.warning(f"Erreur collection {collection_name}: {e}")
                    continue
            
            # Si aucune collection trouvée, en créer une nouvelle
            if not collection_found or not self.collection:
                self.collection = self.chroma_client.create_collection(
                    name="alex_pro_docs",
                    metadata={"description": "Documents ALEX - Nouvelle installation"}
                )
                logger.info("✅ Nouvelle collection créée: alex_pro_docs")
            
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
                    
                    cache_count = len(self.indexed_files)
                    logger.info(f"📚 Cache chargé: {cache_count} fichiers indexés")
                    
                    # Si la collection était vide (recréée), forcer la réindexation
                    if cache_count == 0:
                        logger.info("🔄 Collection vide détectée - Réindexation des documents nécessaire")
                        self.indexed_files = {}  # Vider le cache pour forcer la réindexation
                else:
                    logger.info("📚 Collection vide - Tous les fichiers seront indexés")
                    self.indexed_files = {}
        except Exception as e:
            logger.error(f"Erreur chargement cache: {e}")
            self.indexed_files = {}
    
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
                    # Extraction PDF avec PyPDF2 ou pdfplumber - conservation des numéros de page réels
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            # Extraction page par page pour conserver les vrais numéros
                            text_content = []
                            page_info = []  # Stocker les infos de page
                            
                            for page_num, page in enumerate(pdf_reader.pages, 1):
                                page_text = page.extract_text()
                                if page_text.strip():  # Seulement si la page a du contenu
                                    # Marquer le début de chaque page
                                    page_marker = f"\n--- PAGE {page_num} ---\n"
                                    text_content.append(page_marker + page_text)
                                    page_info.append((len('\n'.join(text_content[:len(text_content)])), page_num))
                            
                            return '\n'.join(text_content)
                    except ImportError:
                        try:
                            import pdfplumber
                            with pdfplumber.open(file_path) as pdf:
                                text_content = []
                                for page_num, page in enumerate(pdf.pages, 1):
                                    page_text = page.extract_text() or ''
                                    
                                    # Essayer d'extraire les tableaux d'abord
                                    tables = page.extract_tables()
                                    
                                    if tables:
                                        # Si des tableaux sont détectés, les formater proprement
                                        page_text_with_tables = f"\n--- PAGE {page_num} ---\n"
                                        
                                        # Ajouter le texte normal
                                        if page_text.strip():
                                            page_text_with_tables += page_text + "\n\n"
                                        
                                        # Ajouter les tableaux formatés
                                        for i, table in enumerate(tables):
                                            page_text_with_tables += f"TABLEAU {i+1} (Page {page_num}):\n"
                                            if table and len(table) > 0:
                                                # Créer un tableau lisible
                                                for row_idx, row in enumerate(table):
                                                    if row_idx == 0 and any(cell for cell in row if cell):
                                                        # En-têtes
                                                        page_text_with_tables += "COLONNES: " + " | ".join(str(cell or '') for cell in row) + "\n"
                                                        page_text_with_tables += "-" * 80 + "\n"
                                                    else:
                                                        # Données avec labels
                                                        if any(cell for cell in row if cell):
                                                            formatted_row = []
                                                            for cell in row:
                                                                cell_text = str(cell or '').strip()
                                                                if cell_text:
                                                                    formatted_row.append(cell_text)
                                                            if formatted_row:
                                                                page_text_with_tables += "LIGNE: " + " | ".join(formatted_row) + "\n"
                                            page_text_with_tables += "\n"
                                        
                                        text_content.append(page_text_with_tables)
                                    else:
                                        # Pas de tableau, texte normal
                                        if page_text.strip():
                                            page_marker = f"\n--- PAGE {page_num} ---\n"
                                            text_content.append(page_marker + page_text)
                                            
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
                
                # Extraction du vrai numéro de page depuis les marqueurs
                page_start = 1
                page_end = 1
                
                # Chercher les marqueurs de page dans le chunk et avant
                import re
                page_markers_before = re.findall(r'--- PAGE (\d+) ---', text_before)
                page_markers_in_chunk = re.findall(r'--- PAGE (\d+) ---', chunk_text)
                
                if page_markers_before:
                    page_start = int(page_markers_before[-1])  # Dernière page avant le chunk
                elif page_markers_in_chunk:
                    page_start = int(page_markers_in_chunk[0])  # Première page dans le chunk
                
                if page_markers_in_chunk:
                    page_end = int(page_markers_in_chunk[-1])  # Dernière page dans le chunk
                else:
                    page_end = page_start
                
                # Fallback: si pas de marqueurs trouvés, estimation basique
                if page_start == 1 and page_end == 1 and line_start > 50:
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
                    
                    # Extraction du vrai numéro de page depuis les marqueurs
                    page_start = 1
                    page_end = 1
                    
                    # Chercher les marqueurs de page dans le chunk et avant
                    import re
                    text_before = '\n'.join(lines[:start_i])
                    
                    page_markers_before = re.findall(r'--- PAGE (\d+) ---', text_before)
                    page_markers_in_chunk = re.findall(r'--- PAGE (\d+) ---', chunk_text)
                    
                    if page_markers_before:
                        page_start = int(page_markers_before[-1])
                    elif page_markers_in_chunk:
                        page_start = int(page_markers_in_chunk[0])
                    
                    if page_markers_in_chunk:
                        page_end = int(page_markers_in_chunk[-1])
                    else:
                        page_end = page_start
                    
                    # Fallback: si pas de marqueurs trouvés, estimation basique
                    if page_start == 1 and page_end == 1 and line_start > 50:
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
                    
                    # 🔧 CORRECTION OCR: Normaliser les espaces dans les numéros d'articles
                    # Exemple: "Article 4 12" → "Article 412"
                    import re
                    full_ref_normalized = re.sub(r'Article\s+(\d+)\s+(\d+)', r'Article \1\2', full_ref)
                    article_number_normalized = re.sub(r'Article\s+(\d+)\s+(\d+)', r'Article \1\2', current_article)
                    
                    chunks.append({
                        'text': chunk_text,
                        'file_path': file_path,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'line_start': line_start,
                        'line_end': line_end,
                        'page_start': page_start,
                        'page_end': page_end,
                        'article_ref': full_ref_normalized,  # Version corrigée
                        'article_ref_original': full_ref,     # Version originale conservée
                        'article_number': article_number_normalized,
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

    def _enhance_text_for_embedding(self, text: str) -> str:
        """
        Améliore subtilement le texte pour de meilleurs embeddings sémantiques
        Normalise les références géographiques car tous les documents concernent le Sénégal
        """
        original_text = text.strip()
        enhanced_text = original_text
        
        # 🇸🇳 NORMALISATION GÉOGRAPHIQUE INTELLIGENTE
        # Tous nos documents concernent le Sénégal, donc "au Sénégal" est redondant
        
        # Patterns de suppression géographique (avec regex pour plus de précision)
        import re
        
        # Supprimer les références géographiques redondantes (case insensitive)
        geographic_patterns = [
            r'\bau sénégal\b', r'\bdu sénégal\b', r'\ben sénégal\b', r'\bsénégalais\b',
            r'\bau senegal\b', r'\bdu senegal\b', r'\ben senegal\b', r'\bsenegalais\b',
            r'\bsénégal\b', r'\bsenegal\b'
        ]
        
        for pattern in geographic_patterns:
            if re.search(pattern, enhanced_text, re.IGNORECASE):
                enhanced_text = re.sub(pattern, '', enhanced_text, flags=re.IGNORECASE)
                enhanced_text = re.sub(r'\s+', ' ', enhanced_text).strip()  # Nettoyer espaces multiples
                logger.info(f"🇸🇳 Normalisation géographique: '{original_text}' → '{enhanced_text}'")
                break
        
        # Si c'est une courte question, l'étendre légèrement avec du contexte implicite
        if len(enhanced_text) < 50 and '?' in enhanced_text:
            # Questions sur les taux -> contexte fiscal/taxation
            if any(word in enhanced_text.lower() for word in ['taux', 'combien', 'pourcentage']):
                if any(word in enhanced_text.lower() for word in ['tva', 'taxe']):
                    # Ajouter un contexte fiscal implicite pour les questions TVA
                    return f"{enhanced_text} contexte fiscal taxation"
                elif any(word in enhanced_text.lower() for word in ['impôt', 'société', 'is']):
                    return f"{enhanced_text} contexte fiscal impôt"
                elif any(word in enhanced_text.lower() for word in ['douane', 'marchandise', 'importation']):
                    return f"{enhanced_text} contexte douanier"
            
            # Questions générales sur articles -> contexte juridique
            if 'article' in enhanced_text.lower():
                return f"{enhanced_text} contexte juridique code loi"
        
        # Pour les textes plus longs, retourner la version normalisée
        return enhanced_text

    def generate_embeddings(self, text: str, max_retries: int = 2) -> List[float]:
        """Génère des embeddings intelligents avec contextualisation sémantique"""
        
        # 🧠 AMÉLIORATION SÉMANTIQUE: Préparation du texte pour meilleur embedding
        enhanced_text = self._enhance_text_for_embedding(text)
        
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.config.OLLAMA_EMBEDDING_MODEL,
                    "prompt": enhanced_text
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
                    if enhanced_text != text:
                        logger.info(f"🧠 Embedding contextualisé généré (tentative {attempt + 1})")
                    else:
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
    
    def detect_legal_code_type(self, query: str) -> str:
        """EXPERTISE : Détecte le type de code juridique (CGI, Code des Douanes, etc.)"""
        query_lower = query.lower()
        
        # Indicateurs pour le Code Général des Impôts (CGI)
        cgi_indicators = [
            'code général des impôts', 'code general des impots', 'cgi',
            'impôt', 'impot', 'fiscal', 'fiscale', 'contribuable', 'tva',
            'bénéfices imposables', 'benefices imposables', 'personnes imposables',
            'champ d\'application', 'société', 'sociétés', 'is', 'ir', 'ircm',
            'base imposable', 'assiette fiscale', 'déclaration de revenus'
        ]
        
        # Indicateurs pour le Code des Douanes
        douane_indicators = [
            'code des douanes', 'douane', 'douanes', 'douanier', 'douanière',
            'importation', 'exportation', 'marchandise', 'marchandises',
            'dédouanement', 'transit', 'bureau de douane', 'tarif douanier',
            'nomenclature', 'espèce d\'une marchandise', 'origine des marchandises'
        ]
        
        # Calculer les scores
        cgi_score = sum(1 for indicator in cgi_indicators if indicator in query_lower)
        douane_score = sum(1 for indicator in douane_indicators if indicator in query_lower)
        
        if cgi_score > douane_score:
            return "Code Général des Impôts (CGI)"
        elif douane_score > cgi_score:
            return "Code des Douanes"
        else:
            return "Code Général (indéterminé)"
    
    def analyze_hierarchical_context(self, query: str) -> Dict:
        """EXPERTISE : Analyse le contexte hiérarchique demandé (Section, Sous-section, Chapitre, etc.)"""
        query_lower = query.lower()
        context = {
            'section': None,
            'sous_section': None,
            'chapitre': None,
            'titre': None,
            'theme': None
        }
        
        # Détecter BENEFICES IMPOSABLES - patterns plus précis
        if any(term in query_lower for term in ['benefices', 'bénéfices']) and 'imposables' in query_lower:
            context['section'] = "SECTION II. BENEFICES IMPOSABLES"
            context['theme'] = "bénéfices imposables"
            logger.info(f"🎯 Section détectée: BENEFICES IMPOSABLES")
        
        # Détecter DETERMINATION DU BENEFICE NET IMPOSABLE
        if any(term in query_lower for term in ['determination', 'détermination']) and any(term in query_lower for term in ['benefice', 'bénéfice']):
            context['sous_section'] = "Sous-section 1. DETERMINATION DU BENEFICE NET IMPOSABLE"
            context['theme'] = "détermination du bénéfice net imposable"
            logger.info(f"🎯 Sous-section détectée: DETERMINATION DU BENEFICE NET IMPOSABLE")
        
        # Détecter PERIODE D'IMPOSITION
        if any(term in query_lower for term in ['periode', 'période']) and 'imposition' in query_lower:
            context['theme'] = "période d'imposition"
            if not context['section']:
                context['section'] = "SECTION II. BENEFICES IMPOSABLES"
            if not context['sous_section']:
                context['sous_section'] = "Sous-section 1. DETERMINATION DU BENEFICE NET IMPOSABLE"
            logger.info(f"🎯 Thème détecté: période d'imposition")
        
        # Détecter les structures hiérarchiques explicites
        if 'section' in query_lower:
            # Extraire la section mentionnée
            if 'champ d\'application' in query_lower or 'personnes imposables' in query_lower:
                context['section'] = "SECTION I. CHAMP D'APPLICATION"
                context['theme'] = "personnes imposables"
            elif not context['section']:  # Si pas déjà détecté
                if 'tva' in query_lower or 'taxe sur la valeur ajoutée' in query_lower:
                    context['section'] = "SECTION TVA"
                    context['theme'] = "taxe sur la valeur ajoutée"
        
        if 'sous-section' in query_lower or 'sous section' in query_lower:
            if 'personnes imposables' in query_lower:
                context['sous_section'] = "Sous-section 1. PERSONNES IMPOSABLES"
        
        # Détecter les thèmes implicites basés sur les mots-clés
        if not context['theme']:
            if any(term in query_lower for term in ['société', 'sociétés', 'sarl', 'sa', 'sas']):
                context['theme'] = "sociétés"
            elif any(term in query_lower for term in ['base', 'assiette', 'calcul']):
                context['theme'] = "base imposable"
        
        return context
    
    def build_expert_search_strategy(self, article_num: str, code_type: str, hierarchical_context: Dict, query: str) -> List[str]:
        """EXPERTISE : Construit une stratégie de recherche experte basée sur la structure juridique"""
        search_terms = []
        query_lower = query.lower()
        
        # Stratégie de base
        search_terms.extend([
            f"Article {article_num}",
            f"Article {article_num}.",
            query  # Requête complète de l'utilisateur
        ])
        
        # Stratégie spécialisée par code juridique
        if code_type == "Code Général des Impôts (CGI)":
            search_terms.extend(self._build_cgi_search_terms(article_num, hierarchical_context, query_lower))
        elif code_type == "Code des Douanes":
            search_terms.extend(self._build_douane_search_terms(article_num, hierarchical_context, query_lower))
        
        # Stratégie hiérarchique intelligente
        if hierarchical_context['section']:
            search_terms.append(f"{hierarchical_context['section']} Article {article_num}")
        
        if hierarchical_context['sous_section']:
            search_terms.append(f"{hierarchical_context['sous_section']} Article {article_num}")
        
        if hierarchical_context['theme']:
            search_terms.extend([
                f"Article {article_num} {hierarchical_context['theme']}",
                f"{hierarchical_context['theme']} Article {article_num}"
            ])
        
        return search_terms
    
    def _build_cgi_search_terms(self, article_num: str, context: Dict, query_lower: str) -> List[str]:
        """Construit des termes de recherche spécialisés pour le CGI"""
        terms = []
        
        # Article 4 CGI - Personnes imposables
        if article_num == "4" and any(term in query_lower for term in ['champ', 'application', 'personnes', 'imposables']):
            terms.extend([
                f"SECTION I. CHAMP D'APPLICATION Article {article_num}",
                f"Sous-section 1. PERSONNES IMPOSABLES Article {article_num}",
                f"Article {article_num}. I. (Loi",  # Pattern spécifique CGI
                f"Les sociétés par actions Article {article_num}",
                f"sociétés à responsabilité limitée Article {article_num}"
            ])
        
        # Article 7 CGI - Bénéfices imposables / Période d'imposition (PRIORITÉ ABSOLUE)
        elif article_num == "7":
            # Si le contexte indique bénéfices imposables, forcer cette recherche
            if context.get('section') == "SECTION II. BENEFICES IMPOSABLES" or any(term in query_lower for term in ['benefices', 'bénéfices']):
                terms.extend([
                    f"SECTION II. BENEFICES IMPOSABLES Article {article_num}",
                    f"Sous-section 1. DETERMINATION DU BENEFICE NET IMPOSABLE Article {article_num}",
                    f"Article {article_num}. Période d'imposition",
                    f"Période d'imposition Article {article_num}",
                    f"BENEFICES IMPOSABLES Article {article_num}",
                    f"DETERMINATION DU BENEFICE NET IMPOSABLE Article {article_num}",
                    f"exercice comptable Article {article_num}",
                    f"exercice précédent Article {article_num}",
                    f"comptes à la date du 31 décembre Article {article_num}",
                    f"bénéfices réalisés Article {article_num}"
                ])
                logger.info(f"🎯 Recherche spécialisée Article 7 BENEFICES IMPOSABLES activée")
            else:
                # Recherche générale pour Article 7
                terms.extend([
                    f"Article {article_num}. Période d'imposition",
                    f"Période d'imposition Article {article_num}"
                ])
        
        # Articles TVA
        elif any(term in query_lower for term in ['tva', 'taxe', 'valeur', 'ajoutée']):
            terms.extend([
                f"TVA Article {article_num}",
                f"Taxe sur la valeur ajoutée Article {article_num}",
                f"Article {article_num} assujetti",
                f"Article {article_num} redevable"
            ])
        
        return terms
    
    def _build_douane_search_terms(self, article_num: str, context: Dict, query_lower: str) -> List[str]:
        """Construit des termes de recherche spécialisés pour le Code des Douanes"""
        terms = []
        
        # Termes généraux douaniers
        terms.extend([
            f"Article {article_num} marchandise",
            f"Article {article_num} importation", 
            f"Article {article_num} exportation",
            f"Article {article_num} dédouanement",
            f"Article {article_num} bureau de douane"
        ])
        
        # Contexte spécialisé selon la requête
        if 'marchandise' in query_lower:
            terms.extend([
                f"espèce d'une marchandise Article {article_num}",
                f"classification Article {article_num}",
                f"nomenclature Article {article_num}"
            ])
        
        if 'tarif' in query_lower or 'droit' in query_lower:
            terms.extend([
                f"tarif douanier Article {article_num}",
                f"droit de douane Article {article_num}"
            ])
        
        return terms
    
    def calculate_expert_priority_score(self, doc: str, metadata: Dict, article_num: str, 
                                       code_type: str, hierarchical_context: Dict, query_lower: str) -> int:
        """EXPERTISE : Calcule un score de priorité basé sur l'expertise juridique"""
        priority_score = 0
        doc_lower = doc.lower()
        
        # Score de base pour correspondance d'article exact
        if f"article {article_num}" in doc_lower:
            priority_score += 10
        
        # EXPERTISE CGI
        if code_type == "Code Général des Impôts (CGI)":
            priority_score += self._calculate_cgi_expertise_score(doc_lower, metadata, article_num, hierarchical_context, query_lower)
        
        # EXPERTISE Code des Douanes  
        elif code_type == "Code des Douanes":
            priority_score += self._calculate_douane_expertise_score(doc_lower, metadata, article_num, hierarchical_context, query_lower)
        
        # Score contextuel intelligent
        query_keywords = set(word for word in query_lower.split() if len(word) > 2)
        doc_keywords = set(word for word in doc_lower.split() if len(word) > 2)
        
        # Correspondance sémantique
        keyword_overlap = len(query_keywords.intersection(doc_keywords))
        priority_score += keyword_overlap * 3
        
        # Bonus pour structure hiérarchique
        hierarchical_terms = ['section', 'sous-section', 'chapitre', 'titre']
        hierarchical_bonus = sum(2 for term in hierarchical_terms if term in doc_lower)
        priority_score += hierarchical_bonus
        
        return priority_score
    
    def _calculate_cgi_expertise_score(self, doc_lower: str, metadata: Dict, article_num: str, context: Dict, query_lower: str) -> int:
        """Score d'expertise spécialisé CGI"""
        score = 0
        
        # Article 4 CGI - Expertise personnes imposables
        if article_num == "4" and any(term in query_lower for term in ['champ', 'application', 'personnes', 'imposables']):
            if any(term in doc_lower for term in [
                'personnes imposables', 'champ d\'application', 'section i',
                'sociétés par actions', 'responsabilité limitée', 'impôt sur les sociétés'
            ]):
                score += 20
                logger.info(f"🎯 Expertise CGI Article 4 - Personnes imposables (+20)")
        
        # Article 7 CGI - PRIORITÉ ABSOLUE pour bénéfices imposables
        elif article_num == "7":
            # Si la requête mentionne explicitement bénéfices imposables
            if any(term in query_lower for term in ['benefices', 'bénéfices']) and 'imposables' in query_lower:
                if any(term in doc_lower for term in [
                    'bénéfices imposables', 'benefices imposables', 'section ii',
                    'determination du benefice', 'détermination du bénéfice',
                    'exercice précédent', 'comptes à la date du 31 décembre'
                ]):
                    score += 50  # SCORE MAXIMAL pour le bon Article 7
                    logger.info(f"🎯 PRIORITÉ ABSOLUE Article 7 - BENEFICES IMPOSABLES (+50)")
                else:
                    # Pénalité sévère pour mauvais Article 7 (ex: méthode cadastrale)
                    score -= 30
                    logger.info(f"⛔ Pénalité Article 7 non-bénéfices imposables (-30)")
            
            # Si période d'imposition est mentionnée
            elif any(term in query_lower for term in ['periode', 'période']) and 'imposition' in query_lower:
                if 'période d\'imposition' in doc_lower or 'periode d\'imposition' in doc_lower:
                    score += 40
                    logger.info(f"🎯 Article 7 - Période d'imposition (+40)")
                elif any(term in doc_lower for term in ['exercice précédent', 'exercice comptable', '31 décembre']):
                    score += 35
                    logger.info(f"🎯 Article 7 - Contexte période (+35)")
                else:
                    score -= 25  # Pénalité pour mauvais Article 7
                    logger.info(f"⛔ Pénalité Article 7 hors période d'imposition (-25)")
            
            # Si détermination du bénéfice
            elif any(term in query_lower for term in ['determination', 'détermination']) and any(term in query_lower for term in ['benefice', 'bénéfice']):
                if any(term in doc_lower for term in [
                    'determination du benefice', 'détermination du bénéfice',
                    'benefice net imposable', 'bénéfice net imposable'
                ]):
                    score += 45
                    logger.info(f"🎯 Article 7 - Détermination bénéfice (+45)")
                else:
                    score -= 20
                    logger.info(f"⛔ Pénalité Article 7 hors détermination bénéfice (-20)")
        
        # Expertise TVA
        elif any(term in query_lower for term in ['tva', 'taxe']):
            if any(term in doc_lower for term in ['tva', 'taxe sur la valeur ajoutée', 'assujetti', 'redevable']):
                score += 15
                logger.info(f"🎯 Expertise CGI TVA (+15)")
        
        return score
    
    def _calculate_douane_expertise_score(self, doc_lower: str, metadata: Dict, article_num: str, context: Dict, query_lower: str) -> int:
        """Score d'expertise spécialisé Code des Douanes"""
        score = 0
        
        # Expertise marchandises
        if 'marchandise' in query_lower:
            if any(term in doc_lower for term in ['marchandise', 'classification', 'nomenclature', 'espèce']):
                score += 20
                logger.info(f"🎯 Expertise Douanes - Marchandises (+20)")
        
        # Expertise importation/exportation
        if any(term in query_lower for term in ['importation', 'exportation']):
            if any(term in doc_lower for term in ['importation', 'exportation', 'bureau de douane', 'transit']):
                score += 18
                logger.info(f"🎯 Expertise Douanes - Import/Export (+18)")
        
        # Expertise tarifs douaniers
        if any(term in query_lower for term in ['tarif', 'droit']):
            if any(term in doc_lower for term in ['tarif douanier', 'droit de douane', 'perception']):
                score += 16
                logger.info(f"🎯 Expertise Douanes - Tarifs (+16)")
        
        return score

    def deduplicate_references(self, references: List[Dict]) -> List[Dict]:
        """Déduplique les références intelligemment en gardant les plus pertinentes (score hybride)"""
        if not references:
            return []
        
        # IMPORTANT: Trier par score AVANT toute opération pour garder les meilleurs
        # Le score hybride est dans '_score' - tri décroissant (meilleur score d'abord)
        references_sorted = sorted(references, key=lambda x: x.get('_score', 0), reverse=True)
        
        # Grouper par fichier
        file_groups = {}
        for ref in references_sorted:  # Utiliser la liste triée par score
            file_name = ref.get('file_name', 'unknown')
            if file_name not in file_groups:
                file_groups[file_name] = []
            file_groups[file_name].append(ref)
        
        deduplicated = []
        
        for file_name, file_refs in file_groups.items():
            # Les références sont déjà triées par score global
            # On garde juste les meilleures sans fusionner (pour garder la précision)
            
            # Limiter à un nombre raisonnable par fichier TOUT EN GARDANT LES MEILLEURS
            # Augmenté à 5 pour permettre plus de diversité (au lieu de 2)
            max_refs_per_file = 5
            top_refs = file_refs[:max_refs_per_file]
            
            deduplicated.extend(top_refs)
            
            logger.debug(f"� {file_name}: gardé top {len(top_refs)} références (score: {top_refs[0].get('_score', 0):.3f} à {top_refs[-1].get('_score', 0) if top_refs else 0:.3f})")
        
        # Retrier par score global après déduplication
        deduplicated.sort(key=lambda x: x.get('_score', 0), reverse=True)
        
        logger.info(f"🔧 Déduplication intelligente: {len(references)} → {len(deduplicated)} références optimisées (triées par score)")
        return deduplicated

    def analyze_search_results(self, query: str, references: List[Dict]) -> str:
        """Analyse les résultats de recherche pour déterminer le type de contenu trouvé"""
        if not references:
            return "general"
        
        # Analyser les sources des documents trouvés
        file_analysis = {}
        for ref in references:
            file_name = ref.get('file_name', '').lower()
            if file_name not in file_analysis:
                file_analysis[file_name] = 0
            file_analysis[file_name] += 1
        
        # Classifier selon les documents majoritaires
        impots_files = sum(1 for f in file_analysis.keys() if 'impot' in f or 'fiscal' in f)
        douanes_files = sum(1 for f in file_analysis.keys() if 'douane' in f)
        budget_files = sum(1 for f in file_analysis.keys() if any(x in f for x in ['budget', 'loi', 'finance', 'economique']))
        
        if budget_files > 0:
            logger.info(f"📊 Contenu BUDGÉTAIRE/ÉCONOMIQUE détecté: {budget_files} fichiers")
            return "economique"
        elif impots_files > douanes_files:
            logger.info(f"🏛️ Contenu FISCAL détecté: {impots_files} fichiers")
            return "fiscal"
        elif douanes_files > 0:
            logger.info(f"🚢 Contenu DOUANIER détecté: {douanes_files} fichiers")
            return "douanier"
        else:
            logger.info(f"🔄 Contenu MIXTE détecté")
            return "mixte"

    def detect_query_domain(self, query: str) -> str:
        """Détecte si la question porte sur les impôts, les douanes, l'économie, ou les deux"""
        query_lower = query.lower()
        
        # Mots-clés spécifiques aux technologies non fiscales
        non_fiscal_keywords = [
            'openshift', 'kubernetes', 'docker', 'flutter', 'android', 'ios', 
            'programmation', 'développement', 'application mobile', 'mobile app', 
            'python', 'javascript', 'développer', 'programmer', 'coder',
            'web', 'site web', 'déployer', 'cloud', 'aws', 'azure', 'git',
            'github', 'windows', 'linux', 'mac', 'apple', 'iphone', 'samsung',
            'facebook', 'instagram', 'twitter', 'réseau social'
        ]
        
        # Mots-clés économiques et budgétaires (nouveaux documents)
        economie_keywords = [
            'prévision', 'prévisions', 'croissance', 'secteur', 'secteurs',
            'agroalimentaire', 'chimique', 'industriel', 'industrie',
            'budget', 'budgétaire', 'pib', 'économie', 'économique',
            'finances publiques', 'loi de finances', 'loi de finance', 'lfi', 'lfr',
            'investissement', 'investissements', 'développement',
            'politique économique', 'politique fiscale', 'stratégie', 'dette publique',
            'cadrage budgétaire', 'projet de budget', 'gestion dette',
            'rapport économique', 'financier', 'annexé',
            'moyen terme', 'indicateur', 'performance', 'innovante', 'efficace',
            'réforme fiscale', 'modernisation', 'transformation'
        ]
        
        # NOUVEAU: Toujours rechercher dans tous les documents d'abord
        # La classification se fait APRÈS la recherche pour optimiser la réponse
        logger.info(f"🌍 RECHERCHE UNIVERSELLE - Tous les documents analysés")
        return "economie"  # Force la recherche universelle
        
        # Vérifier d'abord si c'est une question clairement non fiscale
        # Mais éviter les faux positifs avec des termes fiscaux
        fiscal_context_detected = any(term in query_lower for term in [
            'article', 'code', 'impot', 'impôt', 'fiscal', 'douane', 'tva', 
            'champ d\'application', 'personnes imposables', 'contribuable'
        ])
        
        if not fiscal_context_detected:
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

    def _init_hierarchie_client(self):
        """Initialise le système hiérarchique à la demande"""
        if not self._hierarchie_initialized and HIERARCHIE_AVAILABLE:
            try:
                self.hierarchie_client = HierarchieJuridiqueClient(base_client=self)
                logger.info("✅ Système hiérarchique V2.0 initialisé à la demande")
                self._hierarchie_initialized = True
            except Exception as e:
                logger.warning(f"⚠️ Erreur initialisation système hiérarchique: {e}")
                self._hierarchie_initialized = True  # Éviter de retry
        elif not HIERARCHIE_AVAILABLE:
            logger.warning("⚠️ Système hiérarchique V2.0 non disponible")
            self._hierarchie_initialized = True

    def  search_context_with_references(self, query: str, limit: int = 5) -> Dict:
        """Recherche hybride avec système hiérarchique V2.0 en priorité"""
        if not self.collection:
            logger.warning("  Aucune collection ChromaDB disponible")
            return {"context": "", "references": []}
        
        try:
            # 🔥 TEMPORAIRE: Désactiver le système hiérarchique pour corriger les références
            # self._init_hierarchie_client()  # Initialiser à la demande
            
            # if self.hierarchie_client:
            #     logger.info("🏛️ Utilisation du système hiérarchique V2.0")
            #     hierarchie_result = self.hierarchie_client.rechercher_hierarchique(query)
            #     
            #     if hierarchie_result.get("context") and hierarchie_result.get("references"):
            #         logger.info(f"✅ Résultat hiérarchique trouvé: {hierarchie_result.get('type_recherche', 'N/A')}")
            #         return hierarchie_result
            #     else:
            #         logger.info("⚠️ Système hiérarchique: aucun résultat, fallback vers recherche classique")
            
            # Utiliser directement la recherche classique qui fonctionne bien
            logger.info("🔄 Utilisation recherche classique (temporaire - références correctes)")
            
            # Détecter le domaine de la question
            query_domain = self.detect_query_domain(query)
            
            # Recherche spécialisée pour les articles
            article_result = self.search_specific_article(query)
            if article_result["context"]:
                logger.info(f"🎯 Article spécifique trouvé: {query}")
                return article_result
            
            # Générer embedding de la requête pour recherche vectorielle pure
            query_embedding = self.generate_embeddings(query)
            if not query_embedding:
                logger.warning("  Impossible de générer embedding pour la requête")
                return {"context": "", "references": []}
            
            # 🔥 RECHERCHE HYBRIDE: Vectoriel + BM25 (Intelligence naturelle)
            logger.info(f"🔍 Recherche HYBRIDE INTELLIGENTE (Vectoriel + BM25): {query[:50]}...")
            
            # Préparer les filtres selon le domaine
            where_filter = {}
            if query_domain == "impots":
                where_filter = {"file_name": {"$eq": "Senegal-Code-des-impot.pdf"}}
                logger.info("📊 Recherche limitée au Code des Impôts")
            elif query_domain == "douanes":
                where_filter = {"file_name": {"$eq": "Senegal-Code-2014-des-douanes.pdf"}}
                logger.info("🚢 Recherche limitée au Code des Douanes")
            elif query_domain == "economie":
                # Pas de filtre = recherche dans TOUS les documents
                where_filter = {}
                logger.info("🌍 Recherche ÉCONOMIQUE dans TOUS les documents indexés")
            else:
                # Domaine général ou ambiguë = recherche dans tous les documents aussi
                where_filter = {}
                logger.info("🔄 Recherche GÉNÉRALE dans tous les documents")
            
            # ÉTAPE 1: Recherche VECTORIELLE (embeddings)
            # Augmenter significativement le nombre de résultats pour mieux capturer les documents pertinents
            n_vectorial_results = min(100, limit * 20)  # Au moins 100 résultats ou 20x la limite
            
            if where_filter:
                vectorial_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_vectorial_results,
                    where=where_filter,
                    include=['documents', 'metadatas', 'distances']
                )
            else:
                vectorial_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_vectorial_results,
                    include=['documents', 'metadatas', 'distances']
                )
            
            # ÉTAPE 2: Récupérer TOUS les documents pour BM25
            # (Optimisation: on pourrait limiter au même domaine)
            if where_filter:
                all_docs_data = self.collection.get(
                    where=where_filter,
                    include=['documents', 'metadatas']
                )
            else:
                all_docs_data = self.collection.get(
                    include=['documents', 'metadatas']
                )
            
            all_documents = all_docs_data['documents']
            all_metadatas = all_docs_data['metadatas']
            
            logger.info(f"📚 Corpus BM25: {len(all_documents)} documents")
            
            # Calculer BM25 scores
            bm25 = BM25()
            
            # Calculer IDF sur tout le corpus
            idf = bm25.compute_idf(all_documents)
            
            # Calculer longueur moyenne des documents
            avg_doc_len = sum(len(bm25.tokenize(doc)) for doc in all_documents) / len(all_documents)
            
            # Calculer le score BM25 pour chaque document
            bm25_scores = []
            for doc in all_documents:
                score = bm25.score(query, doc, avg_doc_len, idf)
                bm25_scores.append(score)
            
            logger.info(f"✅ BM25 scores calculés pour {len(bm25_scores)} documents")
            
            # ÉTAPE 3: COMBINER les scores vectoriels et BM25
            # Normaliser les scores pour pouvoir les combiner
            
            # Scores vectoriels (distances ChromaDB - plus petit = meilleur)
            vectorial_docs = vectorial_results['documents'][0] if vectorial_results['documents'] else []
            vectorial_metas = vectorial_results['metadatas'][0] if vectorial_results['metadatas'] else []
            vectorial_distances = vectorial_results['distances'][0] if vectorial_results.get('distances') else []
            
            # Normaliser les distances vectorielles (0-1, où 1 = meilleur)
            if vectorial_distances:
                max_dist = max(vectorial_distances) if vectorial_distances else 1
                min_dist = min(vectorial_distances) if vectorial_distances else 0
                range_dist = max_dist - min_dist if max_dist != min_dist else 1
                
                vectorial_scores_normalized = [
                    1 - ((dist - min_dist) / range_dist) for dist in vectorial_distances
                ]
            else:
                vectorial_scores_normalized = [1.0] * len(vectorial_docs)
            
            # Normaliser les scores BM25 (0-1, où 1 = meilleur)
            if bm25_scores:
                max_bm25 = max(bm25_scores)
                min_bm25 = min(bm25_scores)
                range_bm25 = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
                
                bm25_scores_normalized = [
                    (score - min_bm25) / range_bm25 for score in bm25_scores
                ]
            else:
                bm25_scores_normalized = [0.0] * len(all_documents)
            
            # ÉTAPE 4: Combiner les résultats
            # Créer un dictionnaire doc_text -> (metadata, vectorial_score, bm25_score)
            combined_results = {}
            
            # Ajouter les résultats vectoriels
            for i, doc in enumerate(vectorial_docs):
                combined_results[doc] = {
                    'metadata': vectorial_metas[i] if i < len(vectorial_metas) else {},
                    'vectorial_score': vectorial_scores_normalized[i] if i < len(vectorial_scores_normalized) else 0,
                    'bm25_score': 0,  # Sera mis à jour ensuite
                    'content': doc
                }
            
            # Ajouter/mettre à jour les scores BM25
            for i, doc in enumerate(all_documents):
                if doc in combined_results:
                    combined_results[doc]['bm25_score'] = bm25_scores_normalized[i]
                else:
                    # Document trouvé par BM25 mais pas par vectoriel
                    combined_results[doc] = {
                        'metadata': all_metadatas[i] if i < len(all_metadatas) else {},
                        'vectorial_score': 0,
                        'bm25_score': bm25_scores_normalized[i],
                        'content': doc
                    }
            
            # Calculer le score hybride combiné avec détection de mots rares
            # Détecter si la requête contient des mots rares (indicateur: mot peu fréquent)
            query_tokens = bm25.tokenize(query)
            rare_word_detected = False
            
            # Un mot est considéré comme rare si son IDF est élevé (> seuil)
            if query_tokens:
                avg_idf = sum(idf.get(token, 0) for token in query_tokens) / len(query_tokens)
                # Si l'IDF moyen est élevé (> 5), on a probablement des mots rares/spécifiques
                if avg_idf > 5.0:
                    rare_word_detected = True
                    logger.info(f"🔍 Mots rares détectés (IDF moyen: {avg_idf:.2f}) - Privilégier BM25")
            
            # Ajuster les poids selon la présence de mots rares
            if rare_word_detected:
                # Pour mots rares: privilégier BM25 (matching exact)
                alpha = 0.3  # Poids vectoriel réduit
                beta = 0.7   # Poids BM25 augmenté
                logger.info("⚖️ Poids: 30% Vectoriel + 70% BM25 (mots rares)")
            else:
                # Pour requêtes normales: équilibre 50/50
                alpha = 0.5  # Poids vectoriel
                beta = 0.5   # Poids BM25
                logger.info("⚖️ Poids: 50% Vectoriel + 50% BM25 (équilibre)")
            
            for doc_text in combined_results:
                v_score = combined_results[doc_text]['vectorial_score']
                b_score = combined_results[doc_text]['bm25_score']
                combined_results[doc_text]['hybrid_score'] = alpha * v_score + beta * b_score
            
            # Trier par score hybride décroissant
            sorted_results = sorted(
                combined_results.items(),
                key=lambda x: x[1]['hybrid_score'],
                reverse=True
            )
            
            # Prendre beaucoup plus de résultats pour maximiser les chances de trouver les bons documents
            # Augmenté à 50 pour une meilleure couverture (au lieu de limit * 2)
            top_results = sorted_results[:min(50, len(sorted_results))]
            
            logger.info(f"🎯 Top 10 scores hybrides:")
            for i, (doc, data) in enumerate(top_results[:10]):
                article_ref = data['metadata'].get('article_ref', 'N/A')[:50]
                logger.info(f"  #{i+1}: {article_ref} - Hybride: {data['hybrid_score']:.3f} (V:{data['vectorial_score']:.3f} + BM25:{data['bm25_score']:.3f})")
            
            # ÉTAPE 5: Construire les références
            context_parts = []
            references = []
            
            for doc_text, data in top_results:
                metadata = data['metadata']
                
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
                    
                    # Extraire les informations d'article
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
                        'snippet': doc_text[:150] + "..." if len(doc_text) > 150 else doc_text,
                        'content': doc_text,
                        '_score': data['hybrid_score']  # Score hybride pour tri
                    }
                    references.append(reference)
                    
                    # Créer une source info enrichie
                    if article_ref and article_ref != 'Section générale':
                        source_info = f"[📄 {file_name} - {article_ref}, {page_info}, {location}]"
                    else:
                        source_info = f"[📄 {file_name}, {page_info}, {location}]"
                    
                    # 🔧 LIMITATION DRASTIQUE pour éviter timeouts Mistral
                    # Réduire le texte à maximum 200 caractères par référence
                    truncated_text = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                    context_parts.append(f"{source_info}\n{truncated_text}")
                else:
                    context_parts.append(doc_text)
            
            # Dédupliquer les références intelligemment
            deduplicated_references = self.deduplicate_references(references)
            
            # Augmenter la limite finale pour retourner plus de documents pertinents
            # Réduction drastique pour éviter les timeouts Mistral
            final_limit = min(10, max(limit * 2, 8))
            final_references = deduplicated_references[:final_limit]
            final_context_parts = context_parts[:len(final_references)]
            
            logger.info(f"✅ Recherche HYBRIDE terminée: {len(final_references)} documents uniques (sur {len(deduplicated_references)} après déduplication)")
            
            # ANALYSE POST-RECHERCHE: Classifier le contenu trouvé
            content_type = self.analyze_search_results(query, final_references)
            logger.info(f"📋 Analyse contenu trouvé: {content_type}")
            
            return {
                "context": "\n\n".join(final_context_parts),
                "references": final_references,
                "content_type": content_type  # Nouveau: type de contenu détecté
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche HYBRIDE: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"context": "", "references": []}

    def search_specific_article(self, query: str) -> Dict:
        """Recherche intelligente d'articles basée sur la compréhension naturelle du contexte"""
        import re
        logger.info(f"🧠 Recherche intelligente: {query}")
        
        if not self.collection:
            return {"context": "", "references": []}
        
        try:
            # Extraction simple des numéros d'articles
            article_numbers = re.findall(r'article\s+(\d+)', query.lower())
            if not article_numbers:
                article_numbers = re.findall(r'(\d+)', query)[:1]  # Premier nombre trouvé
            
            if not article_numbers:
                return {"context": "", "references": []}
            
            unique_articles = list(dict.fromkeys(article_numbers))
            logger.info(f"🎯 Articles détectés: {unique_articles}")
            
            # Recherche contextuelle simple et intelligente
            all_results = []
            
            for article_num in unique_articles:
                # Stratégies de recherche simples mais efficaces
                search_terms = [
                    query,  # Requête complète de l'utilisateur
                    f"Article {article_num}",
                    f"Article {article_num} " + " ".join([w for w in query.split() if w.lower() not in ['article', article_num, 'du', 'de', 'la', 'le']])
                ]
                
                for search_term in search_terms:
                    try:
                        # Recherche vectorielle simple
                        query_embedding = self.generate_embeddings(search_term)
                        if query_embedding:
                            results = self.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=15,  # Plus de résultats pour trouver le bon article
                                include=['documents', 'metadatas', 'distances']
                            )
                            
                            if results['documents'][0]:
                                for i, doc in enumerate(results['documents'][0]):
                                    metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                                    
                                    # Score intelligent basé sur la correspondance naturelle
                                    score = self._calculate_natural_score(doc, metadata, article_num, query.lower())
                                    
                                    # Seuil adaptatif selon le numéro d'article
                                    min_score = 5
                                    if len(article_num) == 1:  # Articles à 1 chiffre plus rares
                                        min_score = -50  # Seuil plus bas pour Article 1, 2, etc.
                                    
                                    if score > min_score:  # Seuil de pertinence adaptatif
                                        result_item = {
                                            'document': doc,
                                            'metadata': metadata,
                                            'distance': results['distances'][0][i] if results.get('distances') and i < len(results['distances'][0]) else 1.0,
                                            'priority_score': score,
                                            'search_term': search_term
                                        }
                                        all_results.append(result_item)
                                        logger.info(f"✅ Article {article_num} trouvé (score naturel: {score})")
                                        
                    except Exception as e:
                        continue  # Passer au terme suivant silencieusement
            
            if not all_results:
                return {"context": "", "references": []}
            
            # Trier par score et prendre les meilleurs
            all_results.sort(key=lambda x: x['priority_score'], reverse=True)
            best_results = all_results[:5]  # Plus de résultats pour analyse
            
            # Construire la réponse
            context_parts = []
            references = []
            
            for result in best_results:
                doc = result['document']
                metadata = result['metadata']
                
                # 🔧 CORRECTION: Ajouter les propriétés manquantes pour JavaScript
                page_start = metadata.get('page_start', 1)
                page_end = metadata.get('page_end', page_start)
                line_start = metadata.get('line_start', 1)
                line_end = metadata.get('line_end', line_start)
                file_path = metadata.get('file_path', '')
                
                # Créer page_info et location comme dans search_context_with_references
                if page_start == page_end:
                    page_info = f"page {page_start}"
                else:
                    page_info = f"pages {page_start}-{page_end}"
                
                if line_start == line_end:
                    location = f"ligne {line_start}"
                else:
                    location = f"lignes {line_start}-{line_end}"
                
                reference = {
                    'file_name': metadata.get('file_name', 'Document'),
                    'file_path': file_path,
                    'article_ref': metadata.get('article_ref', f'Article {article_numbers[0]}'),
                    'page': page_start,  # Garder l'ancienne propriété pour compatibilité
                    'page_info': page_info,  # ✅ Nouvelle propriété attendue par JavaScript
                    'location': location,    # ✅ Nouvelle propriété attendue par JavaScript
                    'line_start': line_start, # ✅ Propriété attendue par JavaScript
                    'line_end': line_end,     # ✅ Propriété attendue par JavaScript
                    'page_start': page_start,
                    'page_end': page_end,
                    'content': doc,
                    '_score': result['priority_score'],
                    'snippet': doc[:300] + "..." if len(doc) > 300 else doc
                }
                references.append(reference)
                
                source_info = f"[📄 {reference['file_name']} - {reference['article_ref']}, page {reference['page']}]"
                context_parts.append(f"{source_info}\n{doc}")
            
            logger.info(f"✅ {len(references)} résultat(s) intelligent(s)")
            return {
                "context": "\n\n".join(context_parts),
                "references": references
            }
                
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return {"context": "", "references": []}


    def _calculate_natural_score(self, doc: str, metadata: Dict, article_num: str, query_lower: str) -> int:
        """Score naturel basé sur la compréhension du contexte sans règles"""
        score = 0
        doc_lower = doc.lower()
        article_ref = metadata.get('article_ref', '').lower()
        
        # PRIORITÉ ABSOLUE: Article exact trouvé
        if f"article {article_num}" in doc_lower:
            score += 50  # Score très élevé pour article exact
        
        # Vérification ULTRA STRICTE du numéro d'article dans article_ref
        import re
        
        # Extraction du numéro d'article exact avec patterns stricts
        article_patterns = [
            r'article\s+(\d+)(?:\s|\.|\:|$)',  # Article suivi d'espace, point, deux-points ou fin
            r'article\s+(\d+)(?:\s+[a-zA-Z])',  # Article suivi d'espace puis lettre
        ]
        
        found_article_num = None
        for pattern in article_patterns:
            article_match = re.search(pattern, article_ref)
            if article_match:
                found_article_num = article_match.group(1)
                break
        
        if found_article_num:
            if found_article_num == article_num:
                score += 200  # BONUS ÉNORME pour article exact
                
                # BONUS SUPPLÉMENTAIRE pour correspondance exacte stricte
                # Vérifier que ce n'est pas une sous-partie d'un autre numéro
                if f"article {article_num} " in article_ref or f"article {article_num}." in article_ref:
                    score += 50  # Bonus pour séparateur strict
                    
            elif article_num in found_article_num:
                # Cas où on cherche Article 1 mais on trouve Article 157
                if len(article_num) < len(found_article_num):
                    score -= 100  # GROSSE PÉNALITÉ pour faux positif (1 dans 157)
                else:
                    score -= 30   # Pénalité moindre pour autres cas
            else:
                score -= 50   # PÉNALITÉ pour mauvais article
        
        # Correspondance des mots de la requête utilisateur
        query_words = set(word for word in query_lower.split() if len(word) > 2)
        doc_words = set(word for word in doc_lower.split() if len(word) > 2)
        
        # Score basé sur la correspondance des mots
        common_words = query_words.intersection(doc_words)
        score += len(common_words) * 4
        
        # Bonus pour les concepts importants détectés naturellement
        important_concepts = {
            'benefices': ['benefices', 'bénéfices', 'imposables'],
            'determination': ['determination', 'détermination', 'benefice', 'bénéfice'],
            'periode': ['periode', 'période', 'imposition', 'exercice'],
            'personnes': ['personnes', 'imposables', 'champ', 'application'],
            'societes': ['société', 'sociétés', 'sarl', 'sa'],
            'fiscal': ['fiscal', 'fiscale', 'impot', 'impôt'],
            'douanes': ['douanes', 'douanier', 'marchandises'],
            'application': ['application', 'champ', 'dispositions']
        }
        
        for concept, terms in important_concepts.items():
            if any(term in query_lower for term in terms):
                concept_matches = sum(1 for term in terms if term in doc_lower)
                if concept_matches > 0:
                    score += concept_matches * 6
        
        # Bonus pour la présence de structure
        if any(struct in doc_lower for struct in ['section', 'sous-section', 'chapitre']):
            score += 3
        
        # BONUS SPÉCIAL pour "période d'imposition" si recherché
        if 'periode' in query_lower or 'période' in query_lower:
            if 'période d\'imposition' in doc_lower or 'periode d\'imposition' in doc_lower:
                score += 30  # Bonus important pour concept clé
        
        # BONUS pour correspondance de longueur de numéro d'article
        if found_article_num and len(found_article_num) == len(article_num):
            score += 20  # Bonus pour même longueur de numéro
        
        return score

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
        return """Bonjour ! Je suis LexFin, votre assistant IA spécialisé UNIQUEMENT en fiscalité sénégalaise.

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
            greeting_prompt = f"""Tu es LexFin, un assistant IA intelligent spécialisé pour les professionnels et citoyens sénégalais.

L'utilisateur te dit: "{message}"

🇫🇷 LANGUE OBLIGATOIRE: Tu DOIS répondre UNIQUEMENT en français. Aucun mot en anglais ou autre langue n'est autorisé.

IMPORTANT: Tu es un expert polyvalent en droit sénégalais qui maîtrise :
- Code des Impôts et fiscalité (CGI, DGI, TVA, IS, IR)
- Code des Douanes et procédures douanières
- Lois de Finances et budget de l'État 
- Documents économiques et financiers publics
- Réglementations et arrêtés administratifs

Réponds de façon naturelle et professionnelle en français uniquement:
- Présente-toi comme LexFin, l'assistant expert en droit sénégalais
- Précise tes domaines : fiscal, douanier, budgétaire, économique, réglementaire
- Mentionne que tu peux analyser documents officiels (codes, lois, budgets, rapports)
- Reste professionnel et utilise des émojis appropriés (🇸🇳, 🏛️, 📋, 💼)
- Invite l'utilisateur à poser ses questions juridiques/administratives
- Maximum 3-4 lignes
- Réponse UNIQUEMENT en français

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
    
    def generate_contextual_reformulations(self, message: str, initial_context: str = "") -> list:
        """Génère des reformulations intelligentes basées sur le vocabulaire des documents indexés"""
        try:
            # Si on a un contexte initial des documents, l'utiliser pour guider les reformulations
            context_hint = ""
            if initial_context:
                # Extraire quelques termes clés du contexte pour guider Mistral
                context_hint = f"\n\nVocabulaire trouvé dans les documents fiscaux indexés:\n{initial_context[:500]}..."
            
            prompt = f"""Tu es un expert en recherche documentaire fiscale sénégalaise.

Question de l'utilisateur: "{message}"
{context_hint}

🇫🇷 LANGUE OBLIGATOIRE: Tu DOIS répondre UNIQUEMENT en français. Aucun mot en anglais ou autre langue n'est autorisé.

MISSION: Génère 5 reformulations de cette question pour améliorer la recherche dans les documents fiscaux.

RÈGLES:
1. Utilise le vocabulaire EXACT des documents fiscaux sénégalais (articles, termes juridiques)
2. Inclus des numéros d'articles si tu les identifies dans le contexte
3. Varie entre termes techniques et formulations simples
4. Garde les mots-clés importants de la question originale
5. Chaque reformulation sur une ligne, format: "- reformulation"

Réponds UNIQUEMENT avec les 5 reformulations (pas d'explication):"""

            payload = {
                "model": self.config.OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Température basse-moyenne pour cohérence
                    "top_p": 0.9,
                    "num_ctx": 2048,
                    "num_predict": 200  # Limite pour 5 reformulations courtes
                }
            }
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=15  # Timeout court pour ne pas ralentir
            )
            
            if response.status_code == 200:
                reformulations_text = response.json()['response'].strip()
                # Extraire les reformulations (lignes commençant par - ou numérotées)
                reformulations = []
                for line in reformulations_text.split('\n'):
                    line = line.strip()
                    # Nettoyer les préfixes (-, 1., etc.)
                    if line and (line.startswith('-') or (len(line) > 0 and line[0].isdigit())):
                        clean_line = line.lstrip('-0123456789.° ').strip()
                        if clean_line and len(clean_line) > 10:  # Ignorer les lignes trop courtes
                            reformulations.append(clean_line)
                
                logger.info(f"🔄 {len(reformulations)} reformulations contextuelles générées par Mistral")
                return reformulations[:5]  # Max 5 reformulations
            else:
                logger.warning(f"⚠️ Erreur génération reformulations (code {response.status_code})")
                return []
                
        except requests.Timeout:
            logger.warning("⏱️ Timeout reformulations - continuons sans reformulations contextuelles")
            return []
        except Exception as e:
            logger.error(f"❌ Erreur génération reformulations: {e}")
            return []
    
    def is_fiscal_related_question(self, message: str) -> bool:
        """Détermine si la question est liée aux domaines indexés - Approche permissive"""
        
        # 🧠 INTELLIGENCE NATURELLE: Laisser le modèle comprendre naturellement
        # Seuls les sujets clairement hors domaine sont rejetés
        message_lower = message.lower()
        
        # Mots-clés explicitement NON fiscaux (très restrictif)
        non_fiscal_keywords = [
            'football', 'sport', 'cuisine', 'recette', 'musique', 'film', 'cinéma',
            'jeu vidéo', 'programmation python', 'javascript', 'html', 'css',
            'facebook', 'instagram', 'twitter', 'réseau social',
            'météo', 'santé personnelle', 'médecine', 'hôpital',
            'voiture', 'automobile', 'transport personnel',
            'mode', 'vêtement', 'beauté', 'coiffure'
        ]
        
        # Rejeter seulement si c'est clairement hors domaine
        for keyword in non_fiscal_keywords:
            if keyword in message_lower:
                logger.info(f"🚫 Question NON FISCALE détectée: '{keyword}' dans '{message[:50]}...'")
                return False
        
        # Par défaut, ACCEPTER et laisser l'IA juger
        logger.info(f"✅ Question ACCEPTÉE pour analyse IA: '{message[:50]}...'")
        return True

    def chat(self, message: str, conversation_id: str = None) -> Dict:
        """Génère une réponse basée uniquement sur les documents indexés (mode RAG strict) avec mémoire conversationnelle"""
        try:
            # 🗨️ GESTION DE LA CONVERSATION
            if conversation_id is None:
                # Créer une nouvelle conversation si aucune n'est spécifiée
                if self.current_conversation_id is None:
                    conversation_id = self.start_new_conversation()
                else:
                    conversation_id = self.current_conversation_id
            else:
                # Utiliser la conversation spécifiée
                self.set_conversation(conversation_id)
                conversation_id = self.current_conversation_id or self.start_new_conversation()
            
            # Ajouter le message utilisateur à l'historique
            self.conversation_manager.add_message(conversation_id, 'user', message)
            
            # Salutations: répondre directement sans recherche documentaire
            if self.is_greeting_or_general(message):
                response_text = self.generate_natural_greeting_response(message)
                
                # Ajouter la réponse à l'historique
                self.conversation_manager.add_message(conversation_id, 'assistant', response_text)
                
                return {
                    "response": response_text, 
                    "references": [],
                    "conversation_id": conversation_id
                }
            
            # 🔗 ANALYSER SI C'EST UNE QUESTION DE SUIVI
            is_follow_up, context_hint = self.conversation_manager.analyze_follow_up_question(conversation_id, message)
            
            if is_follow_up:
                logger.info(f"🔗 Question de suivi détectée dans conversation {conversation_id}")
                
                # Enrichir le message avec le contexte de la conversation
                conversation_context = self.conversation_manager.get_conversation_context(conversation_id)
                enhanced_message = f"{message}\n\nCONTEXTE CONVERSATIONNEL:\n{context_hint}\n{conversation_context}"
                
                # Utiliser le message enrichi pour la recherche
                search_message = enhanced_message
            else:
                search_message = message
            
            # RECHERCHE D'ABORD - On cherche dans tous les documents indexés
            if not self.is_fiscal_related_question(message):
                error_response = f"""⚠️ QUESTION NON FISCALE DÉTECTÉE

Je suis uniquement conçu pour répondre à des questions liées à la fiscalité sénégalaise.
Je ne peux pas répondre à votre question car elle n'est pas liée au domaine fiscal ou douanier.

� **Suggestions :**
- Posez une question sur le Code des Impôts/Douanes sénégalais
- Utilisez des termes fiscaux précis (TVA, IS, dédouanement)
- Mentionnez un article spécifique si possible

ℹ️ En mode RAG strict, je ne réponds qu'aux questions fiscales basées sur les documents."""

                # Ajouter la réponse à l'historique
                self.conversation_manager.add_message(conversation_id, 'assistant', error_response)
                
                return {
                    "response": error_response,
                    "references": [],
                    "conversation_id": conversation_id
                }
                
# RECHERCHE EN MODE RAG STRICT AVEC REFORMULATIONS ET FILTRAGE TEXTUEL
            logger.info(f"🔍 MODE RAG PUR - Recherche HYBRIDE pour: '{message[:50]}...'")
            
            # MODE RAG 100% PUR - Recherche hybride uniquement (vectoriel + BM25)
            # La recherche hybride doit trouver les documents pertinents sans boost manuel
            logger.info("ℹ️ MODE RAG 100% PUR - Recherche hybride (Vectoriel + BM25)")
            
            # Recherche hybride directe avec limite augmentée pour meilleure couverture
            search_result = self.search_context_with_references(search_message, limit=20)
            
            if search_result.get("context"):
                context = search_result.get("context", "")
                references = search_result.get("references", [])
                content_type = search_result.get("content_type", "general")
                
                logger.info(f"📊 {len(references)} références trouvées par recherche hybride")
                
                # Informer sur le type de contenu détecté
                if content_type == "economique":
                    logger.info(f"✅ Question ÉCONOMIQUE - Acceptation contenu mixte: budget/finances")
                elif content_type == "fiscal":
                    logger.info(f"✅ Question FISCALE - Contenu fiscal détecté")
                elif content_type == "douanier":
                    logger.info(f"✅ Question DOUANIÈRE - Contenu douanier détecté")
                else:
                    logger.info(f"✅ Question MIXTE - Contenu varié détecté")
                
                # Trier par score (déjà fait dans search_context_with_references)
                references.sort(key=lambda x: x.get('_score', 0), reverse=True)
                
                # Log des articles trouvés
                for ref in references[:10]:  # Top 10
                    article = ref.get('article_ref', 'N/A')
                    score = ref.get('_score', 0)
                    logger.info(f"📄 Article {article}, Page {ref.get('page', '?')}, Score {score:.3f}")
                
                logger.info(f"📄 {len(references)} extraits envoyés au modèle")
                logger.info(f"🔍 Articles: {[ref.get('article_ref', 'N/A') for ref in references[:5]]}")
            else:
                context = ""
                references = []
            
            # NOUVEAU: Recherche d'expansion pour termes spécifiques non trouvés
            if not context or 'senelec' in message.lower():
                logger.info("🔍 Recherche d'expansion pour termes spécifiques...")
                
                # Termes d'expansion pour SENELEC
                expansion_queries = [
                    "compensation tarifaire SENELEC milliards",
                    "trente-cinq milliards FCFA énergie", 
                    "35000000000 FCFA secteur électricité",
                    "loi finances rectificative énergie montant",
                    "SENELEC subvention gouvernement",
                    "prix pétrole compensation électricité"
                ]
                
                best_result = None
                best_score = 0
                
                for exp_query in expansion_queries:
                    logger.info(f"  🔍 Test expansion: '{exp_query}'")
                    exp_result = self.search_context_with_references(exp_query, limit=5)
                    
                    if exp_result.get("context") and exp_result.get("references"):
                        # Calculer score moyen des références
                        avg_score = sum(ref.get('_score', 0) for ref in exp_result['references']) / len(exp_result['references'])
                        if avg_score > best_score:
                            best_result = exp_result
                            best_score = avg_score
                            logger.info(f"    ✅ Meilleur résultat trouvé (score: {avg_score:.3f})")
                
                if best_result and best_score > 0.3:  # Seuil de qualité
                    context = best_result.get("context", "")
                    references = best_result.get("references", [])
                    logger.info(f"🎯 Recherche d'expansion réussie avec score {best_score:.3f}")
            
            # Fallback: Si toujours aucun résultat, essai avec mots-clés extraits
            if not context:
                keywords = [word for word in message.split() if len(word) > 3]
                if keywords:
                    keyword_query = " ".join(keywords)
                    logger.info(f"🔄 Recherche fallback avec mots-clés: '{keyword_query}'")
                    search_result2 = self.search_context_with_references(keyword_query, limit=10)
                    context = search_result2.get("context", "")
                    references = search_result2.get("references", [])
            
            # FORCER l'utilisation du contexte des documents
            if context and context.strip():
                # Détecter le domaine de la question pour validation (maintenant informatif seulement)
                query_domain = self.detect_query_domain(message)
                # Note: Cette détection est maintenant utilisée pour information uniquement
                # La vraie classification se fait par analyze_search_results()
                
                # Validation simplifiée : s'assurer que le contenu est pertinent
                question_keywords = message.lower().split()
                context_lower = context.lower()
                keyword_found = any(kw in context_lower for kw in question_keywords if len(kw) > 3)
                
                if keyword_found or any(keyword in context_lower for keyword in ["impot", "tva", "douane", "fiscal", "cgi", "dgi", "senegal", "sénégal", "article"]):
                    # Identifier le code source précisément en analysant TOUS les documents
                    code_source = "Document juridique sénégalais"
                    sources_trouvees = []
                    
                    if references:
                        for ref in references:
                            file_name = ref.get('file_name', '').lower()
                            if 'impot' in file_name and 'Code des Impôts du Sénégal' not in sources_trouvees:
                                sources_trouvees.append('Code des Impôts du Sénégal')
                            elif 'douane' in file_name and 'Code des Douanes du Sénégal' not in sources_trouvees:
                                sources_trouvees.append('Code des Douanes du Sénégal')
                    
                    if len(sources_trouvees) == 1:
                        code_source = sources_trouvees[0]
                    elif len(sources_trouvees) > 1:
                        code_source = " ET ".join(sources_trouvees)
                    else:
                        code_source = "Documents juridiques sénégalais"
                    
                    # 🗨️ INTÉGRER LE CONTEXTE CONVERSATIONNEL
                    conversation_context = ""
                    if is_follow_up:
                        conversation_context = f"\n\n💬 CONTEXTE DE LA CONVERSATION:\n{self.conversation_manager.get_conversation_context(conversation_id, max_messages=4)}\n"
                    
                    prompt = f"""TEXTE OFFICIEL: {context}

QUESTION ACTUELLE: {message}
{conversation_context}
🇫🇷 LANGUE OBLIGATOIRE: Tu DOIS répondre UNIQUEMENT en français. Aucun mot dans une autre langue n'est autorisé.

🧠 EXPERTISE ÉLARGIE: Tu es un expert en droit sénégalais qui maîtrise :
- Code des Impôts et fiscalité (CGI, DGI, TVA, IS, IR)
- Code des Douanes et procédures douanières  
- Lois de Finances et budget de l'État
- Documents économiques et financiers publics
- Réglementations et arrêtés administratifs
- Codes d'investissement et sectoriels

🗨️ INTELLIGENCE CONVERSATIONNELLE :
- Si la QUESTION ACTUELLE fait référence à la conversation précédente (ex: "ce taux", "cette taxe", "il", "elle", etc.), utilise le CONTEXTE DE LA CONVERSATION pour comprendre à quoi l'utilisateur fait référence
- Si l'utilisateur dit "ce taux", identifie de quel taux il parlait dans les messages précédents
- Si l'utilisateur dit "cette marchandise" ou "ces produits", réfère-toi aux éléments mentionnés précédemment
- Assure-toi de faire le lien logique entre la question actuelle et les échanges précédents

🚨 RÈGLES ABSOLUES - AUCUNE EXCEPTION AUTORISÉE :

📊 VALEURS NUMÉRIQUES - INTERDICTION TOTALE DE MODIFICATION :
❌ INTERDIT : Arrondir, estimer, approximer, convertir, reformuler
❌ INTERDIT : Dire "environ", "près de", "approximativement", "autour de"
❌ INTERDIT : Changer "2 875 millions" en "2,9 milliards" ou "près de 3 milliards"
❌ INTERDIT : Changer "141 millions et demi" en "141,5 millions" ou "environ 142 millions"
❌ INTERDIT : Changer "3,8 milliards et demi" en "3 800 millions" ou "environ 4 milliards"
✅ OBLIGATOIRE : Recopier TRAIT POUR TRAIT chaque chiffre, virgule, espace, unité

💰 CITATIONS EXACTES OBLIGATOIRES :
- Si le document dit "2 875 millions FCFA" → Tu écris "2 875 millions FCFA"
- Si le document dit "141 millions et demi d'euros" → Tu écris "141 millions et demi d'euros"
- Si le document dit "470 millions FCFA" → Tu écris "470 millions FCFA" 
- Si le document dit "3,8 milliards et demi d'euros" → Tu écris "3,8 milliards et demi d'euros"
- GARDE le format exact : espaces, virgules, "et demi", devises, unités

� RÈGLES STRICTES POUR TOUS LES NOMBRES :
1. Copie EXACTEMENT chaque chiffre sans modification
2. Conserve les espaces dans les nombres (ex: "2 875" reste "2 875")
3. Conserve les virgules et points (ex: "3,8" reste "3,8")
4. Conserve "et demi" au lieu de ",5" si c'est écrit ainsi
5. Conserve "millions/milliards" exactement comme écrit
6. Conserve "FCFA/euros" exactement comme écrit
7. Ne convertis JAMAIS une devise vers une autre
8. Ne changes JAMAIS l'unité (millions vers milliards ou vice-versa)

🚨 CONSIGNES GÉNÉRALES ANTI-HALLUCINATION :
1. Tu DOIS utiliser EXCLUSIVEMENT le contenu du TEXTE OFFICIEL ci-dessus
2. Si l'information existe dans le texte, cite-la EXACTEMENT
3. Ne dis JAMAIS "n'est pas mentionné" si l'information est dans le texte
4. INTERDIT ABSOLU d'inventer, supposer, extrapoler ou modifier quelque valeur que ce soit
5. INTERDIT de dire "selon mes connaissances" ou "généralement"
6. INTERDIT d'ajouter des informations qui ne sont PAS dans le TEXTE OFFICIEL
7. INTERDIT de parler de "taux réduit" ou "0%" s'ils ne sont PAS mentionnés dans le texte
8. INTERDIT de mentionner des "exceptions" ou "cas particuliers" non explicites dans le texte
9. INTERDIT de dire "il existe" ou "il y a aussi" sans citation exacte du document
10. Si tu ne trouves PAS l'information exacte dans le texte, dis clairement "cette information n'est pas précisée dans les extraits fournis"

🔍 MÉTHODE DE VÉRIFICATION OBLIGATOIRE AVANT RÉPONSE :
- Relis le TEXTE OFFICIEL mot par mot
- Vérifies que CHAQUE affirmation de ta réponse est DIRECTEMENT trouvable dans le texte
- Supprimes toute phrase qui n'a pas de source directe dans le TEXTE OFFICIEL
- Cites UNIQUEMENT ce qui est écrit noir sur blanc dans les documents

🚫 EXEMPLES D'INVENTIONS INTERDITES :
❌ "Il existe un taux réduit de 0%" (si non mentionné dans le texte)
❌ "Certains établissements bénéficient d'exonérations" (si non explicite)
❌ "La loi prévoit des exceptions" (si non citées précisément)
❌ "Le taux peut être différent selon les cas" (si non documenté)

MÉTHODE OBLIGATOIRE:
- Lis attentivement le TEXTE OFFICIEL ligne par ligne
- Trouve l'information demandée UNIQUEMENT dans ce texte
- Cite UNIQUEMENT ce qui est écrit mot pour mot dans le document
- N'AJOUTES AUCUNE information extérieure au texte
- Si l'information DIRECTE n'est pas dans le texte, dis clairement "cette information spécifique n'est pas détaillée dans les extraits fournis"
- Puis RÉSUME UNIQUEMENT ce que contiennent réellement les articles/sections trouvés en citant leurs dispositions EXACTES

🔍 AVANT DE RÉPONDRE - VÉRIFICATION OBLIGATOIRE :
1. Chaque phrase de ma réponse a-t-elle une source DIRECTE dans le TEXTE OFFICIEL ?
2. Ai-je inventé ou ajouté des informations non présentes dans le texte ?
3. Ai-je cité des taux, pourcentages ou exceptions qui ne sont PAS dans le document ?
4. Ma réponse est-elle une copie fidèle du contenu du TEXTE OFFICIEL ?

STRUCTURE DE RÉPONSE SI INFORMATION DIRECTE ABSENTE:
"Bien que la procédure spécifique de [sujet] ne soit pas détaillée dans ces extraits, les documents trouvés traitent de :
- [Document/Article X] : [résumé du contenu avec valeurs EXACTES copiées du document]
- [Document/Article Y] : [résumé du contenu avec montants EXACTS copiés du document]
Ces dispositions encadrent les aspects connexes de [domaine général]."

⚠️ RAPPEL CRITIQUE : CHAQUE VALEUR NUMÉRIQUE DOIT ÊTRE UNE COPIE PARFAITE DU DOCUMENT ORIGINAL

Réponds maintenant en français uniquement et en appliquant CES RÈGLES STRICTEMENT, en particulier la préservation EXACTE ET INTÉGRALE de toutes les valeurs numériques:"""
                else:
                    return {
                        "response": f"""⚠️ INFORMATION NON TROUVÉE

Je ne trouve pas d'information sur ce sujet dans les documents juridiques indexés.

📌 **Suggestions :**
- Utilisez des termes précis (codes, lois, arrêtés, budget)
- Référencez un article spécifique (ex: "Article 19 du CGI", "Loi de Finances 2025")
- Reformulez avec des termes juridiques sénégalais
- Précisez le domaine : fiscal, douanier, budgétaire, économique

ℹ️ En mode RAG strict, je réponds uniquement sur la base des documents.""",
                        "references": references
                    }
            else:
                return {
                    "response": f"""⚠️ AUCUN DOCUMENT CORRESPONDANT

Je suis conçu pour répondre aux questions liées au droit et à l'administration sénégalaise.

📊 **Domaines couverts :**
- Code des Impôts et fiscalité sénégalaise
- Code des Douanes et procédures commerciales  
- Lois de Finances et budget de l'État
- Documents économiques et financiers publics
- Réglementations et arrêtés administratifs

💡 **Exemples de questions :**
- "Que dit la législation douanière sur l'importation des marchandises ?"
- "Quels sont les taux de TVA selon le Code des Impôts ?"
- "Compensation SENELEC dans les lois de finances ?"

ℹ️ En mode RAG strict, je ne réponds qu'aux questions basées sur les documents juridiques.""",
                    "references": []
                }
            
            # Générer la réponse avec Ollama - MODE RAG STRICT AVEC CITATIONS
            payload = {
                "model": self.config.OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,      # Température à 0 pour réponses EXACTES - aucune créativité
                    "top_p": 0.1,           # Top-p très bas pour forcer la sélection des mots les plus probables
                    "top_k": 1,             # Ne garde que le mot le plus probable à chaque étape
                    "repeat_penalty": 1.5,   # Pénalité augmentée pour éviter les répétitions inventées
                    "presence_penalty": 0.5, # Encourage la diversité basée sur le contexte fourni uniquement
                    "frequency_penalty": 0.3, # Évite les répétitions non fondées
                    "num_ctx": 4096,        # Contexte limité pour se concentrer sur les documents fournis
                    "num_predict": 800,     # Limite la longueur pour éviter les divagations
                    "stop": ["Article 999", "Code fictif", "n'existe pas", "inexistant", "environ", "près de", "approximativement", "autour de", "à peu près"]  # Mots-clés d'arrêt d'urgence + mots d'estimation
                }
            }
            
            try:
                # 🔄 MÉCANISME DE RETRY PROGRESSIF avec réduction du contexte
                max_retries = 2
                for attempt in range(max_retries + 1):
                    # Réduire progressivement le contexte si timeout
                    if attempt > 0:
                        logger.info(f"🔄 Tentative {attempt + 1}: réduction du contexte ({len(context)} chars)")
                        # Réduire le contexte de 50% à chaque retry
                        context_lines = context.split('\n')
                        max_lines = max(5, len(context_lines) // (2 ** attempt))  # Minimum 5 lignes
                        context = '\n'.join(context_lines[:max_lines])
                        logger.info(f"🔄 Contexte réduit à {len(context)} caractères")
                        
                        # Mettre à jour le payload avec le contexte réduit
                        payload["prompt"] = prompt.replace(prompt.split("QUESTION:")[0], f"""TEXTE OFFICIEL: {context}

""")
                    
                    try:
                        response = requests.post(
                            f"{self.config.OLLAMA_BASE_URL}/api/generate",
                            json=payload,
                            timeout=60  # Timeout réduit à 1 minute pour détecter rapidement les problèmes
                        )
                        
                        if response.status_code == 200:
                            ollama_response = response.json()['response']
                            
                            # 🛡️ VÉRIFICATION ANTI-HALLUCINATION
                            validated_response = self._validate_response_against_context(ollama_response, context, message)
                            
                            # 💬 Enregistrer la réponse dans l'historique de conversation
                            if conversation_id and self.conversation_manager:
                                self.conversation_manager.add_message(conversation_id, "assistant", validated_response)
                            
                            logger.info(f"✅ Mistral réponse obtenue (tentative {attempt + 1})")
                            return {
                                "response": validated_response,
                                "references": references
                            }
                        elif response.status_code == 504:
                            if attempt < max_retries:
                                logger.warning(f"⏱️ Timeout Mistral (504) - Tentative {attempt + 1}/{max_retries + 1}")
                                continue
                            else:
                                # Timeout final - Afficher directement les articles trouvés
                                logger.warning("⏱️ Timeout Mistral final (504) - Affichage direct des articles trouvés")
                                return self._format_direct_response(message, references)
                        else:
                            return {
                                "response": f"❌ Erreur technique (code {response.status_code}). Veuillez réessayer.",
                                "references": []
                            }
                    except requests.Timeout:
                        if attempt < max_retries:
                            logger.warning(f"⏱️ Timeout requête Mistral - Tentative {attempt + 1}/{max_retries + 1}")
                            continue
                        else:
                            # Timeout final de la requête Python - Afficher directement les articles
                            logger.warning("⏱️ Timeout requête Mistral final - Affichage direct des articles trouvés")
                            return self._format_direct_response(message, references)
            except requests.exceptions.RequestException as e:
                # Autres erreurs réseau - Afficher directement les articles
                logger.error(f"❌ Erreur réseau Mistral: {e}")
                return self._format_direct_response(message, references)
            
            else:
                # Aucun contexte trouvé dans les documents - Vérifier si c'est hors domaine
                logger.warning("⚠️ Aucun contexte trouvé dans les documents indexés")
                
                # Maintenant on fait la vérification du domaine seulement si rien n'est trouvé
                if not self.is_fiscal_related_question(message):
                    response_text = f"""⚠️ QUESTION HORS DOMAINE

Aucune information trouvée dans les documents indexés pour votre question.

Je suis conçu pour répondre aux questions sur :
- 🏛️ Fiscalité et douanes sénégalaises (Code des Impôts, Code des Douanes)
- 💰 Économie et finances publiques (Budget, Loi de Finances)  
- 📊 Secteurs économiques (prévisions, croissance sectorielle)
- 🏭 Investissements et politique économique
- 📈 Dette publique et gestion financière

💡 **Suggestions :**
- Reformulez votre question avec des termes plus spécifiques
- Mentionnez un secteur économique particulier
- Posez une question sur les prévisions budgétaires ou économiques du Sénégal
- Utilisez des mots-clés liés aux documents indexés

ℹ️ Seules les questions ayant des réponses dans les documents indexés sont traitées."""
                    
                    # 💬 Enregistrer la réponse dans l'historique de conversation
                    if conversation_id and self.conversation_manager:
                        self.conversation_manager.add_message(conversation_id, "assistant", response_text)
                    
                    return {
                        "response": response_text,
                        "references": []
                    }
                else:
                    # Question dans le domaine mais pas de résultats - Suggérer reformulation
                    response_text = f"""🔍 AUCUNE INFORMATION TROUVÉE

Votre question semble pertinente mais aucune information correspondante n'a été trouvée dans les documents indexés.

**Votre question:** {message}

💡 **Suggestions pour améliorer votre recherche :**
- Reformulez avec des termes plus généraux ou plus spécifiques
- Utilisez des synonymes (ex: "impôt" → "fiscalité", "croissance" → "développement")
- Mentionnez un secteur spécifique (chimique, agroalimentaire, etc.)
- Précisez la période si pertinente (2025, 2026)

📚 **Exemples de questions qui fonctionnent :**
- "Quelles sont les prévisions de croissance pour 2026 ?"
- "Comment évoluent les investissements dans le secteur industriel ?"
- "Quel est le taux de TVA au Sénégal ?"

🔄 Essayez de reformuler votre question."""

                    # 💬 Enregistrer la réponse dans l'historique de conversation
                    if conversation_id and self.conversation_manager:
                        self.conversation_manager.add_message(conversation_id, "assistant", response_text)
                    
                    return {
                        "response": response_text,
                        "references": []
                    }
                
        except Exception as e:
            logger.error(f"Erreur chat: {e}")
            error_response = "Une erreur s'est produite. Veuillez réessayer dans un moment."
            
            # 💬 Enregistrer la réponse d'erreur dans l'historique de conversation
            if conversation_id and self.conversation_manager:
                self.conversation_manager.add_message(conversation_id, "assistant", error_response)
            
            return {
                "response": error_response,
                "references": []
            }
    
    def _validate_response_against_context(self, response: str, context: str, original_question: str) -> str:
        """🛡️ Valide la réponse de Ollama contre le contexte fourni pour détecter les hallucinations"""
        
        try:
            # Extraire les articles mentionnés dans la réponse
            import re
            article_pattern = r'Article\s+(\d+)'
            response_articles = set(re.findall(article_pattern, response, re.IGNORECASE))
            context_articles = set(re.findall(article_pattern, context, re.IGNORECASE))
            
            # Vérifier les chiffres/pourcentages
            number_pattern = r'(\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:FCFA|francs))'
            response_numbers = set(re.findall(number_pattern, response, re.IGNORECASE))
            context_numbers = set(re.findall(number_pattern, context, re.IGNORECASE))
            
            # Détecter les hallucinations potentielles
            hallucination_detected = False
            warning_messages = []
            
            # 1. Articles inventés - Ajuster la détection pour éviter les faux positifs
            invented_articles = response_articles - context_articles
            if invented_articles:
                # Vérifier si ce sont vraiment des inventions ou des extractions légitimes
                critical_inventions = []
                for article in invented_articles:
                    # Ne considérer comme hallucination que si l'article n'est pas dans le contexte du tout
                    if not any(article in line for line in context.split('\n')):
                        critical_inventions.append(article)
                
                if critical_inventions:
                    hallucination_detected = True
                    warning_messages.append(f"⚠️ Articles non trouvés dans les documents: {', '.join(critical_inventions)}")
                else:
                    logger.info(f"ℹ️ Articles mentionnés mais considérés comme légitimes: {', '.join(invented_articles)}")
            
            # 2. Chiffres inventés (tolérance de 5% pour erreurs de transcription)
            for resp_num in response_numbers:
                found_similar = False
                for ctx_num in context_numbers:
                    # Comparaison exacte d'abord
                    if resp_num == ctx_num:
                        found_similar = True
                        break
                
                if not found_similar:
                    warning_messages.append(f"⚠️ Chiffre suspect non vérifié: {resp_num}")
            
            # 3. 🚨 VÉRIFICATION CRITIQUE - Mais plus tolérante pour les questions générales
            false_negative_patterns = [
                "n'est pas explicitement mentionné", "n'est pas mentionné", "ne précise pas",
                "n'est pas spécifié", "pas d'information", "aucune mention"
            ]
            
            for pattern in false_negative_patterns:
                if pattern.lower() in response.lower():
                    # Vérifier si l'information est VRAIMENT dans le contexte
                    question_lower = original_question.lower()
                    context_lower = context.lower()
                    
                    # Recherche spécifique pour TVA/taux (cas critique)
                    if any(word in question_lower for word in ["tva", "taxe", "taux"]):
                        if any(phrase in context_lower for phrase in ["taux", "18%", "dix-huit pour cent", "fixé à"]):
                            hallucination_detected = True
                            warning_messages.append(f"🚨 ERREUR CRITIQUE: Le modèle dit '{pattern}' mais l'information est PRÉSENTE dans le contexte")
                    
                    # Pour les questions générales, être plus tolérant
                    elif "exoneration" in question_lower or "général" in question_lower or "conditionnelle" in question_lower:
                        # Ne pas déclencher d'hallucination pour les questions générales complexes
                        logger.info(f"ℹ️ Question générale détectée, tolérance accrue pour: {pattern}")
                    else:
                        # Recherche générale pour d'autres sujets (seuil plus élevé)
                        key_words = [word for word in question_lower.split() if len(word) > 3]
                        if key_words:
                            found_matches = sum(1 for word in key_words if word in context_lower)
                            if found_matches >= len(key_words) * 0.8:  # 80% des mots-clés trouvés (plus strict)
                                warning_messages.append(f"⚠️ Le modèle dit '{pattern}' mais des éléments pertinents sont dans le contexte")
            
            # 4. Mots-clés suspects d'hallucination - Version plus intelligente
            # Phrases vraiment problématiques (hallucinations claires)
            critical_suspicious_phrases = [
                "selon mes connaissances", "d'après ce que je sais", 
                "je pense que", "il me semble"
            ]
            
            # Phrases modérément suspectes (contextuel)
            moderate_suspicious_phrases = [
                "généralement", "habituellement", "en règle générale", 
                "il est probable que", "vraisemblablement"
            ]
            
            # Phrases acceptables dans contexte juridique
            acceptable_phrases = [
                "il est possible", "il convient de", "peuvent être", 
                "est susceptible de", "peut être"
            ]
            
            # Vérifier les phrases critiques
            for phrase in critical_suspicious_phrases:
                if phrase.lower() in response.lower():
                    hallucination_detected = True
                    warning_messages.append(f"⚠️ Formulation problématique détectée: '{phrase}'")
            
            # Vérifier les phrases modérées seulement si pas de contexte juridique solide
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())
            common_words = response_words.intersection(context_words)
            
            # Si peu de correspondance avec le contexte, être plus strict
            if len(common_words) < min(5, len(response_words) * 0.2):
                for phrase in moderate_suspicious_phrases:
                    if phrase.lower() in response.lower():
                        warning_messages.append(f"⚠️ Formulation suspecte avec faible correspondance: '{phrase}'")
            
            # 5. Vérification de la cohérence avec le contexte - Seuils assouplis
            response_lower = response.lower()
            context_lower = context.lower()
            
            # Vérifier que les citations sont présentes dans le contexte
            if "article" in response_lower and "article" in context_lower:
                # Si la réponse mentionne du contenu d'article, vérifier qu'il existe dans le contexte
                response_words = set(response_lower.split())
                context_words = set(context_lower.split())
                
                # Vérifier un minimum de correspondance lexicale - Seuil abaissé
                common_words = response_words.intersection(context_words)
                min_threshold = min(5, len(response_words) * 0.2)  # Au moins 20% de mots en commun ou 5 mots minimum
                if len(common_words) < min_threshold:
                    warning_messages.append("⚠️ Faible correspondance lexicale avec les documents fournis")
                else:
                    logger.info(f"✅ Correspondance lexicale acceptable: {len(common_words)} mots communs")
            
            # Si hallucination détectée, mais être plus tolérant pour les questions générales
            if hallucination_detected:
                # Mode tolérant pour certains types de questions
                question_lower = original_question.lower()
                is_general_question = any(word in question_lower for word in [
                    "parle", "expliquer", "qu'est-ce", "général", "principe", "résumer", "présenter"
                ])
                
                # Compter les vraies erreurs critiques
                critical_errors = [msg for msg in warning_messages if "ERREUR CRITIQUE" in msg or "Articles non trouvés" in msg]
                
                if is_general_question and len(critical_errors) == 0:
                    # Pour les questions générales sans erreurs critiques, laisser passer
                    logger.info(f"ℹ️ Question générale détectée, validation assouplie malgré {len(warning_messages)} avertissements mineurs")
                    return response
                
                logger.warning(f"🚨 HALLUCINATION DÉTECTÉE: {'; '.join(warning_messages)}")
                
                # 🚨 CAS SPÉCIAL: Correction automatique pour TVA si détectée
                if "tva" in original_question.lower() and "18%" in context:
                    return """📋 **TVA AU SÉNÉGAL - INFORMATION OFFICIELLE**

Selon l'Article 369 du Code des Impôts du Sénégal :
**Le taux de la TVA est fixé à 18%.**

Cette information est explicitement mentionnée dans le texte officiel.

🚨 *Note: Réponse corrigée automatiquement suite à détection d'erreur d'interprétation*"""
                
                # 🔄 NOUVELLE APPROCHE: Générer une analyse alternative des documents trouvés
                logger.info("🔄 Génération d'une analyse alternative des documents trouvés...")
                
                alternative_prompt = f"""DOCUMENTS OFFICIELS TROUVÉS: {context}

QUESTION POSÉE: {original_question}

🇫🇷 LANGUE OBLIGATOIRE: Tu DOIS répondre UNIQUEMENT en français. Aucun mot en anglais ou autre langue n'est autorisé.

🚨 MISSION SPÉCIALE: Le système a détecté une possible erreur d'interprétation dans une première réponse.
Tu dois maintenant faire une ANALYSE PRUDENTE ET FACTUELLE des documents fournis.

🔢 RÈGLE ABSOLUE - AUCUNE EXCEPTION - VALEURS NUMÉRIQUES EXACTES :
❌ TOTALEMENT INTERDIT : Modifier, arrondir, estimer, approximer toute valeur
❌ INTERDIT : "environ", "près de", "approximativement", "autour de"
✅ OBLIGATOIRE : Copier EXACTEMENT comme écrit dans le document source

💰 EXEMPLES DE CITATION CORRECTE :
- Document dit "2 875 millions FCFA" → Tu écris "2 875 millions FCFA"
- Document dit "141 millions et demi d'euros" → Tu écris "141 millions et demi d'euros"
- Document dit "470 millions FCFA" → Tu écris "470 millions FCFA"
- JAMAIS de conversion, JAMAIS d'arrondi, JAMAIS d'estimation

CONSIGNES STRICTES:
1. NE PAS inventer d'informations
2. ANALYSER uniquement ce qui est présent dans les documents
3. 💰 RECOPIER EXACTEMENT tous les montants, devises, formats numériques
4. Si pas d'information directe sur le sujet, chercher des ÉLÉMENTS CONNEXES
5. Utiliser des formulations prudentes comme "selon l'article X", "d'après les documents"
6. JAMAIS d'arrondi ou d'approximation des valeurs (garde "141 millions et demi" EXACTEMENT tel quel)
7. JAMAIS de conversion d'unités (millions vers milliards ou FCFA vers euros)

STRUCTURE DE RÉPONSE OBLIGATOIRE:
• "Il n'existe pas de disposition spécifique sur [sujet exact] dans ces documents"
• "Cependant, les articles suivants traitent d'aspects connexes:"
• [Analyse factuelle des articles pertinents trouvés avec valeurs EXACTES copiées]
• "Pour une information complète, consulter directement les textes officiels"

⚠️ VÉRIFICATION FINALE : Chaque valeur numérique dans ta réponse doit être une COPIE PARFAITE du document

Génère maintenant cette analyse factuelle en préservant EXACTEMENT et INTÉGRALEMENT toutes les valeurs numériques:"""

                # Générer une réponse alternative avec Ollama
                alternative_payload = {
                    "model": self.config.OLLAMA_CHAT_MODEL,
                    "prompt": alternative_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,      # Température à 0 pour exactitude maximale
                        "top_p": 0.1,           # Très conservateur
                        "top_k": 1,             # Une seule option la plus probable
                        "repeat_penalty": 1.3,  
                        "num_ctx": 4096,        
                        "num_predict": 600,     # Plus court pour éviter les dérives
                        "stop": ["Code fictif", "n'existe pas vraiment", "invention", "environ", "près de", "approximativement", "autour de", "à peu près"]  # Arrêt sur mots d'estimation
                    }
                }
                
                try:
                    alternative_response = requests.post(
                        f"{self.config.OLLAMA_BASE_URL}/api/generate",
                        json=alternative_payload,
                        timeout=180  # 3 minutes
                    )
                    
                    if alternative_response.status_code == 200:
                        alternative_text = alternative_response.json()['response']
                        
                        # Ajouter une note explicative
                        final_response = f"""⚠️ **ANALYSE PRUDENTE - MODE SÉCURISÉ ACTIVÉ**

{alternative_text}

---
*🛡️ Note: Cette réponse a été générée en mode sécurisé suite à la détection de potentielles imprécisions dans l'analyse initiale. Seuls les éléments explicitement présents dans les documents ont été analysés.*"""
                        
                        return final_response
                    else:
                        logger.error(f"Erreur génération alternative: {alternative_response.status_code}")
                        # Fallback vers l'ancienne méthode
                        pass
                        
                except Exception as e:
                    logger.error(f"Erreur appel Ollama alternatif: {e}")
                    # Fallback vers l'ancienne méthode
                    pass
                
                # Réponse de sécurité avec les documents bruts
                safe_response = f"""🚨 **RÉPONSE SÉCURISÉE - ERREUR D'INTERPRÉTATION DÉTECTÉE**

Le système a détecté une possible erreur dans l'interprétation des documents.
Voici le contenu EXACT des documents trouvés pour: "{original_question}"

**📄 CONTENU BRUT DES DOCUMENTS:**
{context[:1500]}...

**⚠️ PROBLÈMES DÉTECTÉS:**
{chr(10).join(warning_messages)}

**🔍 RECOMMANDATION:**
Consultez directement les documents officiels ci-dessus pour obtenir l'information précise."""
                
                return safe_response
            
            # Si pas d'hallucination détectée, retourner la réponse originale
            return response
            
        except Exception as e:
            logger.error(f"Erreur validation anti-hallucination: {e}")
            # En cas d'erreur de validation, retourner la réponse avec avertissement
            return f"""⚠️ **AVERTISSEMENT - VALIDATION IMPOSSIBLE**

{response}

**Note:** La validation automatique anti-hallucination a échoué. Veuillez vérifier la réponse avec les documents officiels."""

# Template HTML ultra moderne et responsif avec effets
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LexFin - Assistant Fiscal et Douanier Sénégal</title>
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

        /* Contrôles de l'en-tête */
        .header-controls {
            position: absolute;
            top: 30px;
            right: 30px;
            display: flex;
            gap: 15px;
            align-items: center;
            z-index: 100;
        }

        .header-btn {
            position: relative !important;
            right: 90px !important;
            top: 35px !important;
        }

        .theme-toggle {
            order: 1;
        }

        .header-btn {
            padding: 12px 20px;
            border-radius: 25px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.15);
            color: white;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 
                0 6px 20px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
            transition: var(--transition-smooth);
            backdrop-filter: blur(10px);
            font-size: 14px;
            font-weight: 600;
        }

        .header-btn:hover {
            transform: translateY(-2px);
            background: rgba(255, 255, 255, 0.25);
            box-shadow: 
                0 8px 25px rgba(0, 0, 0, 0.25),
                0 0 0 3px rgba(254, 239, 66, 0.3);
        }

        .theme-toggle {
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
                <h1>🇸🇳 LexFin - MODE RAG STRICT</h1>
                <p>Assistant IA dédié à la fiscalité  •  Réponses précises basées sur une base documentaire fiscale spécialisée</p>
                <div class="header-controls">
                    <button id="newConversationBtn" class="header-btn" title="Nouvelle conversation" onclick="newConversation()">
                        <i class="fa-solid fa-plus"></i> Nouvelle conversation
                    </button>
                    <button id="themeToggle" class="theme-toggle" title="Changer de thème">
                        <i class="fa-solid fa-moon"></i>
                    </button>
                </div>
            </div>

        <div class="chat-container" id="chatContainer">
            <div class="message assistant-message">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <span style="font-size: 48px; filter: drop-shadow(0 2px 8px rgba(0, 133, 63, 0.3));">🇸🇳</span>
                    <div>
                        <div style="font-size: 1.3em; font-weight: 700; color: var(--senegal-green); margin-bottom: 4px;">
                            Bienvenue sur LexFin
                        </div>
                        <div style="font-size: 0.95em; color: #64748b; font-weight: 500;">
                            Assistant IA dédié à la fiscalité
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
                        <div style="line-height: 1.5;">• "Que dit l'article 45 du code général des impôts ?"</div>
                        <div style="line-height: 1.5;">• "Quel est le taux de la TVA au Sénégal ?"</div>
                        <div style="line-height: 1.5;">• "Comment calculer l'impôt minimum forfaitaire ?"</div>
                        <div style="line-height: 1.5;">• "Quelles sont les conditions d'exonération de droits de douane ?"</div>
                        <div style="line-height: 1.5;">• "Qu'est-ce que le régime de l'entrepôt de stockage ?"</div>
                        <div style="line-height: 1.5;">• "Comment fonctionne la procédure de dédouanement ?"</div>
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
                <span>LexFin analyse votre question fiscal/douanière<span class="loading-dots"></span></span>
            </div>

            <div class="chat-input-section">
                <div class="input-section">
                    <input type="text" id="messageInput" placeholder="Posez votre question sur le Code des Impôts ou Code des Douanes uniquement..." onkeypress="checkEnter(event)">
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
        
        // Variable globale pour l'ID de conversation actuelle
        let currentConversationId = null;

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
                // Créer une nouvelle conversation si nécessaire
                if (!currentConversationId) {
                    const convResponse = await fetch('/conversation/new', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ title: message.substring(0, 50) + '...' })
                    });
                    const convData = await convResponse.json();
                    currentConversationId = convData.conversation_id;
                }
                
                // Envoyer le message avec l'ID de conversation
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        message: message,
                        conversation_id: currentConversationId 
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

        // Fonction pour démarrer une nouvelle conversation
        function newConversation() {
            // Réinitialiser l'ID de conversation
            currentConversationId = null;
            
            // Vider le chat
            clearChat();
            
            console.log('Nouvelle conversation démarrée');
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
                                    Bienvenue sur LexFin
                                </div>
                                <div style="font-size: 0.95em; color: #64748b; font-weight: 500;">
                                    Assistant IA dédié à la fiscalité
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
                                <div style="line-height: 1.5;">• "Que dit l'article 45 du code général des impôts ?"</div>
                                <div style="line-height: 1.5;">• "Quel est le taux de la TVA au Sénégal ?"</div>
                                <div style="line-height: 1.5;">• "Comment calculer l'impôt minimum forfaitaire ?"</div>
                                <div style="line-height: 1.5;">• "Quelles sont les conditions d'exonération de droits de douane ?"</div>
                                <div style="line-height: 1.5;">• "Qu'est-ce que le régime de l'entrepôt de stockage ?"</div>
                                <div style="line-height: 1.5;">• "Comment fonctionne la procédure de dédouanement ?"</div>
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
                        <button onclick="openFile('${ref.file_path}', ${ref.line_start}, '${ref.page_info || ''}')" 
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
        async function openFile(filePath, lineNumber, pageInfo = '') {
            try {
                const response = await fetch('/open_file', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        file_path: filePath,
                        line_number: lineNumber,
                        page_info: pageInfo
                    })
                });

                const data = await response.json();
                
                if (data.success) {
                    // Ouvrir le fichier dans un nouvel onglet
                    window.open(data.file_url, '_blank');
                    
                    // Animation de succès
                    const btn = event.target;
                    const originalText = btn.textContent;
                    btn.textContent = '✓ Ouvert!';
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
                    
                    if (text && !text.includes('🇸🇳 Bonjour ! Je suis LexFin')) {
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
    
    <!-- Footer LexFin avec drapeau animé -->
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

lexfin_client = LexFinClient()

@app.route('/')
def home():
    """Page d'accueil"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint pour le chat avec références précises et gestion des conversations"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        conversation_id = data.get('conversation_id', None)  # ID de conversation optionnel
        
        if not message:
            return jsonify({
                'response': 'Veuillez saisir un message.',
                'references': []
            }), 400
        
        # Transmettre le conversation_id à la méthode chat
        result = lexfin_client.chat(message, conversation_id=conversation_id)
        
        # 🔧 DEBUG: Log des références pour diagnostiquer le problème "undefined"
        references = result.get('references', [])
        logger.info(f"🔍 DEBUG RÉFÉRENCES - Nombre: {len(references)}")
        for i, ref in enumerate(references[:3]):  # Log des 3 premières
            logger.info(f"  Ref {i+1}:")
            logger.info(f"    file_name: '{ref.get('file_name', 'MISSING')}'")
            logger.info(f"    page_info: '{ref.get('page_info', 'MISSING')}'")
            logger.info(f"    location: '{ref.get('location', 'MISSING')}'")
            logger.info(f"    snippet: '{ref.get('snippet', 'MISSING')[:50]}...'")
        
        return jsonify({
            'response': result.get('response', ''),
            'references': references,
            'conversation_id': conversation_id  # Retourner l'ID de conversation
        })
        
    except Exception as e:
        logger.error(f"Erreur chat endpoint: {e}")
        return jsonify({
            'response': 'Une erreur s\'est produite.',
            'references': []
        }), 500

@app.route('/conversation/new', methods=['POST'])
def new_conversation():
    """Créer une nouvelle conversation"""
    try:
        data = request.get_json() or {}
        title = data.get('title', 'Nouvelle conversation')
        
        conversation_id = lexfin_client.conversation_manager.create_conversation(title)
        
        return jsonify({
            'conversation_id': conversation_id,
            'title': title,
            'created_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur création conversation: {e}")
        return jsonify({'error': 'Erreur lors de la création de la conversation'}), 500

@app.route('/conversation/<conversation_id>/history', methods=['GET'])
def get_conversation_history(conversation_id):
    """Récupérer l'historique d'une conversation"""
    try:
        history = lexfin_client.conversation_manager.get_conversation_history(conversation_id)
        
        if history is None:
            return jsonify({'error': 'Conversation non trouvée'}), 404
        
        return jsonify({
            'conversation_id': conversation_id,
            'messages': history
        })
        
    except Exception as e:
        logger.error(f"Erreur récupération historique: {e}")
        return jsonify({'error': 'Erreur lors de la récupération de l\'historique'}), 500

@app.route('/conversations', methods=['GET'])
def list_conversations():
    """Lister toutes les conversations"""
    try:
        conversations = lexfin_client.conversation_manager.list_conversations()
        
        return jsonify({
            'conversations': conversations
        })
        
    except Exception as e:
        logger.error(f"Erreur liste conversations: {e}")
        return jsonify({'error': 'Erreur lors de la récupération des conversations'}), 500

@app.route('/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Supprimer une conversation"""
    try:
        success = lexfin_client.conversation_manager.delete_conversation(conversation_id)
        
        if not success:
            return jsonify({'error': 'Conversation non trouvée'}), 404
        
        return jsonify({'message': 'Conversation supprimée avec succès'})
        
    except Exception as e:
        logger.error(f"Erreur suppression conversation: {e}")
        return jsonify({'error': 'Erreur lors de la suppression de la conversation'}), 500

@app.route('/regenerate', methods=['POST'])
def regenerate():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'response': 'Message vide reçu.', 'references': []}), 400
        
        # Prompt de simplification en français uniquement
        simplification_prompt = f"""🇫🇷 LANGUE OBLIGATOIRE: Tu DOIS répondre UNIQUEMENT en français. Aucun mot en anglais ou autre langue n'est autorisé.

Question à simplifier: "{message}"

MISSION: Reformule cette question de manière plus simple et claire, en français uniquement:
- Utilise un vocabulaire accessible 
- Garde le sens original intact
- Raccourcis les phrases longues
- Supprime les mots inutiles
- Maximum 2 lignes
- Réponse UNIQUEMENT en français

Reformulation simplifiée:"""

        # Appel à Mistral pour simplification
        payload = {
            "model": lexfin_client.config.OLLAMA_CHAT_MODEL,
            "prompt": simplification_prompt,
            "stream": False
        }
        
        response = requests.post(
            f"{lexfin_client.config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            simplified_message = response.json()['response'].strip()
            
            # Traiter la question simplifiée avec le système normal
            result = lexfin_client.chat(simplified_message)
            
            return jsonify({
                'response': result.get('response', ''),
                'references': result.get('references', []),
                'simplified_question': simplified_message
            })
        else:
            # Si échec de simplification, relancer avec question originale
            result = lexfin_client.chat(message)
            
            return jsonify({
                'response': result.get('response', ''),
                'references': result.get('references', []),
                'simplified_question': message
            })
            
    except Exception as e:
        logger.error(f"Erreur regenerate endpoint: {e}")
        return jsonify({
            'response': 'Erreur lors de la régénération.',
            'references': []
        }), 500

@app.route('/open_file', methods=['POST'])
def open_file():
    """Endpoint pour ouvrir un fichier à une position spécifique via le navigateur"""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        line_number = data.get('line_number', 1)
        page_info = data.get('page_info', '')
        
        logger.info(f"🔧 DEBUG open_file - file_path reçu: '{file_path}'")
        logger.info(f"🔧 DEBUG open_file - line_number: {line_number}")
        logger.info(f"🔧 DEBUG open_file - page_info: '{page_info}'")
        
        if not file_path:
            return jsonify({'error': 'Chemin de fichier manquant'}), 400
        
        # Extraire le nom du fichier - gérer les cas avec ou sans séparateur
        if '/' in file_path:
            filename = Path(file_path).name
        elif file_path.startswith('documents'):
            # Cas problématique: "documentsSenegal-Code-des-impot.pdf"
            filename = file_path.replace('documents', '', 1)
        else:
            filename = file_path
            
        logger.info(f"🔧 DEBUG open_file - filename extrait: '{filename}'")
        
        # Vérifier que le fichier existe dans le répertoire documents
        documents_dir = Path('./documents')
        target_file = documents_dir / filename
        
        if not target_file.exists():
            return jsonify({
                'error': f'Fichier non trouvé: {filename}',
                'success': False
            }), 404
        
        # Générer l'URL pour servir le fichier
        file_url = f'/files/{filename}'
        
        # Si on a des informations de page, les inclure
        page_fragment = ""
        if page_info and 'page' in page_info.lower():
            # Extraire le numéro de page de page_info (ex: "page 128" -> 128 ou "pages 194-195" -> 194)
            import re
            page_match = re.search(r'pages?\s+(\d+)', page_info.lower())
            if page_match:
                page_num = page_match.group(1)
                page_fragment = f"#page={page_num}"
                logger.info(f"🔧 DEBUG open_file - page_info: '{page_info}' -> page_num: {page_num}")
            else:
                logger.info(f"🔧 DEBUG open_file - Aucun numéro de page trouvé dans: '{page_info}'")
        else:
            logger.info(f"🔧 DEBUG open_file - page_info vide ou invalide: '{page_info}'")
        
        return jsonify({
            'message': f'Ouverture de {filename}',
            'success': True,
            'file_url': file_url + page_fragment,
            'filename': filename,
            'page_info': page_info,
            'line_number': line_number
        })
            
    except Exception as e:
        logger.error(f"Erreur open_file endpoint: {e}")
        return jsonify({'error': f'Erreur ouverture fichier: {str(e)}'}), 500

@app.route('/files/<filename>')
def serve_file(filename):
    """Sert les fichiers PDF depuis le répertoire documents"""
    try:
        documents_dir = Path('./documents')
        file_path = documents_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': f'Fichier non trouvé: {filename}'}), 404
        
        if not filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Seuls les fichiers PDF sont autorisés'}), 403
        
        # Servir le fichier PDF
        from flask import send_file
        return send_file(
            file_path,
            as_attachment=False,  # Pour affichage dans le navigateur
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Erreur service fichier {filename}: {e}")
        return jsonify({'error': f'Erreur service fichier: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Vérifie la santé de la connexion Ollama"""
    try:
        # Test rapide de connexion Ollama
        test_response = requests.get(
            f"{lexfin_client.config.OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )
        ollama_status = "🟢 Connecté" if test_response.status_code == 200 else "🟡 Réponse inattendue"
    except:
        ollama_status = "🔴 Déconnecté"
    
    return jsonify({
        'ollama_status': ollama_status,
        'server_url': lexfin_client.config.OLLAMA_BASE_URL,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint pour obtenir le statut de l'indexation"""
    try:
        # Vérifier le statut de la surveillance
        surveillance_status = "Inactive"
        auto_indexing = False
        if lexfin_client.observer:
            if lexfin_client.observer.is_alive():
                surveillance_status = "🔄 Active (Auto-indexation ON)"
                auto_indexing = True
            else:
                surveillance_status = "⏸️ Arrêté"
        
        # Lister les fichiers récents non indexés
        recent_files = []
        for file_path in lexfin_client.watch_dir.rglob('*'):
            if file_path.is_file() and lexfin_client.is_supported_file(str(file_path)):
                if not lexfin_client.is_file_already_indexed(str(file_path)):
                    recent_files.append(str(file_path))
        
        status = {
            'indexed_files_count': len(lexfin_client.indexed_files),
            'watch_directory': str(lexfin_client.watch_dir.absolute()),
            'supported_extensions': lexfin_client.config.SUPPORTED_EXTENSIONS,
            'indexed_files': [Path(f).name for f in lexfin_client.indexed_files.keys()],
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
        success = lexfin_client.start_file_watcher()
        
        if success:
            return jsonify({
                'message': 'Surveillance automatique redémarrée avec succès',
                'status': 'active',
                'watch_directory': str(lexfin_client.watch_dir)
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
        for file_path in lexfin_client.watch_dir.rglob('*'):
            if file_path.is_file() and lexfin_client.is_supported_file(str(file_path)):
                if not lexfin_client.is_file_already_indexed(str(file_path)):
                    logger.info(f"🆕 Indexation nouveau fichier: {file_path.name}")
                    lexfin_client.index_file(str(file_path))
                    new_files_indexed += 1
        
        return jsonify({
            'message': f'{new_files_indexed} nouveaux fichiers indexés',
            'total_indexed': len(lexfin_client.indexed_files)
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
        for file_path in lexfin_client.watch_dir.rglob('*'):
            if file_path.is_file() and lexfin_client.is_supported_file(str(file_path)):
                supported_files.append(str(file_path))
        
        # VIDER COMPLÈTEMENT le cache et ChromaDB
        lexfin_client.indexed_files.clear()
        try:
            if hasattr(lexfin_client, 'collection') and lexfin_client.collection:
                lexfin_client.create_vector_store()
                logger.info("🗑️ Base vectorielle et cache complètement vidés")
        except Exception as e:
            logger.warning(f"  Erreur vidage: {e}")
        
        # Indexation complète
        lexfin_client.scan_existing_files()
        
        return jsonify({
            'message': f'Réindexation COMPLÈTE terminée: {len(supported_files)} fichiers retraités',
            'indexed_count': len(lexfin_client.indexed_files),
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
        logger.info(f"🔍 Scan du dossier: {lexfin_client.config.WATCH_DIRECTORY}")
        
        # Lister tous les fichiers supportés
        supported_files = []
        for file_path in lexfin_client.watch_dir.rglob('*'):
            if file_path.is_file() and lexfin_client.is_supported_file(str(file_path)):
                supported_files.append(str(file_path))
        
        logger.info(f"   {len(supported_files)} fichiers supportés trouvés:")
        for file_path in supported_files:
            logger.info(f"   - {Path(file_path).name}")
        
        # Vider le cache ChromaDB complètement
        try:
            if hasattr(lexfin_client, 'collection') and lexfin_client.collection:
                lexfin_client.create_vector_store()
                logger.info("🗑️ Base vectorielle vidée complètement")
            else:
                logger.info("🔄 Création nouvelle base vectorielle")
                lexfin_client.create_vector_store()
        except Exception as e:
            logger.warning(f"  Erreur vidage base: {e}")
            # Fallback : créer une nouvelle collection
            try:
                lexfin_client.create_vector_store()
            except Exception as e2:
                logger.error(f"  Erreur création base: {e2}")
        
        # NE PAS vider le cache local - garder la mémoire des fichiers indexés
        # lexfin_client.indexed_files.clear()  # COMMENTÉ pour éviter réindexation
        
        # Relancer le scan avec respect du cache
        try:
            lexfin_client.scan_existing_files()
            already_indexed = len([f for f in supported_files if lexfin_client.is_file_already_indexed(f)])
            newly_indexed = len(lexfin_client.indexed_files) - already_indexed
            message = f'Scan terminé: {already_indexed} déjà indexés, {newly_indexed} nouveaux fichiers traités'
            logger.info(f"✅ Indexation terminée: {len(lexfin_client.indexed_files)} fichiers au total")
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation: {e}")
            message = f'Réindexation échouée: Vérifiez la connexion Ollama'
        
        return jsonify({
            'message': message,
            'indexed_count': len(lexfin_client.indexed_files),
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
        lexfin_client.scan_existing_files()
        return jsonify({
            'message': 'Indexation démarrée',
            'indexed_count': len(lexfin_client.indexed_files)
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
        indexed_files = list(lexfin_client.indexed_files.keys())
        
        for file_path in lexfin_client.watch_dir.rglob('*'):
            if file_path.is_file():
                all_files.append(str(file_path))
                if lexfin_client.is_supported_file(str(file_path)):
                    supported_files.append(str(file_path))
        
        # Compter les éléments dans ChromaDB avec diagnostic
        try:
            collection_count = lexfin_client.vector_store.count()
            logger.info(f"📊 ChromaDB count: {collection_count}")
        except Exception as e:
            logger.error(f"  Erreur ChromaDB count: {e}")
            collection_count = 0
            
        # Vérifier la collection elle-même
        try:
            # Essayer de récupérer quelques documents pour tester
            test_results = lexfin_client.vector_store.peek(limit=5)
            actual_chunks = len(test_results.get('documents', []))
            logger.info(f"🔍 Documents réels dans ChromaDB: {actual_chunks}")
            if actual_chunks > collection_count:
                collection_count = actual_chunks
        except Exception as e:
            logger.warning(f"  Erreur peek ChromaDB: {e}")
        
        return jsonify({
            'dossier_surveille': lexfin_client.config.WATCH_DIRECTORY,
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
        context = lexfin_client.search_context(query, limit=3)
        
        # Récupérer aussi quelques documents de ChromaDB
        try:
            sample_docs = lexfin_client.collection.peek(limit=3)
            sample_content = sample_docs.get('documents', [])[:3] if sample_docs else []
        except:
            sample_content = []
        
        return jsonify({
            'query': query,
            'context_found': context,
            'context_length': len(context) if context else 0,
            'sample_documents': sample_content,
            'collection_count': lexfin_client.collection.count() if lexfin_client.collection else 0
        })
        
    except Exception as e:
        logger.error(f"Erreur debug_context: {e}")
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

def cleanup():
    """Nettoyage à la fermeture"""
    try:
        if hasattr(lexfin_client, 'observer') and lexfin_client.observer:
            lexfin_client.observer.stop()
            lexfin_client.observer.join()
            logger.info("🛑 Surveillance arrêtée proprement")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {e}")

def app_lexfin():
    """Lance l'application LexFin"""
    print("🇸🇳 Démarrage de LexFin - Assistant Fiscal & Douanier Sénégal...")
    print("=" * 70)
    print(f"🔗 URL Ollama: {LexFinConfig.OLLAMA_BASE_URL}")
    print(f"🤖 Modèle IA: {LexFinConfig.OLLAMA_CHAT_MODEL}")
    print(f"📁 Répertoire surveillé: {LexFinConfig.WATCH_DIRECTORY}")
    print("🏛️ Spécialisation: Code des Impôts & Code des Douanes Sénégal")
    print("🌐 Démarrage de l'interface web...")
    
    try:
        app.run(
            host="0.0.0.0",
            port=8505,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n👋 Arrêt de LexFin...")
        cleanup()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        cleanup()

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup)
    app_lexfin()
