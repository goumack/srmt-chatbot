#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYSTÈME HIÉRARCHIQUE JURIDIQUE COMPLET V2.0
Gestion avancée de : Sections, Sous-sections, Articles avec contexte hiérarchique
Exemple: "que dit l'article 7 du benefices imposables du determination du benefice net imposable"
"""

import sys
import os
import re
from typing import Dict, List, Tuple, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('hierarchie_module')

class HierarchieJuridiqueClient:
    """Client avancé pour la recherche hiérarchique juridique complète"""
    
    def __init__(self, base_client=None):
        """Initialise le client avec système hiérarchique
        
        Args:
            base_client: Instance du client SRMT de base (SrmtDocumindClient)
        """
        if base_client is None:
            # Import conditionnel pour éviter l'import circulaire
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                import importlib.util
                spec = importlib.util.spec_from_file_location("srmt_module", "boutton memoire nouveau .py")
                srmt_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(srmt_module)
                self.base_client = srmt_module.SrmtDocumindClient()
                logger.info("✅ Client SRMT initialisé automatiquement")
            except Exception as e:
                logger.error(f"❌ Erreur initialisation client SRMT: {e}")
                raise
        else:
            self.base_client = base_client
            logger.info("✅ Client SRMT fourni en paramètre")
        
        # Patterns pour la détection hiérarchique
        self.patterns_hierarchie = {
            'sections': [
                r'section\s+([ivx]+)',           # Section I, II, III (chiffres romains)
                r'section\s+(\d+)',              # Section 1, 2, 3 (chiffres arabes)
                r'section\s+([a-z])',            # Section A, B, C (lettres)
            ],
            'sous_sections': [
                r'sous[- ]?section\s+(\d+)',     # Sous-section 1, 2, 3
                r'sous[- ]?section\s+([a-z])',   # Sous-section A, B, C
                r'sous[- ]?section\s+([ivx]+)',  # Sous-section I, II, III
            ],
            'articles': [
                r'article\s+(\d+)',              # Article 7, 15, etc.
            ],
            'contextes': [
                r'(bénéfices?\s+imposables?)',
                r'(détermination\s+(?:du\s+)?bénéfice\s+net\s+imposable)',
                r'(personnes?\s+imposables?)',
                r'(période\s+d\'imposition)',
                r'(rémunérations?)',
                r'(champ\s+d\'application)',
                r'(sociétés?)',
                r'(code\s+des?\s+douanes?)',
                r'(activités?\s+financières?)',
            ]
        }
    
    def rechercher_hierarchique(self, query: str) -> Dict:
        """Point d'entrée principal pour la recherche hiérarchique"""
        return self.recherche_hierarchique_complete(query)
    
    def recherche_hierarchique_complete(self, query: str) -> Dict:
        """
        Recherche hiérarchique complète avec détection de :
        - Sections (I, II, III, 1, 2, 3, A, B, C)
        - Sous-sections (1, 2, A, B, I, II)
        - Articles (numéros)
        - Contextes thématiques
        """
        logger.info(f"🔍 RECHERCHE HIÉRARCHIQUE: {query}")
        
        # Étape 1: Analyse hiérarchique de la requête
        analyse = self._analyser_requete_hierarchique(query)
        logger.info(f"📋 Analyse: {analyse}")
        
        # Étape 2: Recherche adaptée selon les éléments détectés
        if analyse['articles']:
            # Recherche d'articles avec contexte hiérarchique
            return self._recherche_articles_avec_contexte(query, analyse)
        elif analyse['sections'] or analyse['sous_sections']:
            # Recherche de sections/sous-sections
            return self._recherche_sections_sous_sections(query, analyse)
        else:
            # Recherche contextuelle générale
            return self._recherche_contextuelle(query, analyse)
    
    def _analyser_requete_hierarchique(self, query: str) -> Dict:
        """Analyse complète de la requête pour extraire tous les éléments hiérarchiques"""
        query_lower = query.lower()
        
        analyse = {
            'sections': [],
            'sous_sections': [],
            'articles': [],
            'contextes': [],
            'mots_cles': []
        }
        
        # Détection des sections
        for pattern in self.patterns_hierarchie['sections']:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                analyse['sections'].append(match.upper())
        
        # Détection des sous-sections
        for pattern in self.patterns_hierarchie['sous_sections']:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                analyse['sous_sections'].append(match.upper())
        
        # Détection des articles
        for pattern in self.patterns_hierarchie['articles']:
            matches = re.findall(pattern, query_lower)
            analyse['articles'].extend(matches)
        
        # Détection des contextes
        for pattern in self.patterns_hierarchie['contextes']:
            matches = re.findall(pattern, query_lower)
            analyse['contextes'].extend(matches)
        
        # Extraction des mots-clés
        mots_exclus = {'que', 'dit', 'du', 'de', 'la', 'le', 'les', 'des', 'article', 'section', 'sous', 'sous-section'}
        mots = [mot for mot in query_lower.split() if mot not in mots_exclus and len(mot) > 2]
        analyse['mots_cles'] = mots
        
        return analyse
    
    def _recherche_articles_avec_contexte(self, query: str, analyse: Dict) -> Dict:
        """Recherche d'articles avec prise en compte du contexte hiérarchique"""
        article_num = analyse['articles'][0]  # Premier article détecté
        
        # Construction de requêtes enrichies hiérarchiquement
        requetes_enrichies = [
            query,  # Requête originale
            f"Article {article_num}",  # Article direct
        ]
        
        # Ajouter le contexte hiérarchique si présent
        if analyse['sections']:
            section = analyse['sections'][0]
            requetes_enrichies.append(f"Section {section} Article {article_num}")
            
        if analyse['sous_sections']:
            sous_section = analyse['sous_sections'][0]
            requetes_enrichies.append(f"Sous-section {sous_section} Article {article_num}")
            
        if analyse['contextes']:
            for contexte in analyse['contextes']:
                requetes_enrichies.append(f"Article {article_num} {contexte}")
        
        # Recherche avec scores hiérarchiques
        all_results = []
        
        for requete in requetes_enrichies:
            try:
                # Utiliser la recherche de base mais avec scoring avancé
                result = self.base_client.search_specific_article(requete)
                
                if result.get('references'):
                    for ref in result['references']:
                        # Score hiérarchique avancé
                        score_hierarchique = self._calculer_score_hierarchique(
                            ref, article_num, analyse
                        )
                        
                        ref['score_hierarchique'] = score_hierarchique
                        ref['requete_utilisee'] = requete
                        all_results.append(ref)
                        
                        logger.info(f"✅ Résultat trouvé (score hier: {score_hierarchique}) pour: {requete}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Erreur recherche: {e}")
                continue
        
        if not all_results:
            return {"context": "", "references": [], "analyse": analyse}
        
        # Trier par score hiérarchique
        all_results.sort(key=lambda x: x.get('score_hierarchique', 0), reverse=True)
        
        # Prendre les 5 meilleurs résultats
        best_results = all_results[:5]
        
        # Construire le contexte
        context = self._construire_contexte_hierarchique(best_results, analyse)
        
        return {
            "context": context,
            "references": best_results,
            "analyse": analyse,
            "type_recherche": "articles_avec_contexte"
        }
    
    def _recherche_sections_sous_sections(self, query: str, analyse: Dict) -> Dict:
        """Recherche spécialisée pour les sections et sous-sections"""
        requetes_sections = []
        
        # Construction des requêtes pour sections
        if analyse['sections']:
            for section in analyse['sections']:
                requetes_sections.extend([
                    f"Section {section}",
                    f"SECTION {section}",
                    query  # Requête originale
                ])
        
        # Construction des requêtes pour sous-sections  
        if analyse['sous_sections']:
            for sous_section in analyse['sous_sections']:
                requetes_sections.extend([
                    f"Sous-section {sous_section}",
                    f"Sous -section {sous_section}",  # Avec espace (format des docs)
                    query
                ])
        
        all_results = []
        
        for requete in requetes_sections:
            try:
                result = self.base_client.search_specific_article(requete)
                
                if result.get('references'):
                    for ref in result['references']:
                        # Score spécialisé pour sections
                        score_section = self._calculer_score_section(ref, analyse)
                        ref['score_section'] = score_section
                        ref['requete_utilisee'] = requete
                        all_results.append(ref)
                        
                        logger.info(f"✅ Section/Sous-section trouvée (score: {score_section})")
                        
            except Exception as e:
                continue
        
        if not all_results:
            return {"context": "", "references": [], "analyse": analyse}
        
        # Trier par score de section
        all_results.sort(key=lambda x: x.get('score_section', 0), reverse=True)
        best_results = all_results[:5]
        
        context = self._construire_contexte_sections(best_results, analyse)
        
        return {
            "context": context,
            "references": best_results,
            "analyse": analyse,
            "type_recherche": "sections_sous_sections"
        }
    
    def _recherche_contextuelle(self, query: str, analyse: Dict) -> Dict:
        """Recherche contextuelle directe et simple"""
        try:
            # Recherche vectorielle directe avec la requête originale
            query_embedding = self.base_client.generate_embeddings(query)
            if not query_embedding or not self.base_client.collection:
                return {"context": "", "references": [], "analyse": analyse}
            
            results = self.base_client.collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'][0]:
                return {"context": "", "references": [], "analyse": analyse}
            
            all_results = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                
                ref = {
                    'article_ref': metadata.get('source', 'Document juridique'),
                    'snippet': doc[:500],
                    'metadata': metadata,
                    'distance': results['distances'][0][i] if results.get('distances') and i < len(results['distances'][0]) else 1.0,
                    '_score': 1 - results['distances'][0][i] if results.get('distances') and i < len(results['distances'][0]) else 0
                }
                
                score_contextuel = self._calculer_score_contextuel(ref, analyse)
                ref['score_contextuel'] = score_contextuel
                all_results.append(ref)
                logger.info(f"✅ Résultat contextuel trouvé (score: {score_contextuel}) pour: {query}")
            
            # Tri par score et sélection des meilleurs
            all_results.sort(key=lambda x: x.get('score_contextuel', 0), reverse=True)
            best_results = all_results[:5]
            
            context = self._construire_contexte_general(best_results, analyse)
            
            return {
                "context": context,
                "references": best_results,
                "analyse": analyse,
                "type_recherche": "contextuelle"
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur recherche contextuelle: {e}")
            return {"context": "", "references": [], "analyse": analyse}
    
    def _calculer_score_hierarchique(self, ref: Dict, article_num: str, analyse: Dict) -> int:
        """Calcule un score hiérarchique avancé pour un résultat"""
        score = 0
        
        article_ref = ref.get('article_ref', '').lower()
        snippet = ref.get('snippet', '').lower()
        text_complet = f"{article_ref} {snippet}".lower()
        
        # Score de base pour l'article
        if f"article {article_num}" in text_complet:
            score += 500  # Score de base très élevé
            
            # Vérification exacte du numéro d'article
            import re
            article_match = re.search(rf'article\s+{re.escape(article_num)}(?:\s|\.|\,|$)', text_complet)
            if article_match:
                score += 200  # Bonus pour correspondance exacte
        
        # Bonus pour contexte hiérarchique
        for section in analyse.get('sections', []):
            if f"section {section.lower()}" in text_complet:
                score += 150
                
        for sous_section in analyse.get('sous_sections', []):
            if f"sous-section {sous_section.lower()}" in text_complet or f"sous -section {sous_section.lower()}" in text_complet:
                score += 100
        
        # Bonus pour contextes thématiques
        for contexte in analyse.get('contextes', []):
            contexte_clean = contexte.lower().replace('(', '').replace(')', '').replace('?', '')
            if contexte_clean in text_complet:
                score += 80
        
        # Bonus pour mots-clés
        for mot_cle in analyse.get('mots_cles', []):
            if mot_cle in text_complet:
                score += 20
        
        # Pénalité pour faux positifs
        if f"article {article_num}" not in text_complet:
            score -= 200  # Forte pénalité si pas le bon article
        
        return score
    
    def _calculer_score_section(self, ref: Dict, analyse: Dict) -> int:
        """Calcule un score spécialisé pour les sections et sous-sections"""
        score = 0
        
        article_ref = ref.get('article_ref', '').lower()
        snippet = ref.get('snippet', '').lower()
        text_complet = f"{article_ref} {snippet}".lower()
        
        # Score pour sections
        for section in analyse.get('sections', []):
            if f"section {section.lower()}" in text_complet:
                score += 300
                
        # Score pour sous-sections
        for sous_section in analyse.get('sous_sections', []):
            patterns = [
                f"sous-section {sous_section.lower()}",
                f"sous -section {sous_section.lower()}"
            ]
            for pattern in patterns:
                if pattern in text_complet:
                    score += 250
                    break
        
        # Bonus pour contextes
        for contexte in analyse.get('contextes', []):
            contexte_clean = contexte.lower().replace('(', '').replace(')', '').replace('?', '')
            if contexte_clean in text_complet:
                score += 50
        
        return score
    
    def _calculer_score_contextuel(self, ref: Dict, analyse: Dict) -> int:
        """Calcule un score contextuel simple et direct"""
        score = 0
        
        article_ref = ref.get('article_ref', '').lower()
        snippet = ref.get('snippet', '').lower()
        text_complet = f"{article_ref} {snippet}".lower()
        
        # Score basé sur la similarité vectorielle (distance inversée)
        distance = ref.get('distance', 1.0)
        vector_score = max(0, 1.0 - distance)  # Plus proche = score plus élevé
        score += int(vector_score * 500)
        
        # Bonus pour mots-clés présents
        for mot_cle in analyse.get('mots_cles', []):
            if mot_cle.lower() in text_complet:
                score += 25
        
        # Bonus pour contextes détectés
        for contexte in analyse.get('contextes', []):
            if contexte.lower() in text_complet:
                score += 100
        
        return score
    
    def _construire_contexte_hierarchique(self, results: List[Dict], analyse: Dict) -> str:
        """Construit un contexte hiérarchique structuré"""
        if not results:
            return ""
        
        best_result = results[0]
        contexte = f"📋 RÉSULTAT HIÉRARCHIQUE:\n"
        contexte += f"🎯 Article: {analyse.get('articles', ['N/A'])[0]}\n"
        
        if analyse.get('sections'):
            contexte += f"📂 Section: {analyse['sections'][0]}\n"
            
        if analyse.get('sous_sections'):
            contexte += f"📁 Sous-section: {analyse['sous_sections'][0]}\n"
            
        if analyse.get('contextes'):
            contexte += f"🏷️ Contexte: {', '.join(analyse['contextes'])}\n"
        
        contexte += f"\n📖 CONTENU:\n{best_result.get('snippet', '')[:500]}..."
        
        return contexte
    
    def _construire_contexte_sections(self, results: List[Dict], analyse: Dict) -> str:
        """Construit un contexte spécialisé pour les sections"""
        if not results:
            return ""
        
        best_result = results[0]
        contexte = f"📂 SECTION/SOUS-SECTION TROUVÉE:\n"
        
        if analyse.get('sections'):
            contexte += f"Section: {', '.join(analyse['sections'])}\n"
            
        if analyse.get('sous_sections'):
            contexte += f"Sous-section: {', '.join(analyse['sous_sections'])}\n"
        
        contexte += f"\n📖 CONTENU:\n{best_result.get('snippet', '')[:500]}..."
        
        return contexte
    
    def _construire_contexte_general(self, results: List[Dict], analyse: Dict) -> str:
        """Construit un contexte général"""
        if not results:
            return ""
        
        best_result = results[0]
        contexte = f"🔍 RECHERCHE CONTEXTUELLE:\n"
        
        if analyse.get('contextes'):
            contexte += f"Thème: {', '.join(analyse['contextes'])}\n"
            
        if analyse.get('mots_cles'):
            contexte += f"Mots-clés: {', '.join(analyse['mots_cles'][:5])}\n"
        
        contexte += f"\n📖 CONTENU:\n{best_result.get('snippet', '')[:500]}..."
        
        return contexte

def test_systeme_hierarchique():
    """Test du système hiérarchique complet"""
    
    print("🧪 TEST SYSTÈME HIÉRARCHIQUE JURIDIQUE COMPLET V2.0")
    print("=" * 70)
    
    try:
        client = HierarchieJuridiqueClient()
        print("✅ Client hiérarchique initialisé\n")
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return
    
    # Tests hiérarchiques complets
    tests_hierarchiques = [
        {
            "nom": "Article avec contexte hiérarchique complet",
            "requete": "que dit l'article 7 du benefices imposables du determination du benefice net imposable",
            "type": "article_contexte"
        },
        {
            "nom": "Article simple", 
            "requete": "Article 15 rémunérations",
            "type": "article_simple"
        },
        {
            "nom": "Section spécifique",
            "requete": "Section II commission spéciale",
            "type": "section"
        },
        {
            "nom": "Sous-section spécifique",
            "requete": "Sous-section 1 enregistrement",
            "type": "sous_section"
        },
        {
            "nom": "Recherche contextuelle",
            "requete": "personnes imposables période imposition",
            "type": "contextuel"
        }
    ]
    
    for i, test in enumerate(tests_hierarchiques, 1):
        print(f"📝 TEST {i}/{len(tests_hierarchiques)}: {test['nom']}")
        print(f"🔍 Requête: {test['requete']}")
        print("-" * 60)
        
        try:
            result = client.recherche_hierarchique_complete(test['requete'])
            
            print(f"📊 Type de recherche: {result.get('type_recherche', 'inconnu')}")
            print(f"📋 Analyse: {result.get('analyse', {})}")
            
            if result.get('references'):
                print(f"✅ {len(result['references'])} résultat(s) trouvé(s)")
                
                # Afficher le meilleur résultat
                best = result['references'][0]
                print(f"🏆 MEILLEUR RÉSULTAT:")
                print(f"   Titre: {best.get('article_ref', 'N/A')}")
                print(f"   Score: {best.get('score_hierarchique', best.get('score_section', best.get('score_contextuel', 0)))}")
                print(f"   Extrait: {best.get('snippet', '')[:100]}...")
            else:
                print("❌ Aucun résultat trouvé")
                
            print(f"\n📖 Contexte généré:")
            print(f"{result.get('context', 'Aucun contexte')[:200]}...")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    test_systeme_hierarchique()