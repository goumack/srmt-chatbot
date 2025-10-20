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
                                n_results=8,
                                include=['documents', 'metadatas', 'distances']
                            )
                            
                            if results['documents'][0]:
                                for i, doc in enumerate(results['documents'][0]):
                                    metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                                    
                                    # Score intelligent basé sur la correspondance naturelle
                                    score = self._calculate_natural_score(doc, metadata, article_num, query.lower())
                                    
                                    if score > 5:  # Seuil de pertinence
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
            best_results = all_results[:3]
            
            # Construire la réponse
            context_parts = []
            references = []
            
            for result in best_results:
                doc = result['document']
                metadata = result['metadata']
                
                reference = {
                    'file_name': metadata.get('file_name', 'Document'),
                    'article_ref': metadata.get('article_ref', f'Article {article_numbers[0]}'),
                    'page': metadata.get('page_start', 1),
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
        
        # Article exact trouvé
        if f"article {article_num}" in doc_lower:
            score += 15
        
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
            'fiscal': ['fiscal', 'fiscale', 'impot', 'impôt']
        }
        
        for concept, terms in important_concepts.items():
            if any(term in query_lower for term in terms):
                concept_matches = sum(1 for term in terms if term in doc_lower)
                if concept_matches > 0:
                    score += concept_matches * 6
        
        # Bonus pour la présence de structure
        if any(struct in doc_lower for struct in ['section', 'sous-section', 'chapitre']):
            score += 3
        
        return score