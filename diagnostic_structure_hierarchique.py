#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic de la structure hiérarchique dans les documents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import avec le nom correct du fichier
import importlib.util
spec = importlib.util.spec_from_file_location("srmt_module", "boutton memoire nouveau .py")
srmt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srmt_module)

def analyser_structure_documents():
    """Analyse la structure hiérarchique réelle des documents"""
    
    print("🔍 ANALYSE DE LA STRUCTURE HIÉRARCHIQUE DES DOCUMENTS")
    print("=" * 70)
    
    try:
        dm = srmt_module.SrmtDocumindClient()
        print()
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return

    # Recherches exploratoires pour comprendre la structure
    recherches_structure = [
        "Section I",
        "Section II", 
        "Section III",
        "Section 1",
        "Section 2",
        "Sous-section A",
        "Sous-section B", 
        "Sous-section 1",
        "Chapitre I",
        "Chapitre II",
        "Titre I",
        "Titre II",
        "Article 7 Période",
        "bénéfices imposables",
        "détermination du bénéfice net imposable",
        "personnes imposables"
    ]
    
    for recherche in recherches_structure:
        print(f"\n🔍 Recherche: '{recherche}'")
        print("-" * 50)
        
        try:
            # Recherche générale (pas spécifique aux articles)
            result = dm.search_specific_article(recherche)
            
            if result.get('references') and len(result['references']) > 0:
                print(f"✅ {len(result['references'])} résultat(s) trouvé(s)")
                
                # Afficher les 3 premiers résultats
                for i, ref in enumerate(result['references'][:3], 1):
                    titre = ref.get('article_ref', 'Non identifié')
                    score = ref.get('_score', 0)
                    snippet = ref.get('snippet', '')[:80] + "..." if len(ref.get('snippet', '')) > 80 else ref.get('snippet', '')
                    
                    print(f"  {i}. {titre} (Score: {score})")
                    print(f"     Extrait: {snippet}")
            else:
                print("❌ Aucun résultat")
                
        except Exception as e:
            print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    analyser_structure_documents()