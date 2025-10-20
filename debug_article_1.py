#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import avec le nom correct du fichier
import importlib.util
spec = importlib.util.spec_from_file_location("srmt_module", "boutton memoire nouveau .py")
srmt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srmt_module)

def debug_article_1():
    """Debug spécifique pour Article 1"""
    print("🔍 DEBUG ARTICLE 1")
    print("=" * 50)
    
    try:
        srmt = srmt_module.SrmtDocumindClient()
        
        # Recherche large pour Article 1
        result = srmt.search_specific_article("Article 1")
        
        print(f"📊 Résultats trouvés: {len(result.get('references', []))}")
        
        if result.get('references'):
            for i, ref in enumerate(result['references'][:10]):  # Top 10
                print(f"\n🔎 Résultat {i+1}:")
                print(f"   📄 Article: {ref.get('article_ref', 'Non identifié')}")
                print(f"   🎯 Score: {ref.get('_score', 0)}")
                print(f"   📝 Aperçu: {ref.get('snippet', '')[:100]}...")
        else:
            print("❌ Aucun résultat trouvé")
            
            # Essayons une recherche plus large
            print("\n🔍 Recherche alternative...")
            alternative_result = srmt.search_context_with_references("Article 1", 10)
            
            if alternative_result.get('references'):
                print(f"📊 Résultats alternatifs: {len(alternative_result['references'])}")
                for i, ref in enumerate(alternative_result['references'][:5]):
                    print(f"\n🔎 Alt {i+1}:")
                    print(f"   📄 Article: {ref.get('article_ref', 'Non identifié')}")
                    print(f"   📝 Aperçu: {ref.get('snippet', '')[:100]}...")
            else:
                print("❌ Aucun résultat alternatif")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_article_1()