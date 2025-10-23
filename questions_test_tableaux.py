#!/usr/bin/env python3
"""
Questions de test pour valider l'extraction des tableaux budgétaires
"""

import requests
import json
import time

def test_questions_tableaux():
    """Questions spécifiques pour tester l'extraction de tableaux"""
    
    print("🧪 QUESTIONS DE TEST - Tableaux Budgétaires")
    print("=" * 60)
    
    # Questions ciblées pour tester les capacités d'extraction
    questions_test = [
        {
            "id": 1,
            "question": "Quels sont les programmes du ministère de l'éducation nationale avec leurs codes dans le tableau budgétaire ?",
            "objectif": "Tester extraction colonnes CODE_PROGRAMME et LIBELLE_PROGRAMME",
            "attendu": "Structure COLONNES/LIGNES avec codes spécifiques"
        },
        {
            "id": 2,
            "question": "Affiche-moi le tableau des allocations par ministère avec les montants AE et CP pour 2021",
            "objectif": "Tester extraction données financières structurées",
            "attendu": "Montants en milliards FCFA avec structure tabulaire"
        },
        {
            "id": 3,
            "question": "Dans le document budgétaire, montre-moi les lignes du tableau qui concernent le ministère du travail",
            "objectif": "Tester filtrage intelligent des lignes ministérielles",
            "attendu": "Lignes spécifiques avec 'ministère du travail'"
        },
        {
            "id": 4,
            "question": "Peux-tu extraire les sections et titres budgétaires pour tous les ministères du tableau page 39 ?",
            "objectif": "Tester extraction complète structure hiérarchique",
            "attendu": "SECTION, TITRE organisés par ministère"
        },
        {
            "id": 5,
            "question": "Compare les allocations budgétaires entre le ministère de l'éducation et celui de la santé selon les tableaux",
            "objectif": "Tester analyse comparative des données tabulaires",
            "attendu": "Comparaison chiffrée avec structure claire"
        }
    ]
    
    base_url = "http://127.0.0.1:8505"
    
    # Tester la connexion d'abord
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ Serveur accessible: {response.status_code}")
    except:
        print("❌ Serveur non accessible - Lancez d'abord: python 'boutton memoire nouveau .py'")
        return
    
    # Tester chaque question
    for q in questions_test:
        print(f"\n📋 TEST {q['id']}/5: {q['question'][:60]}...")
        print(f"🎯 Objectif: {q['objectif']}")
        print(f"📊 Attendu: {q['attendu']}")
        
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={"question": q["question"]},
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                reponse = data.get('response', '')
                references = data.get('references', [])
                
                print(f"✅ Réponse reçue: {len(reponse)} caractères")
                
                # Analyse spécifique aux tableaux
                criteres_tableaux = {
                    'Structure COLONNES': 'colonnes:' in reponse.lower(),
                    'Structure LIGNES': 'ligne:' in reponse.lower() or 'lignes:' in reponse.lower(),
                    'Données ministères': any(mot in reponse.lower() for mot in ['ministère', 'ministere', 'éducation', 'santé', 'travail']),
                    'Codes/Programmes': any(mot in reponse.lower() for mot in ['programme', 'code', 'section']),
                    'Montants financiers': any(mot in reponse.lower() for mot in ['milliards', 'fcfa', 'ae', 'cp', 'budget']),
                    'Tableau détecté': 'tableau' in reponse.lower(),
                    'Références valides': len(references) > 0
                }
                
                score = sum(criteres_tableaux.values())
                print(f"📊 Score tableau: {score}/7")
                
                for critere, present in criteres_tableaux.items():
                    status = "✅" if present else "❌"
                    print(f"  {status} {critere}")
                
                # Afficher extrait pertinent
                if len(reponse) > 300:
                    extrait = reponse[:300] + "..."
                else:
                    extrait = reponse
                print(f"📝 Extrait: {extrait}")
                
                # Évaluation spécifique
                if score >= 5:
                    print("🎉 EXCELLENT - Extraction de tableau réussie")
                elif score >= 3:
                    print("✅ BON - Données partiellement structurées")
                else:
                    print("⚠️  LIMITÉ - Amélioration nécessaire")
                    
            else:
                print(f"❌ Erreur HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
        # Pause entre tests
        if q['id'] < len(questions_test):
            print("⏳ Pause 15 secondes...")
            time.sleep(15)
    
    print("\n🏆 TESTS TERMINÉS")
    print("✅ Validez que le système peut extraire:")
    print("  • Structure COLONNES/LIGNES des tableaux")
    print("  • Données ministérielles spécifiques")
    print("  • Codes programmes et sections")
    print("  • Montants budgétaires (AE/CP)")

if __name__ == "__main__":
    test_questions_tableaux()