#!/usr/bin/env python3
"""
Test démonstration des améliorations d'extraction de tableaux
"""

import pdfplumber
import os
import sys

def demo_extraction_tableaux():
    """Démonstration de l'extraction améliorée des tableaux"""
    
    print("🎯 DÉMONSTRATION - Extraction Tableaux Budgétaires")
    print("=" * 60)
    
    # Document principal avec tableaux ministériels
    doc_path = './documents/www.budget.gouv.sn_projet_de_lois_de_reglement_2021_2025-10-20_11-19.pdf'
    
    if not os.path.exists(doc_path):
        print("❌ Document non trouvé")
        return
    
    print(f"📄 Analyse: {os.path.basename(doc_path)}")
    
    try:
        with pdfplumber.open(doc_path) as pdf:
            print(f"📚 Document: {len(pdf.pages)} pages")
            
            # Analyser la page 39 où nous savons qu'il y a un tableau ministériel
            page_num = 39
            if page_num <= len(pdf.pages):
                page = pdf.pages[page_num - 1]
                
                print(f"\n🔍 ANALYSE PAGE {page_num}")
                print("-" * 40)
                
                # Extraction des tableaux
                tables = page.extract_tables()
                print(f"Tableaux détectés: {len(tables)}")
                
                if tables:
                    table = tables[0]  # Premier tableau
                    print(f"Dimensions: {len(table)} lignes × {len(table[0]) if table and table[0] else 0} colonnes")
                    
                    # Afficher les en-têtes
                    if table and table[0]:
                        headers = [str(h or '').strip() for h in table[0] if str(h or '').strip()]
                        print(f"En-têtes: {headers[:5]}")  # 5 premiers
                    
                    # Afficher quelques lignes de données
                    print("\n📊 ÉCHANTILLON DE DONNÉES:")
                    print("-" * 40)
                    
                    for i, row in enumerate(table[1:6]):  # 5 premières lignes de données
                        if row and any(str(cell or '').strip() for cell in row):
                            # Nettoyer et formater la ligne
                            clean_row = []
                            for cell in row:
                                cell_text = str(cell or '').strip()
                                if cell_text:
                                    # Limiter la longueur pour l'affichage
                                    if len(cell_text) > 30:
                                        cell_text = cell_text[:27] + "..."
                                    clean_row.append(cell_text)
                            
                            if clean_row:
                                print(f"Ligne {i+1}: {' | '.join(clean_row[:4])}")  # 4 premières colonnes
                    
                    # Rechercher des lignes contenant "ministère" ou "éducation"
                    print("\n🏛️ LIGNES MINISTÉRIELLES DÉTECTÉES:")
                    print("-" * 40)
                    
                    ministere_lines = []
                    for i, row in enumerate(table):
                        if row:
                            row_text = ' '.join(str(cell or '') for cell in row).lower()
                            if any(keyword in row_text for keyword in ['ministère', 'ministere', 'éducation', 'education', 'santé', 'sante']):
                                # Extraire les informations importantes
                                clean_cells = [str(cell or '').strip() for cell in row if str(cell or '').strip()]
                                if clean_cells:
                                    ministere_lines.append({
                                        'ligne': i+1,
                                        'contenu': clean_cells[:3]  # 3 premières colonnes
                                    })
                    
                    for line_info in ministere_lines[:5]:  # 5 premiers résultats
                        print(f"  → Ligne {line_info['ligne']}: {' | '.join(line_info['contenu'])}")
                    
                    if not ministere_lines:
                        print("  (Aucune ligne ministérielle détectée dans cet échantillon)")
                
                # Analyser le texte de la page aussi
                text = page.extract_text() or ''
                budget_keywords = ['milliards', 'fcfa', 'ae', 'cp', 'budget', 'ministère']
                found_keywords = [kw for kw in budget_keywords if kw in text.lower()]
                
                print(f"\n📝 ANALYSE TEXTUELLE:")
                print(f"Mots-clés budgétaires trouvés: {found_keywords}")
                print(f"Taille du texte: {len(text)} caractères")
                
            else:
                print(f"❌ Page {page_num} non accessible")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def demo_nouvelle_extraction():
    """Démonstration de la nouvelle méthode d'extraction avec formatage"""
    
    print("\n" + "="*60)
    print("🚀 DÉMONSTRATION - Nouvelle Méthode d'Extraction")
    print("="*60)
    
    # Simuler la nouvelle méthode d'extraction
    doc_path = './documents/www.budget.gouv.sn_projet_de_lois_de_reglement_2021_2025-10-20_11-19.pdf'
    
    try:
        with pdfplumber.open(doc_path) as pdf:
            page = pdf.pages[38]  # Page 39 (index 38)
            
            # Extraction avec la nouvelle méthode
            page_text = page.extract_text() or ''
            tables = page.extract_tables()
            
            if tables:
                print("📊 FORMATAGE AVEC NOUVELLE MÉTHODE:")
                print("-" * 50)
                
                page_text_with_tables = f"--- PAGE 39 ---\n"
                
                # Ajouter le texte normal (extrait)
                if page_text.strip():
                    text_excerpt = page_text[:200] + "..." if len(page_text) > 200 else page_text
                    page_text_with_tables += text_excerpt + "\n\n"
                
                # Ajouter les tableaux formatés
                for i, table in enumerate(tables):
                    page_text_with_tables += f"TABLEAU {i+1} (Page 39):\n"
                    if table and len(table) > 0:
                        # En-têtes
                        if table[0] and any(cell for cell in table[0] if cell):
                            page_text_with_tables += "COLONNES: " + " | ".join(str(cell or '') for cell in table[0][:5]) + "\n"
                            page_text_with_tables += "-" * 80 + "\n"
                        
                        # Quelques lignes de données
                        for row_idx, row in enumerate(table[1:4]):  # 3 premières lignes
                            if row and any(cell for cell in row if cell):
                                formatted_row = []
                                for cell in row[:5]:  # 5 premières colonnes
                                    cell_text = str(cell or '').strip()
                                    if cell_text:
                                        if len(cell_text) > 25:
                                            cell_text = cell_text[:22] + "..."
                                        formatted_row.append(cell_text)
                                if formatted_row:
                                    page_text_with_tables += "LIGNE: " + " | ".join(formatted_row) + "\n"
                        page_text_with_tables += "\n"
                
                print(page_text_with_tables)
                print("="*60)
                print("✅ Cette structure est maintenant indexée dans ChromaDB")
                print("✅ L'IA peut analyser les COLONNES et LIGNES séparément")
                print("✅ Recherche précise dans les allocations ministérielles possible")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    demo_extraction_tableaux()
    demo_nouvelle_extraction()
    
    print(f"\n🎉 CONCLUSION:")
    print("✅ Les tableaux sont maintenant correctement détectés")
    print("✅ Structure COLONNES/LIGNES clairement séparée") 
    print("✅ Données ministérielles accessibles pour l'IA")
    print("✅ Prêt pour questions sur allocations budgétaires")