#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script pour simplifier le prompt anti-hallucination trop complexe

def fix_prompt():
    file_path = r"c:\Users\baye.niang\Desktop\Projets et realisations\SRMT CHAT\boutton memoire nouveau .py"
    
    # Lire le fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Identifier le début et la fin du prompt problématique
    start_marker = "🚨 CONSIGNES GÉNÉRALES ANTI-HALLUCINATION :"
    end_marker = "Réponds maintenant en français uniquement et en appliquant CES RÈGLES STRICTEMENT, en particulier la préservation EXACTE ET INTÉGRALE de toutes les valeurs numériques:\"\"\""
    
    # Nouveau prompt simple
    new_prompt = """🇫🇷 Tu es un expert juridique spécialisé dans le droit sénégalais. 

📋 Ta mission est simple :
- Lis le TEXTE OFFICIEL ci-dessus attentivement
- Réponds UNIQUEMENT avec ce qui est écrit dans ce texte
- Cite les articles et références quand tu trouves l'information
- Si tu ne trouves pas l'information, dis-le clairement

Réponds de manière naturelle et précise :\"\"\""""
    
    # Trouver les positions
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)
    
    if start_pos != -1 and end_pos != -1:
        # Calculer la position de fin complète
        end_pos = end_pos + len(end_marker)
        
        # Remplacer le contenu
        new_content = content[:start_pos] + new_prompt + content[end_pos:]
        
        # Écrire le fichier modifié
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Prompt simplifié avec succès !")
        print(f"📏 Ancienne taille: {end_pos - start_pos} caractères")
        print(f"📏 Nouvelle taille: {len(new_prompt)} caractères")
        print(f"📉 Réduction: {((end_pos - start_pos) - len(new_prompt)) / (end_pos - start_pos) * 100:.1f}%")
    else:
        print("❌ Marqueurs non trouvés")
        print(f"Start: {start_pos}")
        print(f"End: {end_pos}")

if __name__ == "__main__":
    fix_prompt()