"""
SRMT-DOCUMIND - Optimisation du prompt pour éviter les timeouts

Ce fichier contient un prompt ultra-court pour le modèle Mistral 7B
qui permet des réponses plus rapides en se concentrant uniquement 
sur les articles pertinents trouvés.

Le prompt simplifié a pour objectif de réduire la charge cognitive
du modèle et réduire le risque de timeout.
"""

# Prompt ULTRA-COURT pour éviter les timeouts
OPTIMIZED_PROMPT = """Question: {query}

Articles trouvés:
{mini_context}

Réponds uniquement en te basant sur ces articles. Cite l'article exact.

INSTRUCTIONS STRICTES:
1. Réponds UNIQUEMENT en citant les articles des documents.
2. Ne mélange PAS les articles - cite-les séparément avec leur source.
3. Si tu vois le même numéro d'article de sources différentes (ex: Article 412 dans Code des Impôts ET dans Code des Douanes) - cite les deux séparément.
4. N'invente RIEN - utilise seulement les textes fournis.
5. Dis clairement si aucun document ne répond à la question.
6. Format obligatoire pour chaque article:
   📄 **Article XXX (Source: Code des Impôts/Douanes)**
   "Citation exacte du texte..."

Maintenant analyse les documents et réponds:"""