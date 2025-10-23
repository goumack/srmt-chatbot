"""
Test pour vérifier l'accès à l'API OpenAI
Ce script vous guide pour obtenir une clé API
"""

import webbrowser
import time

def check_api_access():
    print("🔍 VÉRIFICATION ACCÈS API OPENAI")
    print("=" * 50)
    
    print("\n📋 ÉTAPE 1 : Vérifier votre accès API")
    print("1. Je vais ouvrir la page de connexion OpenAI Platform")
    print("2. Connectez-vous avec vos identifiants ChatGPT")
    print("3. Si vous voyez un dashboard → Vous avez accès !")
    print("4. Si erreur → Vous devez créer un compte API")
    
    input("\nAppuyez sur Entrée pour ouvrir OpenAI Platform...")
    webbrowser.open("https://platform.openai.com/login")
    
    print("\n⏳ Attendez que la page se charge...")
    time.sleep(3)
    
    choice = input("\n❓ Voyez-vous un dashboard avec 'Usage', 'API Keys', etc ? (o/n): ").lower()
    
    if choice == 'o':
        print("\n✅ EXCELLENT ! Vous avez accès à l'API")
        print("\n📋 ÉTAPE 2 : Ajouter des crédits")
        print("1. Je vais ouvrir la page de facturation")
        print("2. Cliquez sur 'Add to credit balance'")
        print("3. Ajoutez $5-10 (largement suffisant)")
        
        input("\nAppuyez sur Entrée pour ouvrir la page de facturation...")
        webbrowser.open("https://platform.openai.com/account/billing")
        
        time.sleep(2)
        
        print("\n📋 ÉTAPE 3 : Créer votre clé API")
        print("1. Je vais ouvrir la page des clés API")
        print("2. Cliquez sur 'Create new secret key'")
        print("3. Nommez-la 'LexFin-Integration'")
        print("4. Copiez la clé générée (commence par sk-)")
        
        input("\nAppuyez sur Entrée pour ouvrir la page des clés API...")
        webbrowser.open("https://platform.openai.com/api-keys")
        
        print("\n🎉 Une fois votre clé créée :")
        print("python test_openai_integration.py")
        print("Et collez votre nouvelle clé quand demandé !")
        
    else:
        print("\n📝 PAS DE PROBLÈME ! Créons un compte API")
        print("\n📋 SOLUTION : Compte API séparé")
        print("1. Je vais ouvrir la page d'inscription")
        print("2. Créez un compte (gratuit)")
        print("3. Ajoutez $5 de crédit")
        print("4. Créez votre clé API")
        
        input("\nAppuyez sur Entrée pour ouvrir la page d'inscription...")
        webbrowser.open("https://platform.openai.com/signup")
        
        print("\n💡 ASTUCE : Même email que ChatGPT OK !")
        print("Avoir ChatGPT + API OpenAI = Parfait !")

def estimate_costs():
    print("\n💰 ESTIMATION DES COÛTS")
    print("=" * 30)
    print("Pour LexFin avec GPT-4o-mini :")
    print("• 1 question   ≈ $0.0009 (moins d'1 centime)")
    print("• 100 questions ≈ $0.09")
    print("• 1000 questions ≈ $0.90")
    print("• $5 de crédit  ≈ 5500+ questions")
    print("\n🎯 Recommandation : Commencez avec $5")

if __name__ == "__main__":
    check_api_access()
    estimate_costs()
    
    print("\n" + "=" * 50)
    print("🆘 BESOIN D'AIDE ?")
    print("• Problème de connexion ? Vérifiez votre email/mot de passe")
    print("• Pas de dashboard ? Créez un compte API séparé") 
    print("• Questions ? L'équipe peut vous aider !")
    print("\n🚀 Une fois configuré, LexFin sera ultra-rapide avec OpenAI !")