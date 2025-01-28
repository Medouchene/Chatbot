from model import ChatbotModel

# Charger le modèle
chatbot = ChatbotModel('intents_ensem.json')
chatbot.load_model()

# Fonction de chat
def chat():
    print("Chatbot: Hi! Type 'quit' to exit.")
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        sub_messages = chatbot.segment_message(message)
        responses = []

        for sub_message in sub_messages:
        # Prédire l'intention
            intents = chatbot.predict_class(sub_message)
            print("Predict class:", intents)

        if intents and float(intents[0]["probability"]) > 0.85:
            # Réponse basée sur les intentions
            response = chatbot.get_response(intents, message)
            print(f"Chatbot: {response}")
        else:
            # Recherche Google si aucune intention fiable n'est trouvée
            print("Chatbot: Je ne sais pas répondre à cela, je vais chercher sur le web...")
            google_response = chatbot.search_google(message)
            print(f"Chatbot: {google_response}")

chat()
