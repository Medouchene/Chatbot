from flask import Flask, request, jsonify
from flask_cors import CORS  # Importer Flask-CORS
from model import ChatbotModel

app = Flask(__name__)
CORS(app)  # Activer les CORS pour toutes les routes

# Charger le modèle
chatbot = ChatbotModel('intents_ensem.json')
chatbot.load_model()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"response": "Message manquant"}), 400
    
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
            response = chatbot.search_google(message)
            print(f"Chatbot: {response}")

        responses.append(response)

    return jsonify({"response": " ".join(responses)})

if __name__ == "__main__":
    app.run(debug=True)
