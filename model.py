import nltk
#ntlk.download('all')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import warnings
warnings.filterwarnings('ignore')
import requests

class ChatbotModel:
    def __init__(self, intents_file='intents_ensem.json'):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']
        self.intents_file = intents_file
        self.intents = None
        self.model = None
        self.train_x = None
        self.train_y = None

        # Load intents
        with open(self.intents_file, 'r') as file:
            self.intents = json.load(file)
        
        self._preprocess_data()
    def search_google(self, query):
        """
        Recherche une question sur Google à l'aide de Custom Search JSON API.
        """
        API_KEY = "AIzaSyAmxhE11di0CeCE2s7h8ED4MKCtkVzClp0"  #  clé API Google
        SEARCH_ENGINE_ID = "b608b4cda1c65459d"  # ID de moteur de recherche
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={SEARCH_ENGINE_ID}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Extraire les résultats de recherche
            items = data.get('items', [])
            if not items:
                return "Je n'ai trouvé aucun résultat pertinent pour votre recherche."

            # Formater les résultats
            results = []
            for item in items[:3]:  # Limiter aux 3 premiers résultats
                title = item.get('title', 'Titre non disponible')
                link = item.get('link', 'Lien non disponible')
                results.append(f"- {title}: {link}")

            return "\n".join(results)

        except requests.exceptions.RequestException as e:
            return f"Erreur lors de la recherche : {e}"
    def _preprocess_data(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        # documents = combination between patterns and intents
        print (len(self.documents), "documents")
        # classes = intents
        print (len(self.classes), "classes", self.classes)
        # words = all words, vocabulary
        print (len(self.words), "unique lemmatized words", self.words)
        self._create_training_data()

    def _create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)
            
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        self.train_x = list(training[:, 0])
        self.train_y = list(training[:, 1])

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.train_y[0]), activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train_model(self, epochs=200, batch_size=5):
        hist = self.model.fit(np.array(self.train_x), np.array(self.train_y), epochs=epochs, batch_size=batch_size, verbose=1)
        return hist

    def save_model(self, model_file='chatbot_model.h5', words_file='words.pkl', classes_file='classes.pkl'):
        if self.model:
            self.model.save(model_file)
        with open(words_file, 'wb') as wf:
            pickle.dump(self.words, wf)
        with open(classes_file, 'wb') as cf:
            pickle.dump(self.classes, cf)

    def load_model(self, model_file='chatbot_model.h5', words_file='words.pkl', classes_file='classes.pkl'):
        from keras.models import load_model
        self.model = load_model(model_file)
        with open(words_file, 'rb') as wf:
            self.words = pickle.load(wf)
        with open(classes_file, 'rb') as cf:
            self.classes = pickle.load(cf)

    # New methods for predictions and responses
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        return [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    def bow(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print(f"Found in bag: {w}")
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]
    """
    def get_response(self, intents_list):
        tag = intents_list[0]['intent']
        for i in self.intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    
    
    def get_response(self, intents_list, message):
        if intents_list:  # Si une intention est détectée
            tag = intents_list[0]['intent']
            for i in self.intents['intents']:
                if i['tag'] == tag:
                    return random.choice(i['responses'])
        else:  # Aucune intention détectée, effectuer une recherche sur Google
            return self.search_google(message)
    """
    def segment_message(self, message):
        delimiters = [",", "et", "ou", ".", ";"]
        for delimiter in delimiters:
            message = message.replace(delimiter, "|")
        return [segment.strip() for segment in message.split("|") if segment.strip()]

    def get_response(self, intents_list, message):

        # Segmenter le message
        sub_messages = self.segment_message(message)
        responses = []

        for sub_message in sub_messages:
            # Obtenir les intentions pour chaque sous-message
            intents_list = self.predict_class(sub_message)
            if intents_list:  # Si une intention est détectée
                tag = intents_list[0]['intent']
                for i in self.intents['intents']:
                    if i['tag'] == tag:
                        responses.append(random.choice(i['responses']))
            else:  # Aucune intention détectée, effectuer une recherche sur Google
                responses.append(self.search_google(sub_message))

        # Combiner les réponses
        return " ".join(responses)

# Initialiser le modèle
chatbot = ChatbotModel('intents_ensem.json')

# Construire et entraîner le modèle
chatbot.build_model()
chatbot.train_model(epochs=200, batch_size=5)

# Sauvegarder le modèle
chatbot.save_model()