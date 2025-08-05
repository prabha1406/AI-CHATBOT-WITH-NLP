import json
import random
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def predict_class(sentence):
    sentence_words = clean_up_sentence(sentence)
    for doc in documents:
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        if set(sentence_words) & set(pattern_words):
            return doc[1]
    return None

def get_response(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "I don't understand, please try again."

print("🤖 Chatbot is ready! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    intent = predict_class(message)
    response = get_response(intent)
    print("Bot:", response)