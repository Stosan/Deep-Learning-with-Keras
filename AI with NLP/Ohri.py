import tensorflow as tf
import nltk
import speech_recognition as sr
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
import datetime
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder

import json
import speech_recognition as sr
import numpy as np

import pyttsx3
import pickle
import random

stemmer = LancasterStemmer()

# open the dataset
with open('./sethead.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

enc = LabelEncoder()
enc.fit(training_labels)
training_labels = enc.transform(training_labels)

vocab_size = 1000
enbedding_dim = 16
max_len = 28
trunc_type = 'post'
oov_token = "<00v>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, truncating=trunc_type, maxlen=max_len)
#"""
classes = len(labels)

model = keras.Sequential()

model.add(layers.Embedding(vocab_size, enbedding_dim, input_length=max_len))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(100, activation="relu")) #hidden layer1
model.add(layers.Dense(100, activation="relu")) #hidden layer2
model.add(layers.Dense(100, activation="relu")) #hidden layer3
model.add(layers.Dense(100, activation="relu")) #hidden layer4
model.add(layers.Dense(100, activation="relu")) #hidden layer5
model.add(layers.Dense(100, activation="relu")) #hidden layer6
model.add(layers.Dense(100, activation="relu")) #hidden layer7
model.add(layers.Dense(100, activation="relu")) #hidden layer8
model.add(layers.Dense(100, activation="relu")) #hidden layer8
model.add(layers.Dense(100, activation="relu")) #hidden layer8
model.add(Flatten()) #connect every node of the previous layer to the node of the next layer
model.add(layers.Dense(100, activation="relu")) #hidden layer1
model.add(layers.Dense(classes, activation="softmax")) #output layer

model.summary()

training_labels_final = np.array(training_labels)

try:
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.fit(padded, training_labels_final, batch_size=12, epochs=900)
    model.save('model.h5')
    print('model trained and saved!')
except FileNotFoundError:
    print('error with neuron')
    #model = load_model(model.h5)
"""
try:
    model = load_model('model.h5')
    model.summary()
except FileNotFoundError:
    print("model wasn't loaded")
#"""
#put chatbot to talking
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')

engine.setProperty('voice', voices[2].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def Intro1():
    hour = int(datetime.datetime.now().hour)
    if hour >=0 and hour<12:
        speak("Good Morning")
    if hour >=12 and hour<18:
        speak("Good Afternoon")
    if hour >=18 and hour<23:
        speak("Good Evening")
    speak("I'm Riah, I am an artificial intelligence, and i love stem education. To activate our conversation, enter okay, in the console")


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening...")
        # print("Recognising...")
        audio = r.listen(source)
        r.pause_threshold = 1

    try:
        print("Recognising...")
        query = r.recognize_google(audio, language='en-US')
        print(f"You said: {query}\n")

    except Exception as e:
        print(e)
        print("come again please")
        return "None"
    return query
#"""
def check():
    Intro1()
    #speak('Type your question in the console')

    string = input("Enter: ")
    while True:



        """
        string = input('Enter: ')
        print("I'm Riah, I am an artificial intelligence and i love stem education.\ntell me what you'll like to know")
        if string == 'quit': break
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([string]),
                                             truncating=trunc_type, maxlen=max_len))
        category = enc.inverse_transform([np.argmax(result)])
        # processing the output to console
            userthought = string
            AIthought = category
        for i in data['intents']:
            if i['tag'] == category:
                saying = (np.random.choice(i['responses']))
                print("AI: ", saying)
                print(category)
                if AIthought == "userchoice" and userthought == "ok":
                    if i['tag'] == "stemeducation":
                        saying = (np.random.choice(i['responses']))
                        print("AI: ", saying)
                if AIthought == "stemeducation" and userthought == "ok" or "no":
                    if i['tag'] == "engage":
                        saying = (np.random.choice(i['responses']))
                        print("AI: ", saying)
                if AIthought == "stemquestion" and userthought == "ok" or "oh":
                    if i['tag'] == "stemeducation":
                        saying = (np.random.choice(i['responses']))
                        print("AI: ", saying)


 """
        vocals = takeCommand().lower()
        if string == 'quit': break
        elif string == 'okay' or string == 'ok':
            vocals = takeCommand().lower()
            result = model.predict(pad_sequences(tokenizer.texts_to_sequences([vocals]),
                                                     truncating=trunc_type, maxlen=max_len))
            category = enc.inverse_transform([np.argmax(result)])
            userthought = vocals
            AIthought = category
            for i in data['intents']:
                if i['tag'] == category:
                    saying = (np.random.choice(i['responses']))
                    speak(saying)
                    print(category)
                    if AIthought == "userchoice" and (userthought == "ok" or "okay"):
                        if i['tag'] == "stemeducation":
                            saying = (np.random.choice(i['responses']))
                            speak(saying)
                    if AIthought == "stemeducation" and userthought == "ok" or "no":
                        if i['tag'] == "engage":
                            saying = (np.random.choice(i['responses']))
                            speak(saying)
                    if AIthought == "stemquestion" and userthought == "ok" or "oh":
                        if i['tag'] == "stemeducation":
                            saying = (np.random.choice(i['responses']))
                            speak(saying)
#"""
check()
