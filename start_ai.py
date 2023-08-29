import os
import urllib.request
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import numpy as np
import pickle
import random


class Train:
    def __init__(self, path: str):
        self.path = path

    def data(self, ) -> None:
        filename = self.path
        with open(filename, 'r', encoding="utf8") as f:
            self.dataframe = json.load(f)
        self.X = []
        self.y = []
        for intent in tqdm(self.dataframe, desc='Идет обработка'):
            for example in self.dataframe[intent]['examples']:
                self.X.append(example)
                self.y.append(intent)
            for response in self.dataframe[intent]['responses']:
                self.X.append(response)
                self.y.append(intent)

    def give_vector(self) -> None:
        self.vectorizer = CountVectorizer()
        print('[INFO] Кодирование слов')
        self.vectorizer.fit(self.X)
        self.X_vec = self.vectorizer.transform(self.X)

    def train(self, ):
        if 'network.pkl' in os.listdir():
            print('[INFO] Веса уже есть')
        else:
            print('[INFO] Идет обучение')
            self.model = MLPClassifier()
            self.model.fit(self.X_vec, self.y)
            print('[INFO] Обучение закончилось')
            with open("network.pkl", "wb") as network:
                pickle.dump(self.model, network)
            print('[INFO] Веса сохранены ')

    def load(self, ):
        with open(r'network.pkl', 'rb') as network:
            self.model = pickle.load(network)
            print('[INFO] Готовы отвечать')

    def get_intent(self, text):
        text = self.vectorizer.transform([text])
        self.intent = self.model.predict(text)[0]

    def get_response(self, intent):
        self.response = random.choice(self.dataframe[intent]['responses'])

    def bot(self, text):
        intent = self.get_intent(text)
        self.get_response(intent)
        print(self.response)

class Bot(Train):
    def __init__(self):
        pass


class Doit(Train):
    def __init__(self, path: str) -> None:
        self.path = path

    def all(self, ):
        model = Train(self.path)
        model.data()
        model.give_vector()
        model.train()
        model.load()
        model.bot('Шутка')
