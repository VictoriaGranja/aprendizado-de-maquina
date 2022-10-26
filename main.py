import pandas as pd
import spacy
import nltk
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#nltk.download() stopword, punkt
pontuacao = string.punctuation
nlp = spacy.load("pt_core_news_sm")
#python -m spacy download pt
stop_words = set(stopwords.words('portuguese'))

def tokenizando(review):
    tokens = nltk.word_tokenize(review)
    tokens = [word.lower().strip(pontuacao) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = " ".join(tokens)
    return tokens

def lematizacao(review):
    tokens = nlp(review)
    tokens = [word.lemma_ for word in tokens]
    return tokens


df = pd.read_csv(r'.\database\olist.csv')

df['review_tokenizada'] = [tokenizando(review) for review in df.review_text]
print(df)

df['review_tokenizada'] = [lematizacao(review) for review in df.review_tokenizada]
print(df)
