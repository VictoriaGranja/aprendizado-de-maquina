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
print(stop_words)

def tokenizando(review):
    tokens = nlp(review)
    tokens = [ word.lemma_.lower().strip() for word in tokens ]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in pontuacao]
    return tokens

df = pd.read_csv(r'.\database\olist.csv')

print(df.rating.value_counts())

df['review_tokenizada'] = [tokenizando(review) for review in df.review_text]
print(df)