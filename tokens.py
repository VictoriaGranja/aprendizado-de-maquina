import spacy
import nltk
from nltk.corpus import stopwords
import string

#nltk.download() stopword, punkt
pontuacao = string.punctuation
nlp = spacy.load("pt_core_news_sm")
#python -m spacy download pt
stop_words = set(stopwords.words('portuguese'))

def tokenizando(review):
    tokens = nltk.word_tokenize(review)
    tokens = [word.lower().strip(pontuacao) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and word != ""]
    tokens = " ".join(tokens)
    return tokens

def lematizacao(review):
    tokens = nlp(review)
    tokens = [word.lemma_ for word in tokens]
    tokens = " ".join(tokens)
    return tokens
