import pandas as pd
import spacy
import nltk
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

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
    tokens = " ".join(tokens)
    return tokens


df = pd.read_csv(r'.\database\olist.csv')

df['review_tokenizada'] = [tokenizando(review) for review in df.review_text]
print(df)

df['review_tokenizada'] = [lematizacao(review) for review in df.review_tokenizada]
print(df)

#df['rating'] = df.rating.replace(1, 1)
#df['rating'] = df.rating.replace(2, 1)
#df['rating'] = df.rating.replace(3, 2)
#df['rating'] = df.rating.replace(4, 3)
#df['rating'] = df.rating.replace(5, 3)

x = df['review_tokenizada']
y = df['rating']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

freq_vector = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(df.review_tokenizada)

x_train = freq_vector.transform(x_train)
x_test = freq_vector.transform(x_test)

classifier = LogisticRegression(max_iter=500)

# model generation
classifier.fit(x_train,y_train)


y_pred_train=classifier.predict(x_train)
precision_recall_fscore_support(y_train, y_pred_train, average='macro')

y_pred=classifier.predict(x_test)
precision_recall_fscore_support(y_test, y_pred, average='macro')


cm=confusion_matrix(y_test, y_pred)

def plot_cm(conf_matrix):
  sns.set(font_scale=1.4,color_codes=True,palette="deep")
  sns.heatmap(cm,annot=True,annot_kws={"size":16},fmt="d",cmap="YlGnBu")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted Value")
  plt.ylabel("True Value")
  plt.show()

plot_cm(cm)