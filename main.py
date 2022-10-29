import pandas as pd
import spacy
import nltk
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.stats import uniform

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


df = pd.read_csv(r'.\database\olist.csv')
df = df[['review_text', 'rating']]

df['review_tokenizada'] = [tokenizando(review) for review in df.review_text]

df['review_tokenizada'] = [lematizacao(review) for review in df.review_tokenizada]

df['rating'] = df.rating.replace(1, 1)
df['rating'] = df.rating.replace(2, 1)
df['rating'] = df.rating.replace(3, 2)
df['rating'] = df.rating.replace(4, 3)
df['rating'] = df.rating.replace(5, 3)

x = df['review_tokenizada']
y = df['rating']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

freq_vector = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(df.review_tokenizada)

x_train = freq_vector.transform(x_train)
x_test = freq_vector.transform(x_test)

classificador = LogisticRegression(max_iter=500)
classificador.fit(x_train,y_train)

y_pred_train=classificador.predict(x_train)
print("precision_recall_fscore_support", precision_recall_fscore_support(y_train, y_pred_train, average='macro'))

y_pred=classificador.predict(x_test)
print("precision_recall_fscore_support", precision_recall_fscore_support(y_test, y_pred, average='macro'))

print("f1", f1_score(y_train, y_pred_train, average='macro'))
print("f1", f1_score(y_test, y_pred, average='macro'))

cm=confusion_matrix(y_test, y_pred)

def plot_cm(conf_matrix, title):
  sns.set(font_scale=1.4,color_codes=True,palette="deep")
  sns.heatmap(cm,annot=True,annot_kws={"size":16},fmt="d",cmap="YlGnBu")
  plt.title(title)
  plt.xlabel("Predicted Value")
  plt.ylabel("True Value")
  plt.show()


plot_cm(cm, 'Confusion Matrix')

print(cross_val_score(LogisticRegression(random_state=42), x_train, y_train, cv=10, verbose=1, n_jobs=-1).mean())

params = {
    #'solver':['liblinear','saga','newton-cg','lbfgs'],
    'C':[0.001,0.01,0.1,1,10,100],
    'penalty':['l1','l2']
}

print("grid")
# Grid search for hyper-parametres
lr_grid = GridSearchCV(LogisticRegression(random_state=42),params, cv=5, verbose=2, n_jobs=-1)
lr_grid.fit(x_train, y_train)

y_predict=lr_grid.predict(x_test)
cm=confusion_matrix(y_test, y_predict)
plot_cm(cm, 'Confusion matrix with Grid search')

print(precision_recall_fscore_support(y_test, y_predict, average='macro'))

y_pred = classificador.predict(x_test)
print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

y_pred_train = classificador.predict(x_train)
print(precision_recall_fscore_support(y_train, y_pred_train, average='macro'))

print("random")
# Randomized hyper-parametres
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
random = RandomizedSearchCV(LogisticRegression(random_state=42), distributions, random_state=0)

random.fit(x_train, y_train)
y_predict=random.predict(x_test)

cm=confusion_matrix(y_test, y_predict)
plot_cm(cm, 'Confusion matrix with Randomized search')

print(precision_recall_fscore_support(y_test, y_predict, average='macro'))

y_pred = classificador.predict(x_test)
print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

y_pred_train = classificador.predict(x_train)
print(precision_recall_fscore_support(y_train, y_pred_train, average='macro'))