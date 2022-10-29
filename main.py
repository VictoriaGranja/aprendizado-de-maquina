from tokens import tokenizando, lematizacao
from treinamento import treino, previsao, metricas, crossValidation, regressaoLogistica, alteracaoPorGridSearchCV, alteracaoPorRandomizedSearchCV
from grafico import geraGrafico
import pandas as pd
import numpy as np

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
x_train, x_test, y_train, y_test = treino(x, y, df)

classificador = regressaoLogistica(x_train, y_train)
y_prev_train = previsao(classificador, x_train)
metricas(y_train, y_prev_train)
y_prev = previsao(classificador, x_test)
metricas(y_test, y_prev)
geraGrafico(y_test, y_prev, 'Confusion Matrix')

crossValidation(x_train, y_train)

lr_grid = alteracaoPorGridSearchCV(x_train, y_train)
y_predict = previsao(lr_grid, x_test)
geraGrafico(y_test, y_predict, 'Confusion matrix with Grid search')
metricas(y_test, y_predict)
y_prev = previsao(classificador, x_test)
metricas(y_test, y_prev)
y_prev_train = previsao(classificador, x_train)
metricas(y_train, y_prev_train)

random = alteracaoPorRandomizedSearchCV(x_train, y_train)
y_predict = previsao(random, x_test)
geraGrafico(y_test, y_predict, 'Confusion matrix with Randomized search')
metricas(y_test, y_predict)
y_prev = previsao(classificador, x_test)
metricas(y_test, y_prev)
y_prev_train = previsao(classificador, x_train)
metricas(y_train, y_prev_train)