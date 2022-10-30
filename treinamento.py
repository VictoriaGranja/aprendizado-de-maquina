from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import  GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from scipy.stats import uniform

params = {
    'C':[0.001,0.01,0.1,1,10,100],
    'penalty':['l1','l2']
}

def treino(x, y, df):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    freq_vector = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(df.review_tokenizada)
    x_train = freq_vector.transform(x_train)
    x_test = freq_vector.transform(x_test)
    return x_train, x_test, y_train, y_test

def regressaoLogistica(x, y):
    classificador = LogisticRegression(max_iter=500)
    return classificador.fit(x, y)

def previsao(c, x):
    return c.predict(x)

def metricas(y, y_prev, text):
    print('Métricas de: ', text)
    print("precision_recall_fscore_support", precision_recall_fscore_support(y, y_prev, average='macro'))
    print("F1", f1_score(y, y_prev, average='macro'))

def crossValidation(x_train, y_train):
    print(cross_val_score(LogisticRegression(random_state=42), x_train, y_train, cv=10, verbose=1, n_jobs=-1).mean())

def alteracaoPorGridSearchCV(x, y):
    lr_grid = GridSearchCV(LogisticRegression(random_state=42),params, cv=5, verbose=2, n_jobs=-1)
    return lr_grid.fit(x, y)

def alteracaoPorRandomizedSearchCV(x, y):
    distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
    random = RandomizedSearchCV(LogisticRegression(random_state=42), distributions, random_state=0)
    return random.fit(x, y)