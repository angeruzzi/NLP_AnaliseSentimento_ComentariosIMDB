import pandas as pd
from datetime import datetime

from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

import selector as sc
import loader as ld
import cleaner as cl
import vectorizer as vc
import modeler as md

import warnings
warnings.filterwarnings('ignore')

#CONFIGURAÇÕES
#PATH_CORPUS  = '../corpus_test'
PATH_CORPUS  = '../corpus'

if __name__ == '__main__':

  model = LinearSVC()

  print("Inicio execução Tuning:", datetime.now(), flush=True)

  #Tratamento Morfologia = 'lemmatização' 
  #Tipo de Contagem de palavras: Tfidf
  #Tratamento Morfologia: Lematização
  #Tipos de Tokens: unigrams e bigrams  

  print("\nTratamento de Dados de Treino:", flush=True)
  
  dfTrain   = ld.loadTrain(PATH_CORPUS)
  print("Loaded", flush=True)

  dfTrain   = cl.cleanData(dfTrain, 'lemma')
  print("Cleaned", flush=True)

  bow_train, vetorizador = vc.vectTrain(dfTrain, 'Tfidf', (1,2))
  print("Vectorized", flush=True)

  matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train, columns=vetorizador.get_feature_names_out())
  print("Count Columns: ", matriz_esparsa.shape[1], flush=True)
 
  print("\n Starting model tuning")

  #distributions = dict(C=uniform(1, 10), gamma=reciprocal(0.001, 0.1), penalty=['l2', 'l1'])
  #distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
  distributions = {"C": uniform(1, 10), "penalty":['l2', 'l1']}

  clf = RandomizedSearchCV(model, distributions, random_state=0, scoring='f1')
  search = clf.fit(bow_train, dfTrain['target'])
  result = search.best_params_

  print(result)

  print("Fim execução Tuning:", datetime.now())  


