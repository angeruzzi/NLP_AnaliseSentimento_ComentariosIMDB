import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

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
	
	print("Inicio execução:", datetime.now(), flush=True)
	print("\nInstanciando modelos", flush=True)


	models_list = [
		LogisticRegression(max_iter = 10000),
		SVC(max_iter = 10000),
		LinearSVC(),
		SGDClassifier(),	 
		MultinomialNB(),
		DecisionTreeClassifier(),
		RandomForestClassifier()
	]

	models_name = [
		'LogisticRegression',
		'SVC',
		'LinearSVC',
		'SGDClassifier',		
		'MultinomialNB',
		'DecisionTreeClassifier',
		'RandomForestClassifier'		
	]

	#Tratamento 1: 
	#Tratamento Morfologia = 'stemmer'
	print("\nTratamento de Dados de Treino 1:", flush=True)
	
	dfTrain1   = ld.loadTrain(PATH_CORPUS)
	print("Loaded", flush=True)

	dfTrain1   = cl.cleanData(dfTrain1, 'stemmer')
	print("Cleaned", flush=True)

	#Teste
	print("\nDados de Teste:", flush=True)	
	dfTest1   = ld.loadTest(PATH_CORPUS)
	print("Loaded", flush=True)

	dfTest1   = cl.cleanData(dfTest1, 'stemmer')
	print("Cleaned", flush=True)


	# Avaliação 1:
	#Todos os campos
	#Tipo de Contagem de palavras: Count
	#Tratamento Morfologia: Stemming
	#Tipos de Tokens: unigrams
	print("\n Avaliação 1", flush=True)
	bow_train1, vetorizador = vc.vectTrain(dfTrain1, 'count')	
	bow_test1  = vc.vectTest(dfTest1, 'count')
	print("Vectorized", flush=True)

	matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train1, columns=vetorizador.get_feature_names_out())

	print("Count Columns: ", matriz_esparsa.shape[1], flush=True)

	 
	print("\n Starting model comparison")
	result = sc.CompareModels(bow_train1, dfTrain1['target'], bow_test1, dfTest1['target'], models_list, models_name)
	
	print(result)
	
	print("Fim execução 1 :", datetime.now())

	# Avaliação 2:
	#Todos os campos
	#Tipo de Contagem de palavras: Count
	#Tratamento Morfologia: Stemming
	#Tipos de Tokens: unigrams e bigrams
	print("\n Avaliação 2", flush=True)
	bow_train2, vetorizador = vc.vectTrain(dfTrain1, 'count', (1,2))	
	bow_test2  = vc.vectTest(dfTest1, 'count')
	print("Vectorized", flush=True)

	matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train2, columns=vetorizador.get_feature_names_out())

	print("Count Columns: ", matriz_esparsa.shape[1], flush=True)

	 
	print("\n Starting model comparison")
	result = sc.CompareModels(bow_train2, dfTrain1['target'], bow_test2, dfTest1['target'], models_list, models_name)
	
	print(result)
	
	print("Fim execução 2 :", datetime.now())


	# Avaliação 3
	#Todos os campos
	#Tipo de Contagem de palavras: Tfidf
	#Tratamento Morfologia: Stemming
	#Tipos de Tokens: unigrams
	print("\n Avaliação 3", flush=True)
	bow_train3, vetorizador = vc.vectTrain(dfTrain1, 'Tfidf')
	bow_test3  = vc.vectTest(dfTest1, 'Tfidf')
	print("Vectorized", flush=True)

	matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train3, columns=vetorizador.get_feature_names_out())
	print("Count Columns: ", matriz_esparsa.shape[1], flush=True)
 
	print("\n Starting model comparison")
	result = sc.CompareModels(bow_train3, dfTrain1['target'], bow_test3, dfTest1['target'], models_list, models_name)

	print(result)

	print("Fim execução 3:", datetime.now())

	# Avaliação 4
	#Todos os campos
	#Tipo de Contagem de palavras: Tfidf
	#Tratamento Morfologia: Stemming
	#Tipos de Tokens: unigrams e bigrams
	print("\n Avaliação 4", flush=True)
	bow_train4, vetorizador = vc.vectTrain(dfTrain1, 'Tfidf', (1,2))
	bow_test4  = vc.vectTest(dfTest1, 'Tfidf')
	print("Vectorized", flush=True)

	matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train4, columns=vetorizador.get_feature_names_out())
	print("Count Columns: ", matriz_esparsa.shape[1], flush=True)
 
	print("\n Starting model comparison")
	result = sc.CompareModels(bow_train4, dfTrain1['target'], bow_test4, dfTest1['target'], models_list, models_name)

	print(result)

	print("Fim execução 4:", datetime.now())


	#Tratamento 2:
	#Tratamento Morfologia = 'lemmatização'	
	print("\nTratamento de Dados de Treino 2:", flush=True)
	
	dfTrain2   = ld.loadTrain(PATH_CORPUS)
	print("Loaded", flush=True)

	dfTrain2   = cl.cleanData(dfTrain2, 'lemma')
	print("Cleaned", flush=True)

	#Teste
	print("\nDados de Teste:", flush=True)	
	dfTest2   = ld.loadTest(PATH_CORPUS)
	print("Loaded", flush=True)

	dfTest2   = cl.cleanData(dfTest2, 'lemma')
	print("Cleaned", flush=True)


	# Avaliação 5:
	#Todos os campos
	#Tipo de Contagem de palavras: Count
	#Tratamento Morfologia: Lematização
	#Tipos de Tokens: unigrams	
	print("\n Avaliação 5", flush=True)
	bow_train5, vetorizador = vc.vectTrain(dfTrain2, 'count')	
	bow_test5  = vc.vectTest(dfTest2, 'count')
	print("Vectorized", flush=True)

	matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train5, columns=vetorizador.get_feature_names_out())
	print("Count Columns: ", matriz_esparsa.shape[1], flush=True)

	 
	print("\n Starting model comparison")
	result = sc.CompareModels(bow_train5, dfTrain2['target'], bow_test5, dfTest2['target'], models_list, models_name)
	
	print(result)
	
	print("Fim execução 5 :", datetime.now())

	# Avaliação 6:
	#Todos os campos
	#Tipo de Contagem de palavras: Count
	#Tratamento Morfologia: Lematização
	#Tipos de Tokens: unigrams e bigrams	
	print("\n Avaliação 6", flush=True)
	bow_train6, vetorizador = vc.vectTrain(dfTrain2, 'count', (1,2))	
	bow_test6  = vc.vectTest(dfTest2, 'count')
	print("Vectorized", flush=True)

	matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train6, columns=vetorizador.get_feature_names_out())
	print("Count Columns: ", matriz_esparsa.shape[1], flush=True)

	 
	print("\n Starting model comparison")
	result = sc.CompareModels(bow_train6, dfTrain2['target'], bow_test6, dfTest2['target'], models_list, models_name)
	
	print(result)
	
	print("Fim execução 6 :", datetime.now())

	# Avaliação 7
	#Todos os campos
	#Tipo de Contagem de palavras: Tfidf
	#Tratamento Morfologia: Lematização
	#Tipos de Tokens: unigrams	
	print("\n Avaliação 7", flush=True)
	bow_train7, vetorizador = vc.vectTrain(dfTrain2, 'Tfidf')
	bow_test7  = vc.vectTest(dfTest2, 'Tfidf')
	print("Vectorized", flush=True)

	matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train7, columns=vetorizador.get_feature_names_out())
	print("Count Columns: ", matriz_esparsa.shape[1], flush=True)
 
	print("\n Starting model comparison")
	result = sc.CompareModels(bow_train7, dfTrain2['target'], bow_test7, dfTest2['target'], models_list, models_name)

	print(result)

	print("Fim execução 7:", datetime.now())	

	# Avaliação 8
	#Todos os campos
	#Tipo de Contagem de palavras: Tfidf
	#Tratamento Morfologia: Lematização
	#Tipos de Tokens: unigrams e bigrams	
	print("\n Avaliação 8", flush=True)
	bow_train8, vetorizador = vc.vectTrain(dfTrain2, 'Tfidf', (1,2))
	bow_test8  = vc.vectTest(dfTest2, 'Tfidf')
	print("Vectorized", flush=True)

	matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bow_train8, columns=vetorizador.get_feature_names_out())
	print("Count Columns: ", matriz_esparsa.shape[1], flush=True)
 
	print("\n Starting model comparison")
	result = sc.CompareModels(bow_train8, dfTrain2['target'], bow_test8, dfTest2['target'], models_list, models_name)

	print(result)

	print("Fim execução 8:", datetime.now())	