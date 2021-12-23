import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import loader as ld
import cleaner as cl
import vectorizer as vc
import modeler as md

#CONFIGURAÇÕES
PATH_CORPUS  = '../corpus'
PATH_PREDICT = '../topredict' 
TYPE_CONT	 = 'Tfidf' 	 # 'count' or 'Tfidf'
TYPE_MORPHO	 = 'lemma' # 'stemmer' or 'lemma'
N_GRAM		 = (1,2) # (1,1) unigrams , (1,2) unigrams e bigrams
MODELO_BASE  = LinearSVC(C=1.5671297731744318, penalty = 'l2')

def train():

	print("Carregando arquivos de Corpus", flush=True)
	dfTrain   = ld.loadTrain(PATH_CORPUS)
	print("Tratando Dados de treino", flush=True)
	dfTrain   = cl.cleanData(dfTrain, TYPE_MORPHO)
	print("Vetorizando dados e gerando vocabulário", flush=True)	
	bow_train, vet = vc.vectTrain(dfTrain, TYPE_CONT, N_GRAM)
	print("Treinado o modelo", flush=True)
	model = md.trainModel(MODELO_BASE, bow_train, dfTrain['target'], True)
	print("Modelo treinado e salvo", flush=True)

def trainAndTest():

	#Treino
	print("Carregando arquivos de Corpus", flush=True)
	dfTrain   = ld.loadTrain(PATH_CORPUS)
	print("Tratando Dados de treino", flush=True)
	dfTrain   = cl.cleanData(dfTrain, TYPE_MORPHO)
	print("Vetorizando dados e gerando vocabulário", flush=True)	
	bow_train, vet = vc.vectTrain(dfTrain, TYPE_CONT, N_GRAM)
	print("Treinado o modelo", flush=True)	
	modelo_treinado = md.trainModel(MODELO_BASE, bow_train, dfTrain['target'], True)
	print("Modelo treinado e salvo", flush=True)

	#Teste
	print("Carregando arquivos de teste", flush=True)	
	dfTest   = ld.loadTest(PATH_CORPUS)
	print("Tratando Dados de teste", flush=True)
	dfTest   = cl.cleanData(dfTest,TYPE_MORPHO)
	print("Vetorizando dados de teste", flush=True)	
	bow_test = vc.vectTest(dfTest, TYPE_CONT)
	print("Executando testes", flush=True)	
	result   = md.testModel(modelo_treinado, bow_test, dfTest['target'])

	return result
	

def predictSaveModel():

	print("Carregando arquivos para predição", flush=True)
	dfPred   = ld.loadPredict(PATH_PREDICT)
	print("Tratando Dados", flush=True)	
	dfPred   = cl.cleanData(dfPred, TYPE_MORPHO, 'text', 'text_pos')
	print("Vetorizando dados", flush=True)		
	bow_pred = vc.vectPredict(dfPred, TYPE_CONT, 'text_pos')
	print("Executando predições", flush=True)		
	preditos = md.predictSaveModel(bow_pred)

	if preditos is not None:

		dfRet    = dfPred[['name','text']]
		dfRet['prediction'] = preditos
		return dfRet

	return None
	