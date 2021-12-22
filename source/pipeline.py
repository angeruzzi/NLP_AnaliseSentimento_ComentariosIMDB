import pandas as pd 
from sklearn.linear_model import LogisticRegression

import loader as ld
import cleaner as cl
import vectorizer as vc
import modeler as md

#CONFIGURAÇÕES
PATH_CORPUS  = '../corpus'
PATH_PREDICT = '../topredict' 
TYPE_CONT	 = 'count' 	 # 'count' or 'Tfidf'
TYPE_MORPHO	 = 'stemmer' # 'stemmer' or 'lemma'

def train():

	modelo_base = LogisticRegression()

	dfTrain   = ld.loadTrain(PATH_CORPUS)
	dfTrain   = cl.cleanData(dfTrain, TYPE_MORPHO)
	bow_train = vc.vectTrain(dfTrain, TYPE_CONT)
	md.trainModel(modelo_base, bow_train, dfTrain['target'], True)

	return "Modelo Treinado"


def trainAndTest():

	modelo_base = LogisticRegression()

	#Treino
	dfTrain   = ld.loadTrain(PATH_CORPUS)
	dfTrain   = cl.cleanData(dfTrain, TYPE_MORPHO)
	bow_train = vc.vectTrain(dfTrain, TYPE_CONT)
	modelo_treinado = md.trainModel(modelo_base, bow_train, dfTrain['target'], True)

	#Teste
	dfTest   = ld.loadTest(PATH_CORPUS)
	dfTest   = cl.cleanData(dfTest,TYPE_MORPHO)
	bow_test = vc.vectTest(dfTest, TYPE_CONT)
	result   = md.testModel(modelo_treinado, bow_test, dfTest['target'])

	return result
	

def predictSaveModel():

	dfPred   = ld.loadPredict(PATH_PREDICT)
	dfPred   = cl.cleanData(dfPred, TYPE_MORPHO, 'text', 'text_pos')
	bow_pred = vc.vectPredict(dfPred, TYPE_CONT, 'text_pos')
	preditos = md.predictSaveModel(bow_pred)

	if preditos is not None:

		dfRet    = dfPred[['name','text']]
		dfRet['prediction'] = preditos
		return dfRet

	return None
	