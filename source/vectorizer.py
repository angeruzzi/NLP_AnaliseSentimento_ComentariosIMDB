
import numpy as np 
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def vectTrain(dfTrain, typeCont='count', name_column='text'):

	#Cria o objeto de vetorização
	if typeCont == 'count':
		vetorizador = CountVectorizer(lowercase=False)
	else: #Tfidf
		vetorizador = TfidfVectorizer(lowercase=False)

	#Vetoriza os dados de Treino e treina o vetorizador
	bow_train 	= vetorizador.fit_transform(dfTrain[name_column])

	#Recupera e salva em um binário o vocabulario gerado
	vocabulary = vetorizador.get_feature_names_out()
	np.save('../models/vocabulary.npy', vocabulary)

	#Salva o modelo Tfidf
	if typeCont == 'Tfidf':
		pickle.dump(tfidf, open("../models/tfidf.pkl", "wb"))

	#Retorna as matrizes esparsas de treino e teste
	return bow_train

def vectTest(dfTest, typeCont='count', name_column='text'):

	#Verifica e recupera os dados do arquivo de vocabulario gerado no treinamento
	fNameVocab = r"../models/vocabulary.npy"
	fObjVocab = Path(fNameVocab)

	if not fObjVocab.is_file():
		print("Por favor realize o treinamento do modelo antes de realizar uma predição")
		return None

	vocabulary = np.load(fNameVocab, allow_pickle=True)

	#Vetoriza os dados de Teste com o vetorizador treinado
	if typeCont == 'count':
		vetorizador = CountVectorizer(vocabulary=vocabulary, lowercase=False)
	else: #Tfidf
		fNameTfidf = r"../models/tfidf.pkl"
		fObjTfidf = Path(fNameTfidf)

		if not fObjTfidf.is_file():
			print("O modelo de vetorização não foi gerado")
			return None

		with open(fNameTfidf, 'rb') as file: 
			vetorizador = pickle.load(file)

	bow_test 	= vetorizador.transform(dfTest[name_column])

	#Retorna as matrizes esparsas de treino e teste
	return bow_test


def vectPredict(dfPredict, typeCont='count', name_column='text'):	

	#Verifica e recupera os dados do arquivo de vocabulario gerado no treinamento
	fNameVocab = r"../models/vocabulary.npy"
	fObjVocab = Path(fNameVocab)

	if not fObjVocab.is_file():
		print("Por favor realize o treinamento do modelo antes de realizar uma predição")
		return None

	vocabulary = np.load(fNameVocab, allow_pickle=True)

	#Vetoriza os dados de predição a partir do vocabulario
	if typeCont == 'count':
		vetorizador = CountVectorizer(vocabulary=vocabulary, lowercase=False)
	else: #Tfidf
		fNameTfidf = r"../models/tfidf.pkl"
		fObjTfidf = Path(fNameTfidf)

		if not fObjTfidf.is_file():
			print("O modelo de vetorização não foi gerado")
			return None

		with open(fNameTfidf, 'rb') as file: 
			vetorizador = pickle.load(file)

	bow_Pred    = vetorizador.transform(dfPredict[name_column])

	return bow_Pred



