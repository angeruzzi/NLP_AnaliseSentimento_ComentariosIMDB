import numpy as np 
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def vectTrain(dfTrain, dfTest, name_column='text'):

	#Cria o objeto de vetorização
	vetorizador = CountVectorizer(lowercase=False)
	#Vetoriza os dados de Treino e treina o vetorizador
	bow_train 	= vetorizador.fit_transform(dfTrain[name_column])
	#Vetoriza os dados de Teste com o vetorizador treinado
	bow_test 	= vetorizador.transform(dfTest[name_column])

	#Recupera e salva em um binário o vocabulario gerado
	vocabulary = vetorizador.get_feature_names_out()
	np.save('../models/vocabulary.npy', vocabulary)

	#Retorna as matrizes esparsas de treino e teste
	return bow_train, bow_test


def vectPredict(dfPredict, name_column='text'):	

	#Verifica e recupera os dados do arquivo de vocabulario gerado no treinamento
	fNameVocab = r"../models/vocabulary.npy"
	fObjVocab = Path(fNameVocab)

	if not fObjVocab.is_file():
		print("Por favor realize o treinamento do modelo antes de realizar uma predição")
		#print("Não foi encontrado o arquivo de vocabulário, por favor realize o treinamento do modelo")
		return

	vocabulary = np.load(fNameVocab, allow_pickle=True)

	#Vetoriza os dados de predição a partir do vocabulario
	vetorizador = CountVectorizer(vocabulary=vocabulary, lowercase=False)
	bow_Pred    = vetorizador.transform(dfPredict[name_column])

	return bow_Pred



