import pandas as pd 
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, f1_score

def trainModel(modelo_base, X, y, save_model=False):

	modelo = modelo_base
	modelo.fit(X, y)

	if save_model:
		pickle.dump(modelo, open('../models/model.pkl', 'wb'))

	return modelo


def loadSaveModel():

	fNameModel = r"../models/model.pkl"
	fObjModel = Path(fNameModel)

	if not fObjModel.is_file():
		print("O modelo ainda não foi gerado")
		return None

	with open(fNameModel, 'rb') as file: 
		modelo = pickle.load(file)

	return modelo


def testModel(modelo_treinado, X, y):

	pred = modelo_treinado.predict(X)

	score_names = ['Acurácia', 'Sensibilidade (recall)', 'Precisão', 'Especificidade (bac)', 'F1-score']

	acc = accuracy_score(y, pred)
	sen = recall_score(y, pred)
	pre = precision_score(y, pred)
	bac = balanced_accuracy_score(y, pred)
	f1s = f1_score(y, pred)

	df = pd.DataFrame( [[acc,sen,pre,bac,f1s]], columns=score_names)
	return df


def predictSaveModel(matriz):

	modelo = loadSaveModel()
	if modelo:
		pred = modelo.predict(matriz)
		return pred
	return None