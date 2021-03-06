import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

def CompareModels(Xtrain, ytrain, Xtest, ytest, models_list, models_name):
	score_names = ['Acurácia', 'Sensibilidade (recall)', 'Precisão', 'Especificidade (bac)', 'F1-score']
	results = {}

	for i in range(len(models_list)):

		print("Modelo:", models_name[i], flush=True)
		model = models_list[i]

		model.fit(Xtrain, ytrain)
		predicted = model.predict(Xtest)

		acc = accuracy_score(ytest, predicted)
		sen = recall_score(ytest, predicted)
		pre = precision_score(ytest, predicted)
		bac = balanced_accuracy_score(ytest, predicted)
		f1s = f1_score(ytest, predicted)

		results[models_name[i]] = [acc, sen, pre, bac, f1s]

	compareResults = pd.DataFrame(results, index = score_names).T
	return compareResults