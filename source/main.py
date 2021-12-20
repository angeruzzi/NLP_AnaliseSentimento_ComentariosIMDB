import loader as ld

if __name__ == '__main__':

 	#Teste Carregamento dos Arquivos
	#path = '../corpus_teste'
	#texto = ld.loadFiles(path, -1, True)
	#print(texto)

	#Teste Carregamento dos dados de Treino
	dfTreino, dfTeste = ld.loadTrain()
	print(dfTreino.head())
	print(dfTeste.head())

	#Teste Carregamento dos dados Para Predição
	#dfPred = ld.loadPredict()
	#print(dfPred.head())
	