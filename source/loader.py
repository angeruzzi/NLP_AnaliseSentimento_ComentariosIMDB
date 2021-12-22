import pandas as pd 
from os import walk

def loadTrain(path_files):

	#Recupera dados disponíveis para Treino 
	#target: Informa -1 para as opiniões negativas e 1 para as positivas
	#takeScore: True Solicita que seja recuperado o score do comentário
	dfTrainNeg = loadFiles(path_files+'/train/neg', target=-1, takeScore=True)
	dfTrainPos = loadFiles(path_files+'/train/pos', target= 1, takeScore=True)	

	#Concatena as linhas dos 2 dataframes em apenas um
	dfTrain = pd.concat([dfTrainNeg, dfTrainPos], axis=0)

	#Retorna o Dataframe de Treino
	return dfTrain

def loadTest(path_files):
	#Faz o load dos corpus disponíveis para teste

	#Recupera dados disponíveis para Teste
	#target: Informa -1 para as opiniões negativas e 1 para as positivas
	#takeScore: False - Não solicita o score do comentário
	dfTestNeg = loadFiles(path_files+'/test/neg', target=-1, takeScore=False)
	dfTestPos = loadFiles(path_files+'/test/pos', target= 1, takeScore=False)

	#Concatena as linhas dos 2 dataframes em apenas um
	dfTest = pd.concat([dfTestNeg, dfTestPos], axis=0)

	#Retorna o Dataframe de Teste
	return dfTest

def loadPredict(path_files):
	#Faz o load dos corpus disponíveis para treino e teste

	#Recupera dados que serão utilizados para Predição
	#Target: 0 , Sem Predição prévia
	#takeScore: False - Não solicita o score do comentário
	dfPredict = loadFiles(path_files, target=0, takeScore=False)

	#Retorna o dataFrame gerado
	return dfPredict


def loadFiles(path_files, target=0, takeScore=False):

	files = []
	list_content = []

	#Percorre o diretório informado e gera uma lista dos arquivos
	for (dirpath, dirnames, filenames) in walk(path_files):
	    files.extend(filenames)
	    break

	#Percorre a lista dos arquivos
	for arquivo in files:
	    name_file = arquivo
	    path_file = path_files + '/' + name_file

	    #Abre o arquivo em mode de leitura
	    with open(path_file, 'r', encoding='utf-8') as txt_file:
	    	#Lê o texto do arquivo
	        text  = txt_file.read()
	        score = ''

	        #Caso solicitado recupera o score do nome do arquivo
	        if takeScore:
	        	score = name_file[name_file.index('_')+1:name_file.index('.')]

	        #Adiciona os dados do arquivo na lista
	        list_content.append([name_file, text, score, target])

	#Gera um dataframe com a lista      
	df = pd.DataFrame( list_content, columns=['name', 'text', 'score', 'target'])

	return df