import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from string import punctuation
from nltk import tokenize
from nltk.stem import porter
from nltk import wordnet
from nltk import ngrams
from nltk.corpus import stopwords

def cleanData(df, typeMorpho='stemmer', column_in='text', column_out='text'):

	token_punct = tokenize.WordPunctTokenizer()
	stemmer = porter.PorterStemmer()
	lemma  = wordnet.WordNetLemmatizer()

	#Conjunto de stopwords em inglês
	stopwords_list  = stopwords.words('english')

	#Gera uma lista com caracters de pontuação e outros 
	punctuation_list = list()
	for simbol in punctuation:
		punctuation_list.append(simbol)

	#Concatena as listas stopwords e pontuação
	removal_words = stopwords_list + punctuation_list

	comments_processed = list()
	for comment in df[column_in]:

		comment_clean = list()
		
		#Remove as tags HTML
		regxHtmlTags = re.compile(r'<.*?>')
		re.sub(regxHtmlTags,'',comment)

		#Padroniza o texto para lowercase
		comment = comment.lower()

		#Tokeniza o comentário pelas pontuações e espaços
		words_list = token_punct.tokenize(comment)

		for word in words_list:
			if word not in removal_words:

				if typeMorpho == 'stemmer':
					#Aplica o Stemmer na palavra
					comment_clean.append(stemmer.stem(word))
				else:
					#Aplica a lemmatização na palavra
					comment_clean.append(lemma.lemmatize(word))

		#Adiciona a frase higienizada na lista
		comments_processed.append(' '.join(comment_clean))

	#Substitui os textos originais pelos tratados
	df[column_out] = comments_processed

	return df