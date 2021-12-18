# NLP - Análise de Sentimento de Comentarios de Filmes do IMDB

Este projeto consiste na implementação de um modelo de machine learning que possa avaliar comentários em inglês de filmes e classificá-los como negativos ou positivos.

## Sobre os dados utilizados

Para treinamento do modelo foi utilizado um dataset disponibilizado pelo pesquisador Andrew Maas ( https://ai.stanford.edu/~amaas/ ) no link http://ai.stanford.edu/~amaas/data/sentiment/ . Estes dados foram utilizados originalmente no artigo "Learning Word Vectors for Sentiment Analysis" ( https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf).

Foi disponibilizado 50 mil avaliações de filmes coletadas do site IMDB (https://www.imdb.com/), dentre elas 25 mil positivas e 25 mil negativas, sendo no máximo 30 avaliações do mesmo filme, visto que as críticas para o mesmo filme tendem a ter classificações próximas.

A classificação fornecida foi baseado nas notas de 1 a 10 que acompanhavam os comentários, sendo notas menores ou iguais a 4 consideradas como negativas e notas maiores ou iguais a 7 positivas. Os comentários com notas 5 e 6 não foram considerados para se evitar avaliações neutras.

Na pasta disponibilizada há dois diretórios, [train] e [test], e em cada um contém outros dois diretórios, [pos] e [neg], que contêm os arquivos sendo um por avaliação.

O titulo de cada arquivo segue a seguinte convenção: [[id] _ [rating] .txt] , onde [id] é um id único e [rating] é a avaliação com estrelas do ocmentário em uma escala de 1 a 10.

Além dos arquivos de comentários, foram incluidos arquivos de Bag of words (BoW) no formato LIBSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/), que foram utilizados nos seus experimentos, e um arquivo com a classificação esperada de algumas palavras já calculadas por Potts,2011.
