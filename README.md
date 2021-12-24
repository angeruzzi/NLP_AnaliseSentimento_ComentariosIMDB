## 📽️ NLP - Análise de Sentimento de Comentarios de Filmes do IMDB

Este projeto consiste na implementação de um modelo de predição baseado em Análise de Sentimento, que possa avaliar comentários em inglês de filmes e classificá-los como negativos ou positivos.

Acesse o [relatório do projeto](  https://github.com/angeruzzi/NLP_AnaliseSentimento_ComentariosIMDB/blob/5dfb86c064fd80bfcdf734fa6bbb838570275d55/docs/relatorio_NLP_AnaliseSentimento_ComentariosIMDB.pdf )
para maiores informações.
 
### 📋 Pré-requisitos

Para execução do projeto é necessário que tenha o instalado em seu computador:
* Git: para dowload dos fontes
* Python: para execução; a versão de desenvolvimento deste projeto foi a 3.9.9. 



### 🔧 Instalação

Para a instalação deve-se clonar o repositório deste projeto e fazer a instalação das dependências:
```
# Clone o repositorio
$ git clone https://github.com/angeruzzi/NLP_AnaliseSentimento_ComentariosIMDB.git

# Acesse a pasta do projeto no terminal/cmd e instale o requerimento
$ pip install -r requeriments
```

### ⚙️ Executando o programa

Para executar o programa acesse a pasta de fontes e execute o módulo main

```
#Acesso aos fontes
$ cd source

#Executar o main
$ python main.py
```

Após a inicialização será habilitado um menu com as opções disponíveis

![image](https://user-images.githubusercontent.com/31965992/147369946-7ff18fa1-e2fc-4b9c-bc3d-bbee9559244e.png)

Nas opções que envolvam treino ou teste do modelo o programa irá utilizar os arquivos da pasta "corpus".
Para novas predições devem ser salvos arquivos txt na pasta "topredict", sendo 1 comentário por arquivo; nesta pasta já há alguns arquivos de exemplo. 


