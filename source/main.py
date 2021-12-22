import pipeline as pp
import interface as itf
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

	while True:
		resposta = itf.menu(['Fazer predição com o modelo já treinado', 'Treinar novamente o modelo', 'Treinar o modelo com execução de testes', 'Sair'])
		if resposta == 1:

			print('Aguarde...')
			ret = pp.predictSaveModel()
			if ret is not None:
				print("\nResultados da Predição :")
				print(ret)

		elif resposta == 2:

			print('Aguarde...')
			ret = pp.train()
			print(ret)

		elif resposta == 3:

			print('Aguarde...')
			ret = pp.trainAndTest()
			print("\nModelo Treinado.")
			print("Score dos Testes Realizados:")
			print(ret)

		elif resposta == 4:

			print('Encerrando')
			break

		else:

			print('Digite uma opção válida')

