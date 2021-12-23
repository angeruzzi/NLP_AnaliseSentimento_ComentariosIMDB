import pipeline as pp
import interface as itf
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

	while True:
		resposta = itf.menu(['Fazer predição com o modelo já treinado', 'Treinar novamente o modelo', 'Treinar o modelo e executar os testes', 'Sair'])
		if resposta == 1:

			print('\nAguarde...', flush=True)
			ret = pp.predictSaveModel()
			if ret is not None:
				print("\nResultados da Predição :")
				print(" 1 : Comentários Positivos")
				print("-1 : Comentários Negativos")
				print(ret)

		elif resposta == 2:

			print('\nAguarde...', flush=True)
			ret = pp.train()

		elif resposta == 3:

			print('\nAguarde...', flush=True)
			ret = pp.trainAndTest()

			print("\nScore dos Testes Realizados:")
			print(ret)

		elif resposta == 4:

			print('Encerrando')
			break

		else:

			print('Digite uma opção válida')

